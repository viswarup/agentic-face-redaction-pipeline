"""FastAPI application for Body-Camera Video Face Anonymization & Verification Pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from app.config import config
from app.db import (
    create_or_get_job,
    get_job,
    init_db,
    list_jobs,
    update_job_status,
    update_review_action,
)

logger = logging.getLogger("anonymizer")
logging.basicConfig(level=logging.INFO)


def to_relative_path(path_val: str | Path) -> str:
    """Convert absolute path to project-relative path if within project root."""
    try:
        p = Path(path_val)
        base = config.BASE_DIR.resolve()
        if p.is_absolute():
            resolved = p.resolve()
            if resolved.is_relative_to(base):
                return str(resolved.relative_to(base))
        return str(path_val)
    except Exception:
        return str(path_val)


# Bounded single-worker thread pool for serial, deterministic processing queue
job_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pipeline_worker")


def background_job_worker(job_id: str) -> None:
    """
    Background worker processing video jobs through the full safety pipeline:
    QUEUED -> PROCESSING_REDACTION -> VERIFYING_OUTPUT -> (SAFE | NEEDS_REVIEW).
    """
    logger.info("Background worker started for job_id=%s", job_id)
    job = get_job(job_id)
    if not job:
        return

    input_path_str = job.get("input_path")
    if not input_path_str:
        update_job_status(
            job_id=job_id,
            status="NEEDS_REVIEW",
            failure_reason="MODEL_OR_PROCESSING_ERROR: Input path missing.",
        )
        return

    input_path = Path(input_path_str)
    if not input_path.is_absolute():
        input_path = config.BASE_DIR / input_path

    if not input_path.exists():
        update_job_status(
            job_id=job_id,
            status="NEEDS_REVIEW",
            failure_reason="MODEL_OR_PROCESSING_ERROR: Input file does not exist or is missing.",
        )
        return

    output_path = config.REDACTED_VIDEOS_DIR / f"{job_id}_redacted.mp4"
    rel_output_path = to_relative_path(output_path)
    tracks_json_path = Path(f"{output_path}.tracks.json")

    try:
        # Phase 1: Redaction
        update_job_status(job_id=job_id, status="PROCESSING_REDACTION")
        from app.pipeline.redactor import process_video_redaction

        redaction_telemetry = process_video_redaction(
            input_video_path=input_path,
            output_video_path=output_path,
            conf_thresh=config.PRIMARY_CONF_THRESH,
            dilation_ratio=config.BBOX_DILATION_RATIO,
            interpolation_window=config.INTERPOLATION_WINDOW,
        )

        # Phase 2: Independent Post-Render QA Verification
        update_job_status(
            job_id=job_id,
            status="VERIFYING_OUTPUT",
            output_path=rel_output_path,
            telemetry=redaction_telemetry,
            processing_fps=redaction_telemetry.get("processing_fps", 0.0),
        )

        from app.pipeline.verifier import verify_rendered_video

        verification = verify_rendered_video(
            rendered_mp4_path=output_path,
            tracks_json_path=tracks_json_path,
            conf_thresh=config.VERIFIER_CONF_THRESH,
            stride=config.VERIFIER_STRIDE,
            job_id=job_id,
        )

        final_status = verification.get("status", "NEEDS_REVIEW")
        combined_telemetry = {
            "redaction": redaction_telemetry,
            "verification": verification.get("telemetry", {}),
            "blind_spot_warning": verification.get("blind_spot_warning", False),
            "blind_spot_details": verification.get("blind_spot_details", []),
        }

        update_job_status(
            job_id=job_id,
            status=final_status,
            output_path=rel_output_path,
            failure_reason=verification.get("reason"),
            flagged_frames=verification.get("flagged_frames", []),
            telemetry=combined_telemetry,
            processing_fps=redaction_telemetry.get("processing_fps", 0.0),
        )
        logger.info(
            "Completed job_id=%s with final_status=%s (flagged=%d)",
            job_id,
            final_status,
            len(verification.get("flagged_frames", [])),
        )

    except Exception as e:
        logger.exception("Fail-closed: Unhandled error in background worker for job_id=%s: %s", job_id, e)
        update_job_status(
            job_id=job_id,
            status="NEEDS_REVIEW",
            failure_reason=f"MODEL_OR_PROCESSING_ERROR: {str(e)}",
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.ensure_directories()
    init_db()
    yield
    job_executor.shutdown(wait=False)


app = FastAPI(
    title="Body-Camera Video Face Anonymization & Verification API",
    version="1.0.0",
    lifespan=lifespan,
)


class ReviewActionRequest(BaseModel):
    status: str = Field(..., description="Target status: 'SAFE' override or 'NEEDS_REVIEW' confirmed")
    reviewer_notes: Optional[str] = Field(None, description="Reviewer feedback or notes")


class JobResponse(BaseModel):
    job_id: str
    status: str
    is_duplicate_event: bool
    duplicate_request_count: int
    event_id: str
    file_sha256: str
    input_path: Optional[str] = None
    output_path: Optional[str] = None
    failure_reason: Optional[str] = None
    flagged_frames: list[Any] = Field(default_factory=list)
    telemetry: dict[str, Any] = Field(default_factory=dict)
    processing_fps: float = 0.0
    reviewer_notes: Optional[str] = None
    created_at: str
    updated_at: str


def format_job_record(job: dict[str, Any], is_duplicate_event: Optional[bool] = None) -> dict[str, Any]:
    """Format raw SQLite dictionary row into response-friendly dict with parsed JSON fields."""
    flagged_frames_raw = job.get("flagged_frames_json")
    if isinstance(flagged_frames_raw, str):
        try:
            flagged_frames = json.loads(flagged_frames_raw)
        except Exception:
            flagged_frames = []
    elif isinstance(flagged_frames_raw, list):
        flagged_frames = flagged_frames_raw
    else:
        flagged_frames = []

    telemetry_raw = job.get("telemetry_json")
    if isinstance(telemetry_raw, str):
        try:
            telemetry = json.loads(telemetry_raw)
        except Exception:
            telemetry = {}
    elif isinstance(telemetry_raw, dict):
        telemetry = telemetry_raw
    else:
        telemetry = {}

    dup_count = job.get("duplicate_request_count", 0)
    duplicate_flag = is_duplicate_event if is_duplicate_event is not None else (dup_count > 0)

    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "is_duplicate_event": duplicate_flag,
        "duplicate_request_count": dup_count,
        "event_id": job["event_id"],
        "file_sha256": job["file_sha256"],
        "input_path": job.get("input_path"),
        "output_path": job.get("output_path"),
        "failure_reason": job.get("failure_reason"),
        "flagged_frames": flagged_frames,
        "telemetry": telemetry,
        "processing_fps": float(job.get("processing_fps") or 0.0),
        "reviewer_notes": job.get("reviewer_notes"),
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
    }


@app.post("/jobs", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
async def submit_job(
    event_id: str = Form(..., description="Unique body-camera upload event ID"),
    file: Optional[UploadFile] = File(None, description="Video file payload"),
    file_path: Optional[str] = Form(None, description="Path to local video file (alternative to file upload)"),
):
    """
    Submit a video upload event for face anonymization and verification.
    Submitting the same (event_id, file_sha256) tuple returns the existing job
    with an incremented duplicate_request_count and does NOT re-queue background work.
    """
    if not event_id or not event_id.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="event_id must not be empty.")

    file_bytes: bytes
    saved_input_path: str

    if file is not None:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")
        file_bytes = content
        file_sha256 = hashlib.sha256(file_bytes).hexdigest()

        # Save to video snippets directory
        filename = file.filename or "video.mp4"
        dest_filename = f"{file_sha256[:16]}_{Path(filename).name}"
        dest_path = config.VIDEO_SNIPPETS_DIR / dest_filename
        if not dest_path.exists():
            dest_path.write_bytes(file_bytes)
        saved_input_path = to_relative_path(dest_path)

    elif file_path is not None:
        path_obj = Path(file_path)
        if not path_obj.is_absolute():
            path_obj = config.BASE_DIR / path_obj
        if not path_obj.exists() or not path_obj.is_file():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"file_path does not exist or is not a valid file: {file_path}",
            )
        file_bytes = path_obj.read_bytes()
        if not file_bytes:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File at file_path is empty.")
        file_sha256 = hashlib.sha256(file_bytes).hexdigest()
        saved_input_path = to_relative_path(path_obj)

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'file' upload or 'file_path' must be provided.",
        )

    # Atomic create or get job
    job, is_new = create_or_get_job(
        event_id=event_id.strip(),
        file_sha256=file_sha256,
        input_path=saved_input_path,
    )

    # Only enqueue the background worker if this is a brand new event
    if is_new:
        job_executor.submit(background_job_worker, job["job_id"])

    return format_job_record(job, is_duplicate_event=(not is_new))


@app.get("/jobs", response_model=list[JobResponse])
async def get_all_jobs():
    """List all jobs with status and telemetry for the dashboard."""
    jobs = list_jobs()
    return [format_job_record(j) for j in jobs]


@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_by_id(job_id: str):
    """Retrieve details for a single job by its ID."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' not found.")
    return format_job_record(job)


@app.post("/jobs/{job_id}/review_action", response_model=JobResponse)
async def review_action(job_id: str, payload: ReviewActionRequest):
    """Allow a human reviewer to update status ('SAFE' override or 'NEEDS_REVIEW' confirmed) with reviewer notes."""
    valid_statuses = {"SAFE", "NEEDS_REVIEW", "QUEUED", "PROCESSING_REDACTION", "VERIFYING_OUTPUT"}
    target_status = payload.status.upper().strip()
    if target_status not in valid_statuses:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid status '{payload.status}'. Allowed: {sorted(valid_statuses)}",
        )

    updated = update_review_action(
        job_id=job_id,
        status=target_status,
        reviewer_notes=payload.reviewer_notes,
    )
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' not found.")

    return format_job_record(updated)
