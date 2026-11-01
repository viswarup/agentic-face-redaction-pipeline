"""Post-Render RetinaFace QA Verifier and Fail-Closed Verification Engine."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from app.config import config
from app.pipeline.tracker_interpolator import compute_iou

logger = logging.getLogger("anonymizer.verifier")

# Point DeepFace / RetinaFace cache to project weights directory
os.environ.setdefault("DEEPFACE_HOME", str(config.BASE_DIR / "weights" / "deepface"))


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


class RetinaFaceVerifier:
    """
    Independent QA face verifier running a deep RetinaFace backbone
    to audit rendered videos and benchmark frames.
    Implements a strict fail-closed design: fails if weights cannot be loaded.
    """

    def __init__(
        self,
        conf_thresh: Optional[float] = None,
        custom_weights_path: Optional[str | Path] = None,
    ):
        self.conf_thresh = conf_thresh if conf_thresh is not None else config.VERIFIER_CONF_THRESH
        self.custom_weights_path = custom_weights_path or config.RETINAFACE_R152_PATH
        self.model: Any = None
        self.backend: str = "retinaface_pth"
        self._init_model()

    def _init_model(self) -> None:
        """Initialize RetinaFace model using PyTorch ResNet-50 weights or custom R152 checkpoint."""
        try:
            from retinaface import RetinaFace

            if self.custom_weights_path and Path(self.custom_weights_path).exists():
                import torch
                from retinaface.model.retinaface_pth_model import RetinaFace as TorchRetinaFace

                logger.info("Loading custom RetinaFace checkpoint from %s", self.custom_weights_path)
                model = TorchRetinaFace()
                state_dict = torch.load(
                    str(self.custom_weights_path), map_location="cpu", weights_only=True
                )
                model.load_state_dict(state_dict)
                model.eval()
                self.model = model
            else:
                self.model = RetinaFace.build_model()
            logger.info("RetinaFace QA Verifier successfully initialized.")
        except Exception as e:
            logger.error("Fail-closed: Failed to initialize RetinaFace QA Verifier: %s", e)
            raise RuntimeError(
                f"MODEL_OR_PROCESSING_ERROR: RetinaFace QA Verifier initialization failed: {e}"
            ) from e

    def detect(self, image_bgr: np.ndarray, conf_thresh: Optional[float] = None) -> list[dict[str, Any]]:
        """
        Detect any visible faces in an image using RetinaFace.

        Returns:
            list of dicts: [
                {
                    "bbox": [x1, y1, x2, y2],
                    "score": float,
                    "landmarks": {"left_eye": [...], ...}
                },
                ...
            ]
        """
        if image_bgr is None or image_bgr.size == 0:
            return []

        if self.model is None:
            raise RuntimeError("MODEL_OR_PROCESSING_ERROR: RetinaFace model is not initialized.")

        threshold = conf_thresh if conf_thresh is not None else self.conf_thresh
        h, w = image_bgr.shape[:2]
        results: list[dict[str, Any]] = []

        try:
            from retinaface import RetinaFace

            raw_faces = RetinaFace.detect_faces(image_bgr, threshold=threshold, model=self.model)
            if not isinstance(raw_faces, dict):
                return []

            for key, face_info in raw_faces.items():
                score = float(face_info.get("score", 0.0))
                if score < threshold:
                    continue

                facial_area = face_info.get("facial_area", [])
                if len(facial_area) != 4:
                    continue

                x1 = float(np.clip(facial_area[0], 0, w))
                y1 = float(np.clip(facial_area[1], 0, h))
                x2 = float(np.clip(facial_area[2], 0, w))
                y2 = float(np.clip(facial_area[3], 0, h))

                if x2 <= x1 or y2 <= y1:
                    continue

                landmarks = face_info.get("landmarks", {})
                results.append(
                    {
                        "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                        "score": round(score, 4),
                        "landmarks": landmarks,
                    }
                )

        except Exception as e:
            logger.exception("RetinaFace detection error: %s", e)
            raise RuntimeError(f"MODEL_OR_PROCESSING_ERROR: RetinaFace detection failed: {e}") from e

        return results


def verify_single_image(
    rendered_bgr: np.ndarray,
    conf_thresh: Optional[float] = None,
    verifier: Optional[RetinaFaceVerifier] = None,
) -> list[dict[str, Any]]:
    """Verify single rendered image with RetinaFace."""
    if verifier is None:
        verifier = RetinaFaceVerifier(conf_thresh=conf_thresh)
    return verifier.detect(rendered_bgr, conf_thresh=conf_thresh)


def verify_rendered_video(
    rendered_mp4_path: str | Path,
    tracks_json_path: Optional[str | Path] = None,
    conf_thresh: Optional[float] = None,
    stride: Optional[int] = None,
    job_id: Optional[str] = None,
    verifier: Optional[RetinaFaceVerifier] = None,
) -> dict[str, Any]:
    """
    Independently inspect rendered video from disk with RetinaFace to ensure zero unblurred faces.
    Implements a strict fail-closed safety state machine: any file or codec error defaults to NEEDS_REVIEW.

    Returns:
        {
            "status": "SAFE" | "NEEDS_REVIEW",
            "reason": Optional[str],
            "decoded_frames": int,
            "inspected_frames": int,
            "flagged_frames": list[dict],
            "blind_spot_warning": bool,
            "blind_spot_details": list[str],
            "telemetry": dict,
        }
    """
    threshold = conf_thresh if conf_thresh is not None else config.VERIFIER_CONF_THRESH
    step = stride if stride is not None else config.VERIFIER_STRIDE
    job_prefix = job_id or "job"

    # FAIL-CLOSED EXCEPTION GUARD
    try:
        mp4_path = Path(rendered_mp4_path).resolve()
        if not mp4_path.exists() or not mp4_path.is_file():
            logger.error("Fail-closed: Rendered MP4 does not exist at %s", rendered_mp4_path)
            return {
                "status": "NEEDS_REVIEW",
                "reason": "VERIFICATION_OR_CODEC_ERROR",
                "decoded_frames": 0,
                "inspected_frames": 0,
                "flagged_frames": [],
                "blind_spot_warning": True,
                "blind_spot_details": ["Rendered video file is missing or unreadable."],
                "telemetry": {},
            }

        cap = cv2.VideoCapture(str(mp4_path))
        if not cap.isOpened():
            logger.error("Fail-closed: Cannot open video %s via OpenCV", mp4_path)
            return {
                "status": "NEEDS_REVIEW",
                "reason": "VERIFICATION_OR_CODEC_ERROR",
                "decoded_frames": 0,
                "inspected_frames": 0,
                "flagged_frames": [],
                "blind_spot_warning": True,
                "blind_spot_details": ["OpenCV failed to decode video container."],
                "telemetry": {},
            }

        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0

        # Load applied blur boxes from companion tracks.json if available
        applied_boxes_by_frame: dict[int, list[list[float]]] = {}
        tracks_telemetry: dict[str, Any] = {}

        if tracks_json_path and Path(tracks_json_path).exists():
            try:
                with open(tracks_json_path, "r", encoding="utf-8") as f:
                    companion_data = json.load(f)
                tracks_telemetry = companion_data.get("metadata", {})
                for frame_entry in companion_data.get("frames", []):
                    f_idx = frame_entry.get("frame_index")
                    boxes = [b.get("dilated_bbox") or b.get("raw_bbox") for b in frame_entry.get("boxes", [])]
                    applied_boxes_by_frame[f_idx] = [b for b in boxes if b is not None]
            except Exception as e:
                logger.warning("Error reading tracks json (%s): %s", tracks_json_path, e)

        if verifier is None:
            verifier = RetinaFaceVerifier(conf_thresh=threshold)

        diagnostics_dir = config.DIAGNOSTICS_DIR
        diagnostics_dir.mkdir(parents=True, exist_ok=True)

        decoded_frames = 0
        inspected_frames = 0
        flagged_frames: list[dict[str, Any]] = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            decoded_frames += 1

            if frame_idx % step == 0:
                inspected_frames += 1
                faces = verifier.detect(frame, conf_thresh=threshold)

                if faces:
                    timestamp_sec = round(frame_idx / fps, 3)
                    applied_boxes = applied_boxes_by_frame.get(frame_idx, [])

                    for face in faces:
                        rf_box = [float(v) for v in face["bbox"]]
                        rf_score = float(face["score"])
                        clean_landmarks = {
                            k: [float(coord) for coord in v]
                            if isinstance(v, (list, tuple, np.ndarray))
                            else v
                            for k, v in face.get("landmarks", {}).items()
                        }

                        # Calculate IoU with applied blur boxes
                        max_iou = 0.0
                        for abox in applied_boxes:
                            iou = compute_iou(rf_box, abox)
                            if iou > max_iou:
                                max_iou = iou

                        # Diagnostic Classification
                        if max_iou >= 0.10:
                            diag_class = "RESIDUAL_DILATION_LEAK"
                        else:
                            diag_class = "UNBLURRED_MISSED_FACE"

                        # Save diagnostic JPEG snapshot
                        snapshot_filename = f"{job_prefix}_f{frame_idx}.jpg"
                        snapshot_path = diagnostics_dir / snapshot_filename

                        diag_img = frame.copy()
                        # Draw applied blur boxes in Blue (255, 0, 0)
                        for ab in applied_boxes:
                            cv2.rectangle(
                                diag_img,
                                (int(ab[0]), int(ab[1])),
                                (int(ab[2]), int(ab[3])),
                                (255, 0, 0),
                                2,
                            )
                        # Draw RetinaFace leak box in Red (0, 0, 255)
                        cv2.rectangle(
                            diag_img,
                            (int(rf_box[0]), int(rf_box[1])),
                            (int(rf_box[2]), int(rf_box[3])),
                            (0, 0, 255),
                            2,
                        )
                        label = f"{diag_class} ({rf_score:.2f}, IoU:{max_iou:.2f})"
                        cv2.putText(
                            diag_img,
                            label,
                            (int(rf_box[0]), max(15, int(rf_box[1]) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            2,
                        )
                        cv2.imwrite(str(snapshot_path), diag_img)

                        flagged_frames.append(
                            {
                                "frame_idx": int(frame_idx),
                                "timestamp_sec": float(timestamp_sec),
                                "retinaface_bbox": rf_box,
                                "retinaface_score": rf_score,
                                "landmarks": clean_landmarks,
                                "diagnostic_classification": diag_class,
                                "max_applied_blur_iou": float(round(max_iou, 3)),
                                "snapshot_path": to_relative_path(snapshot_path),
                            }
                        )

            frame_idx += 1

        cap.release()

        # Fail-closed if 0 frames could be decoded
        if decoded_frames == 0:
            logger.error("Fail-closed: 0 frames decoded from %s", mp4_path)
            return {
                "status": "NEEDS_REVIEW",
                "reason": "VERIFICATION_OR_CODEC_ERROR",
                "decoded_frames": 0,
                "inspected_frames": 0,
                "flagged_frames": [],
                "blind_spot_warning": True,
                "blind_spot_details": ["Zero video frames could be decoded."],
                "telemetry": {},
            }

        # Check Epistemic Correlated Blind-Spot Risk
        blind_spot_warning = False
        blind_spot_details: list[str] = []

        mean_y = tracks_telemetry.get("mean_y_luminance", 100.0)
        mean_lap = tracks_telemetry.get("mean_laplacian_blur_variance", 500.0)

        if mean_y < 40.0:
            blind_spot_warning = True
            blind_spot_details.append(
                f"Low luminance alert: mean Y luminance ({mean_y:.1f}) < 40 indicates severe underexposure."
            )
        if mean_lap < 100.0:
            blind_spot_warning = True
            blind_spot_details.append(
                f"Low sharpness alert: Laplacian variance ({mean_lap:.1f}) < 100 indicates heavy motion blur."
            )

        # Final Verification Status: Fail-closed if ANY leak detected
        if len(flagged_frames) > 0:
            status = "NEEDS_REVIEW"
            reason = f"{len(flagged_frames)} potentially visible face instances flagged by RetinaFace."
        else:
            status = "SAFE"
            reason = None

        verifier_telemetry = {
            "decoded_frames": decoded_frames,
            "inspected_frames": inspected_frames,
            "stride": step,
            "conf_thresh": threshold,
            "flagged_count": len(flagged_frames),
            "blind_spot_warning": blind_spot_warning,
            "blind_spot_details": blind_spot_details,
        }

        return {
            "status": status,
            "reason": reason,
            "decoded_frames": decoded_frames,
            "inspected_frames": inspected_frames,
            "flagged_frames": flagged_frames,
            "blind_spot_warning": blind_spot_warning,
            "blind_spot_details": blind_spot_details,
            "telemetry": verifier_telemetry,
        }

    except Exception as e:
        logger.exception("Fail-closed: Unhandled exception during video verification: %s", e)
        return {
            "status": "NEEDS_REVIEW",
            "reason": "MODEL_OR_PROCESSING_ERROR",
            "decoded_frames": 0,
            "inspected_frames": 0,
            "flagged_frames": [],
            "blind_spot_warning": True,
            "blind_spot_details": [f"Verification exception: {str(e)}"],
            "telemetry": {},
        }
