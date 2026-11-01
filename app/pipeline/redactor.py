"""Face Obfuscation, Video Stream Redaction, and Telemetry Engine."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from app.config import config
from app.pipeline.detector import PrimaryFaceDetector
from app.pipeline.tracker_interpolator import TwoPassTrajectorySmoother, VideoFaceTracker

logger = logging.getLogger("anonymizer.redactor")


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



def dilate_bbox(
    bbox: list[float] | tuple[float, float, float, float],
    frame_shape: tuple[int, int, ...],
    ratio: float = 0.28,
) -> list[int]:
    """
    Dilate bounding box width and height by ratio (default 28%) to cover
    forehead, hairline, ears, chin, and fast motion lag.

    Returns:
        [x1, y1, x2, y2] clamped to frame boundaries as integer pixel coordinates.
    """
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    box_w = max(1.0, float(x2 - x1))
    box_h = max(1.0, float(y2 - y1))

    dw = (box_w * ratio) / 2.0
    dh = (box_h * ratio) / 2.0

    dx1 = max(0, int(np.floor(x1 - dw)))
    dy1 = max(0, int(np.floor(y1 - dh)))
    dx2 = min(w, int(np.ceil(x2 + dw)))
    dy2 = min(h, int(np.ceil(y2 + dh)))

    return [dx1, dy1, dx2, dy2]


def apply_obfuscation(
    frame: np.ndarray,
    dilated_boxes: list[list[int]],
    mode: str = "gaussian_or_mosaic",
) -> np.ndarray:
    """
    Apply heavy adaptive obfuscation to dilated bounding boxes so facial features
    are completely unrecoverable.

    Args:
        frame: BGR image array (H, W, 3).
        dilated_boxes: list of [x1, y1, x2, y2] integer bounding boxes.
        mode: "gaussian_or_mosaic" (default), "gaussian", or "mosaic".

    Returns:
        Redacted BGR image array.
    """
    if frame is None or len(dilated_boxes) == 0:
        return frame

    out = frame.copy()
    h, w = out.shape[:2]

    for box in dilated_boxes:
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        roi_w = x2 - x1
        roi_h = y2 - y1
        if roi_w <= 2 or roi_h <= 2:
            continue

        roi = out[y1:y2, x1:x2]

        if mode == "mosaic":
            # Solid pixelation
            block_w = max(4, roi_w // 10)
            block_h = max(4, roi_h // 10)
            small = cv2.resize(
                roi,
                (max(1, roi_w // block_w), max(1, roi_h // block_h)),
                interpolation=cv2.INTER_LINEAR,
            )
            pixelated = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
            out[y1:y2, x1:x2] = pixelated

        elif mode == "gaussian":
            # Adaptive Gaussian blur with kernel size proportional to box width (odd >= 31)
            ksize = max(31, int(roi_w * 0.55) | 1)
            blurred = cv2.GaussianBlur(roi, (ksize, ksize), sigmaX=ksize / 3.0)
            out[y1:y2, x1:x2] = blurred

        else:
            # Default "gaussian_or_mosaic": Hybrid pixelation + heavy Gaussian blur with elliptical feathering
            # 1. Pre-pixelate to destroy underlying high-frequency biometric details
            block_size = max(4, int(min(roi_w, roi_h) * 0.12))
            small = cv2.resize(
                roi,
                (max(1, roi_w // block_size), max(1, roi_h // block_size)),
                interpolation=cv2.INTER_AREA,
            )
            pixelated = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)

            # 2. Heavy adaptive Gaussian blur over pixelated block
            ksize = max(31, int(roi_w * 0.60) | 1)
            blurred = cv2.GaussianBlur(pixelated, (ksize, ksize), sigmaX=ksize / 2.5)

            # 3. Smooth elliptical feathering to seamlessly blend edges without square artifacts
            mask = np.zeros((roi_h, roi_w), dtype=np.float32)
            center = (roi_w // 2, roi_h // 2)
            axes = (roi_w // 2, roi_h // 2)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
            feather_k = max(7, (min(roi_w, roi_h) // 6) | 1)
            mask = cv2.GaussianBlur(mask, (feather_k, feather_k), sigmaX=feather_k / 3.0)
            mask = np.clip(mask, 0.0, 1.0)[:, :, None]

            out[y1:y2, x1:x2] = (blurred.astype(np.float32) * mask + roi.astype(np.float32) * (1.0 - mask)).astype(
                np.uint8
            )

    return out


def redact_single_image(
    image_bgr: np.ndarray,
    detector: Optional[PrimaryFaceDetector] = None,
    conf_thresh: Optional[float] = None,
    dilation_ratio: float = 0.28,
    mode: str = "gaussian_or_mosaic",
    use_pose_fallback: bool = False,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """
    Redact a static image (for benchmark hand-annotated images).

    Returns:
        (redacted_image_bgr, list_of_detection_records)
    """
    if detector is None:
        detector = PrimaryFaceDetector(conf_thresh=conf_thresh, use_pose_fallback=use_pose_fallback)

    detections = detector.detect(image_bgr, conf_thresh=conf_thresh)
    dilated_boxes = [dilate_bbox(d["bbox"], image_bgr.shape, ratio=dilation_ratio) for d in detections]
    redacted = apply_obfuscation(image_bgr, dilated_boxes, mode=mode)

    records = []
    for d, dbox in zip(detections, dilated_boxes):
        records.append(
            {
                "raw_bbox": d["bbox"],
                "dilated_bbox": dbox,
                "score": d["score"],
                "landmarks": d.get("landmarks", []),
                "source": d.get("source", "detection"),
            }
        )

    return redacted, records


def process_video_redaction(
    input_video_path: str | Path,
    output_video_path: str | Path,
    detector: Optional[PrimaryFaceDetector] = None,
    conf_thresh: Optional[float] = None,
    dilation_ratio: Optional[float] = None,
    interpolation_window: Optional[int] = None,
    obfuscation_mode: str = "gaussian_or_mosaic",
    use_pose_fallback: bool = False,
    use_tracker: bool = True,
) -> dict[str, Any]:
    """
    Stream and redact an input video file using detection + ByteTrack + two-pass trajectory smoothing.

    Generates:
        1. Encoded H.264/MP4 redacted video at output_video_path.
        2. Companion tracks and telemetry file at f"{output_video_path}.tracks.json".

    Returns:
        Telemetry summary dictionary.
    """
    in_path = Path(input_video_path).resolve()
    out_path = Path(output_video_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tracks_json_path = Path(f"{out_path}.tracks.json")

    ratio = dilation_ratio if dilation_ratio is not None else config.BBOX_DILATION_RATIO
    max_gap = interpolation_window if interpolation_window is not None else config.INTERPOLATION_WINDOW
    thresh = conf_thresh if conf_thresh is not None else config.PRIMARY_CONF_THRESH

    if detector is None:
        detector = PrimaryFaceDetector(conf_thresh=thresh, use_pose_fallback=use_pose_fallback)

    tracker = VideoFaceTracker(
        high_thresh=max(0.20, thresh),
        low_thresh=max(0.10, thresh / 2.0),
        max_age=config.TRACKER_MAX_AGE,
    )
    smoother = TwoPassTrajectorySmoother(max_gap=max_gap, coast_frames=3, interior_margin=25)

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open input video: {in_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(
        "Starting redaction on %s: %d frames, %dx%d, %.2f fps",
        in_path.name,
        total_frames,
        width,
        height,
        fps,
    )

    # Pass 1: Stream frames, collect telemetry, run detection & tracking
    start_time = time.perf_counter()
    raw_frames: list[np.ndarray] = []
    raw_tracked_detections: list[list[dict[str, Any]]] = []

    y_luminances: list[float] = []
    laplacian_variances: list[float] = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        raw_frames.append(frame)

        # Telemetry: Mean Y luminance & Laplacian blur variance
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        y_lum = float(np.mean(gray))
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        y_luminances.append(y_lum)
        laplacian_variances.append(lap_var)

        # Detect and Track
        dets = detector.detect(frame, conf_thresh=thresh)
        if use_tracker:
            tracked = tracker.update(dets)
        else:
            tracked = [
                {
                    "track_id": None,
                    "bbox": d["bbox"],
                    "score": d["score"],
                    "source": "detection",
                    "landmarks": d.get("landmarks", []),
                }
                for d in dets
            ]
        raw_tracked_detections.append(tracked)

        frame_idx += 1

    cap.release()
    actual_frames = len(raw_frames)
    if actual_frames == 0:
        raise RuntimeError(f"No frames could be read from video {in_path}")

    # Pass 2: Bidirectional Trajectory Smoothing (Gap interpolation + birth/death coasting)
    if use_tracker:
        smoothed_tracked_detections = smoother.smooth(raw_tracked_detections, (height, width))
    else:
        smoothed_tracked_detections = raw_tracked_detections

    # Pass 3: Apply Dilated Obfuscation and Encode Video
    ffmpeg_bin = shutil.which("ffmpeg")
    ffmpeg_proc: Optional[subprocess.Popen] = None
    cv_writer: Optional[cv2.VideoWriter] = None

    if ffmpeg_bin:
        ffmpeg_cmd = [
            ffmpeg_bin,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "bgr24",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        cv_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    total_interpolated = 0
    total_coasted = 0
    total_detections = 0
    per_frame_records: list[dict[str, Any]] = []

    for f_idx, (frame, boxes) in enumerate(zip(raw_frames, smoothed_tracked_detections)):
        frame_boxes_record: list[dict[str, Any]] = []
        dilated_boxes_for_frame: list[list[int]] = []

        for b in boxes:
            src = b.get("source", "detection")
            if src == "interpolated_coaster":
                total_interpolated += 1
            elif src.startswith("coast_"):
                total_coasted += 1
            else:
                total_detections += 1

            raw_box = b["bbox"]
            dilated = dilate_bbox(raw_box, (height, width), ratio=ratio)
            dilated_boxes_for_frame.append(dilated)

            frame_boxes_record.append(
                {
                    "track_id": b.get("track_id"),
                    "raw_bbox": raw_box,
                    "dilated_bbox": dilated,
                    "score": b.get("score"),
                    "source": src,
                }
            )

        redacted_frame = apply_obfuscation(frame, dilated_boxes_for_frame, mode=obfuscation_mode)

        if ffmpeg_proc and ffmpeg_proc.stdin:
            ffmpeg_proc.stdin.write(redacted_frame.tobytes())
        elif cv_writer:
            cv_writer.write(redacted_frame)

        per_frame_records.append(
            {
                "frame_index": f_idx,
                "box_count": len(frame_boxes_record),
                "boxes": frame_boxes_record,
            }
        )

    if ffmpeg_proc and ffmpeg_proc.stdin:
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
    elif cv_writer:
        cv_writer.release()

    elapsed = time.perf_counter() - start_time
    processing_fps = round(actual_frames / elapsed, 2) if elapsed > 0 else 0.0

    telemetry = {
        "input_video_path": to_relative_path(in_path),
        "output_video_path": to_relative_path(out_path),
        "resolution": [width, height],
        "fps": fps,
        "total_frames": actual_frames,
        "elapsed_seconds": round(elapsed, 2),
        "processing_fps": processing_fps,
        "mean_y_luminance": round(float(np.mean(y_luminances)), 2) if y_luminances else 0.0,
        "mean_laplacian_blur_variance": round(float(np.mean(laplacian_variances)), 2)
        if laplacian_variances
        else 0.0,
        "total_detections": total_detections,
        "total_interpolated_boxes": total_interpolated,
        "total_coasted_boxes": total_coasted,
        "detector_model": detector.model_name,
        "detector_weight_path": to_relative_path(detector.loaded_weight_path) if detector.loaded_weight_path else None,
    }

    # Save companion tracks JSON
    companion_payload = {
        "metadata": telemetry,
        "frames": per_frame_records,
    }
    with open(tracks_json_path, "w", encoding="utf-8") as f:
        json.dump(companion_payload, f, indent=2)

    logger.info(
        "Completed redaction for %s in %.2fs (%.1f fps). Tracks saved to %s",
        in_path.name,
        elapsed,
        processing_fps,
        tracks_json_path.name,
    )

    return telemetry
