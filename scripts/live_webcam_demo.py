"""
Live Webcam Demonstration Tool (Non-Agentic Pipeline).

Demonstrates real-time face redaction at 20-30 FPS:
  - Base Face Detection: SCRFD-34G
  - Skeletal Keypoint Reasoning: YOLOv8n-Pose (Tracks head when turned/occluded)
  - Obfuscation Engine: Heavy Dual Mosaic / Gaussian Redaction
  - Live Surveillance HUD Overlay: FPS, face count, active status

Interactive Hotkeys:
  [SPACE] - Toggle Redaction ON / OFF (Show raw face vs. blurred face)
  [P]     - Toggle Pose Fusion ON / OFF (Demonstrate head tracking on/off)
  [M]     - Toggle Redaction Mode (Gaussian vs Mosaic)
  [Q/ESC] - Quit Demo
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import config
from app.pipeline.detector import PrimaryFaceDetector
from app.pipeline.redactor import apply_obfuscation, dilate_bbox

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("webcam_demo")


def draw_hud(
    frame: np.ndarray,
    fps: float,
    detected_count: int,
    redaction_enabled: bool,
    pose_enabled: bool,
    mode: str,
) -> np.ndarray:
    """Renders a sleek, professional surveillance HUD overlay on the display frame."""
    out = frame.copy()
    h, w = out.shape[:2]

    # Top Status Bar (Translucent Dark Banner)
    header_h = 44
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), (12, 16, 26), -1)
    cv2.addWeighted(overlay, 0.85, out, 0.15, 0, out)

    # Accent divider line
    accent_color = (0, 230, 118) if redaction_enabled else (0, 80, 255)
    cv2.line(out, (0, header_h), (w, header_h), accent_color, 2)

    # 1. System Title
    cv2.putText(
        out,
        "PRIVACY SHIELD LIVE DEMO",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # 2. Status Badge (PROTECTED vs RAW)
    status_text = "● REDACTION ACTIVE" if redaction_enabled else "○ RAW FEED (UNPROTECTED)"
    status_color = (0, 230, 118) if redaction_enabled else (50, 70, 255)
    cv2.putText(
        out,
        status_text,
        (330, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        status_color,
        2,
        cv2.LINE_AA,
    )

    # 3. Pose Tracking Indicator
    pose_text = "POSE: ON" if pose_enabled else "POSE: OFF"
    pose_color = (0, 220, 255) if pose_enabled else (120, 120, 120)
    cv2.putText(
        out,
        pose_text,
        (w - 330, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        pose_color,
        2,
        cv2.LINE_AA,
    )

    # 4. Faces Count & FPS
    stats_text = f"Faces: {detected_count} | {fps:.1f} FPS"
    cv2.putText(
        out,
        stats_text,
        (w - 195, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 220, 240),
        2,
        cv2.LINE_AA,
    )

    # Bottom Hotkey Bar (Subtle banner)
    footer_h = 32
    overlay_footer = out.copy()
    cv2.rectangle(overlay_footer, (0, h - footer_h), (w, h), (12, 16, 26), -1)
    cv2.addWeighted(overlay_footer, 0.85, out, 0.15, 0, out)
    cv2.line(out, (0, h - footer_h), (w, h - footer_h), (60, 70, 90), 1)

    controls_text = "[SPACE] Toggle Redaction   |   [P] Toggle Pose Tracking   |   [M] Toggle Blur Mode   |   [Q] Quit"
    cv2.putText(
        out,
        controls_text,
        (16, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (160, 175, 195),
        1,
        cv2.LINE_AA,
    )

    return out


def run_live_demo(
    camera_id: int = 0,
    conf_thresh: float = 0.35,
    dilation_ratio: float = 0.28,
) -> None:
    """Captures live webcam feed, runs SCRFD + Pose fusion, and displays live HUD."""
    logger.info("Initializing models for Live Webcam Demo...")
    logger.info("Loading Primary Face Detector (SCRFD-34G)...")
    detector_with_pose = PrimaryFaceDetector(conf_thresh=conf_thresh, use_pose_fallback=True)
    detector_without_pose = PrimaryFaceDetector(conf_thresh=conf_thresh, use_pose_fallback=False)

    logger.info("Opening webcam device (index %d)...", camera_id)
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        # Fallback to camera index 1 if 0 fails (common on Macs with Continuity Camera)
        logger.warning("Camera index %d failed, trying camera index 1...", camera_id)
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            logger.error("Could not open any webcam. Please ensure camera permissions are granted.")
            print("\n[ERROR] Unable to access webcam.")
            print("Please check that your terminal has Camera permissions in macOS:")
            print("System Settings -> Privacy & Security -> Camera -> Allow Terminal/IDE.\n")
            return

    # Attempt to set comfortable 720p resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "Live Face Redaction Demo (Non-Agentic Pipeline)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # State variables
    redaction_enabled = True
    pose_enabled = True
    blur_mode = "gaussian_or_mosaic"  # "gaussian_or_mosaic" or "mosaic"
    
    fps = 0.0
    frame_count = 0
    t_start = time.perf_counter()

    print("\n" + "=" * 70)
    print("      LIVE FACE REDACTION DEMO (NON-AGENTIC PIPELINE)")
    print("=" * 70)
    print("Controls:")
    print("  [SPACE] - Toggle Redaction ON / OFF")
    print("  [P]     - Toggle Pose Fusion Tracking ON / OFF")
    print("  [M]     - Toggle Blur Mode (Dual vs Mosaic)")
    print("  [Q/ESC] - Exit demo")
    print("=" * 70 + "\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("Dropped frame or camera disconnected.")
                break

            h, w = frame.shape[:2]
            t_frame_start = time.perf_counter()

            # Select active detector
            active_detector = detector_with_pose if pose_enabled else detector_without_pose
            detections = active_detector.detect(frame)
            detected_count = len(detections)

            # Apply redaction if enabled
            if redaction_enabled and detected_count > 0:
                dilated_boxes = [dilate_bbox(d["bbox"], (h, w), ratio=dilation_ratio) for d in detections]
                display_frame = apply_obfuscation(frame, dilated_boxes, mode=blur_mode)
            else:
                display_frame = frame.copy()

            # Compute smooth moving-average FPS
            frame_count += 1
            elapsed = time.perf_counter() - t_frame_start
            instant_fps = 1.0 / max(elapsed, 1e-5)
            fps = 0.90 * fps + 0.10 * instant_fps if fps > 0 else instant_fps

            # Render HUD
            hud_frame = draw_hud(
                display_frame,
                fps=fps,
                detected_count=detected_count,
                redaction_enabled=redaction_enabled,
                pose_enabled=pose_enabled,
                mode=blur_mode,
            )

            cv2.imshow(window_name, hud_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # 27 = ESC
                logger.info("Exiting live demo.")
                break
            elif key == ord(" "):  # Space: Toggle Redaction
                redaction_enabled = not redaction_enabled
                logger.info("Redaction toggled: %s", "ENABLED" if redaction_enabled else "DISABLED")
            elif key in (ord("p"), ord("P")):  # P: Toggle Pose
                pose_enabled = not pose_enabled
                logger.info("Pose tracking toggled: %s", "ENABLED" if pose_enabled else "DISABLED")
            elif key in (ord("m"), ord("M")):  # M: Toggle Mode
                blur_mode = "mosaic" if blur_mode == "gaussian_or_mosaic" else "gaussian_or_mosaic"
                logger.info("Blur mode changed to: %s", blur_mode)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Webcam released and windows closed.")


if __name__ == "__main__":
    run_live_demo()
