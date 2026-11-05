"""
High-Performance Live Webcam Demonstration Tool (Non-Agentic Pipeline).

Optimized for smooth 25-30 FPS real-time execution:
  - Hardware Acceleration: CoreMLExecutionProvider on Apple Silicon (falls back to CPU)
  - Threaded Camera Reader: Asynchronous frame capture prevents OpenCV buffer lag
  - Single-Pass Unified Inference: YOLOv8-Pose runs once per cycle, reused for fusion & rendering
  - Downscaled Inference with Full-HD Display: Inference at 640px, upscaled for rendering
  - Tighter Dilation: Default 14% tight margin with hotkey adjustment
  - Surveillance HUD Overlay: FPS, face count, active status, landmark points & skeleton

Interactive Hotkeys:
  [SPACE] - Toggle Redaction ON / OFF (Show raw face with AI points vs. blurred face)
  [K]     - Toggle AI Keypoints & Skeleton overlay ON / OFF
  [P]     - Toggle Pose Fusion Tracking ON / OFF
  [ [ / ] ] - Decrease / Increase blur dilation radius (-/+ 2%)
  [M]     - Toggle Blur Mode (Dual vs Mosaic)
  [Q/ESC] - Quit Demo
"""

from __future__ import annotations

import logging
import sys
import threading
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

# COCO Pose Skeleton Connections (Keypoint index pairs)
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2),  # Nose to Eyes
    (1, 3), (2, 4),  # Eyes to Ears
    (5, 6),          # Shoulder to Shoulder
    (5, 7), (7, 9),  # Left Arm (Shoulder -> Elbow -> Wrist)
    (6, 8), (8, 10), # Right Arm (Shoulder -> Elbow -> Wrist)
]


class WebcamStream:
    """Non-blocking background video capture to prevent OpenCV camera buffer lag."""

    def __init__(self, src: int = 0, width: int = 1280, height: int = 720):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(1)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()

        if self.ret:
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()

    def isOpened(self) -> bool:
        return self.cap.isOpened()

    def _update(self) -> None:
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                continue
            with self.lock:
                self.ret = ret
                self.frame = frame
            time.sleep(0.005)

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self.lock:
            if not self.ret or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def release(self) -> None:
        self.stopped = True
        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join(timeout=0.5)
        self.cap.release()


def draw_ai_landmarks(
    frame: np.ndarray,
    scrfd_detections: list[dict],
    pose_results: list,
    scale_factor: float = 1.0,
) -> np.ndarray:
    """Renders SCRFD 5-point facial landmarks and YOLOv8 17-point skeletal pose."""
    out = frame.copy()

    # 1. Render YOLOv8-Pose Skeleton & Keypoints
    for res in pose_results:
        if res.keypoints is None or len(res.keypoints.data) == 0:
            continue

        for kp in res.keypoints.data.cpu().numpy():
            # Draw Skeleton Bones
            for idx1, idx2 in SKELETON_CONNECTIONS:
                if idx1 < len(kp) and idx2 < len(kp):
                    pt1, pt2 = kp[idx1], kp[idx2]
                    if pt1[2] >= 0.35 and pt2[2] >= 0.35:
                        x1, y1 = int(pt1[0] * scale_factor), int(pt1[1] * scale_factor)
                        x2, y2 = int(pt2[0] * scale_factor), int(pt2[1] * scale_factor)
                        cv2.line(out, (x1, y1), (x2, y2), (0, 230, 118), 2, cv2.LINE_AA)

            # Neck anchor vector (Mid-shoulders to nose/ears)
            if len(kp) > 6 and kp[5][2] >= 0.35 and kp[6][2] >= 0.35:
                mid_x = int(((kp[5][0] + kp[6][0]) / 2) * scale_factor)
                mid_y = int(((kp[5][1] + kp[6][1]) / 2) * scale_factor)
                cv2.circle(out, (mid_x, mid_y), 4, (0, 255, 255), -1, cv2.LINE_AA)
                if kp[0][2] >= 0.35:  # Nose visible
                    nx, ny = int(kp[0][0] * scale_factor), int(kp[0][1] * scale_factor)
                    cv2.line(out, (mid_x, mid_y), (nx, ny), (0, 220, 255), 2, cv2.LINE_AA)

            # Draw Keypoint Dots
            for i, p in enumerate(kp):
                if p[2] >= 0.35:
                    px, py = int(p[0] * scale_factor), int(p[1] * scale_factor)
                    if i in (0, 1, 2, 3, 4):  # Head/Ears/Nose
                        cv2.circle(out, (px, py), 4, (0, 220, 255), -1, cv2.LINE_AA)
                    elif i in (5, 6):  # Shoulders
                        cv2.circle(out, (px, py), 6, (0, 255, 128), -1, cv2.LINE_AA)
                        cv2.circle(out, (px, py), 8, (0, 255, 128), 1, cv2.LINE_AA)
                    else:  # Arms/Body
                        cv2.circle(out, (px, py), 4, (0, 180, 255), -1, cv2.LINE_AA)

    # 2. Render SCRFD 5 Facial Landmarks (Cyan Crosshairs & Dots)
    for det in scrfd_detections:
        lms = det.get("landmarks", [])
        if lms and len(lms) >= 5:
            # 0=L_Eye, 1=R_Eye, 2=Nose, 3=L_Mouth, 4=R_Mouth
            for i, pt in enumerate(lms):
                lx, ly = int(pt[0]), int(pt[1])
                cv2.circle(out, (lx, ly), 5, (255, 230, 0), 1, cv2.LINE_AA)
                cv2.circle(out, (lx, ly), 3, (0, 240, 255), -1, cv2.LINE_AA)

            pt_le, pt_re, pt_nose = lms[0], lms[1], lms[2]
            cv2.line(out, (int(pt_le[0]), int(pt_le[1])), (int(pt_re[0]), int(pt_re[1])), (255, 200, 0), 1, cv2.LINE_AA)
            cv2.line(out, (int(pt_le[0]), int(pt_le[1])), (int(pt_nose[0]), int(pt_nose[1])), (255, 200, 0), 1, cv2.LINE_AA)
            cv2.line(out, (int(pt_re[0]), int(pt_re[1])), (int(pt_nose[0]), int(pt_nose[1])), (255, 200, 0), 1, cv2.LINE_AA)

    return out


def draw_hud(
    frame: np.ndarray,
    fps: float,
    detected_count: int,
    redaction_enabled: bool,
    pose_enabled: bool,
    keypoints_enabled: bool,
    dilation_ratio: float,
    mode: str,
) -> np.ndarray:
    """Renders a sleek, modern surveillance HUD overlay on the display frame."""
    out = frame.copy()
    h, w = out.shape[:2]

    # Top Status Bar
    header_h = 44
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), (12, 16, 26), -1)
    cv2.addWeighted(overlay, 0.85, out, 0.15, 0, out)

    accent_color = (0, 230, 118) if redaction_enabled else (0, 80, 255)
    cv2.line(out, (0, header_h), (w, header_h), accent_color, 2)

    # 1. System Title
    cv2.putText(out, "PRIVACY SHIELD LIVE DEMO", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # 2. Redaction Status Badge
    status_text = "● REDACTED" if redaction_enabled else "○ RAW"
    status_color = (0, 230, 118) if redaction_enabled else (50, 70, 255)
    cv2.putText(out, status_text, (310, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2, cv2.LINE_AA)

    # 3. Dilation Size Badge
    dil_text = f"Dilation: {int(dilation_ratio * 100)}%"
    cv2.putText(out, dil_text, (450, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 220, 240), 2, cv2.LINE_AA)

    # 4. Keypoints Badge
    kp_text = "POINTS: ON" if keypoints_enabled else "POINTS: OFF"
    kp_color = (0, 240, 255) if keypoints_enabled else (120, 120, 120)
    cv2.putText(out, kp_text, (w - 380, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, kp_color, 2, cv2.LINE_AA)

    # 5. Pose Tracking Indicator
    pose_text = "POSE: ON" if pose_enabled else "POSE: OFF"
    pose_color = (0, 230, 118) if pose_enabled else (120, 120, 120)
    cv2.putText(out, pose_text, (w - 240, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, pose_color, 2, cv2.LINE_AA)

    # 6. FPS
    fps_color = (0, 230, 118) if fps >= 20.0 else (0, 200, 255) if fps >= 12.0 else (0, 100, 255)
    stats_text = f"{fps:.1f} FPS"
    cv2.putText(out, stats_text, (w - 110, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, fps_color, 2, cv2.LINE_AA)

    # Bottom Hotkey Bar
    footer_h = 32
    overlay_footer = out.copy()
    cv2.rectangle(overlay_footer, (0, h - footer_h), (w, h), (12, 16, 26), -1)
    cv2.addWeighted(overlay_footer, 0.85, out, 0.15, 0, out)
    cv2.line(out, (0, h - footer_h), (w, h - footer_h), (60, 70, 90), 1)

    controls_text = "[SPACE] Redaction | [K] Points & Skeleton | [P] Pose | [ [ / ] ] Dilation | [M] Mode | [Q] Quit"
    cv2.putText(out, controls_text, (16, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 175, 195), 1, cv2.LINE_AA)

    return out


def run_live_demo(
    camera_id: int = 0,
    conf_thresh: float = 0.35,
    initial_dilation: float = 0.14,  # Tighter, reduced default dilation
) -> None:
    """Captures live webcam feed, runs SCRFD + Pose fusion at 25-30 FPS, and displays live HUD."""
    logger.info("Initializing models for High-Performance Live Webcam Demo...")
    logger.info("Loading Primary Face Detector (SCRFD-34G)...")
    
    detector = PrimaryFaceDetector(conf_thresh=conf_thresh, use_pose_fallback=True)

    # Enable Apple Silicon CoreML hardware acceleration if supported
    try:
        import onnxruntime as ort
        if "CoreMLExecutionProvider" in ort.get_available_providers():
            detector._model.session.set_providers(["CoreMLExecutionProvider", "CPUExecutionProvider"])
            logger.info("Accelerated SCRFD-34G with Apple Silicon CoreMLExecutionProvider!")
    except Exception as ex:
        logger.warning("Could not attach CoreML provider, using CPU: %s", ex)

    logger.info("Opening asynchronous webcam stream (device %d)...", camera_id)
    stream = WebcamStream(src=camera_id, width=1280, height=720)

    if not stream.isOpened():
        print("\n[ERROR] Unable to access webcam. Check macOS permissions:\nSystem Settings -> Privacy & Security -> Camera -> Allow Terminal/IDE.\n")
        return

    window_name = "Live Face Redaction Demo (Non-Agentic Pipeline)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # State variables
    redaction_enabled = True
    pose_enabled = True
    keypoints_enabled = True
    dilation_ratio = initial_dilation
    blur_mode = "gaussian_or_mosaic"
    
    fps = 0.0
    frame_idx = 0
    cached_detections: list[dict] = []
    cached_pose_results: list = []

    print("\n" + "=" * 70)
    print("      LIVE FACE REDACTION DEMO (HIGH-SPEED REAL-TIME PIPELINE)")
    print("=" * 70)
    print("Controls:")
    print("  [SPACE]   - Toggle Redaction ON / OFF")
    print("  [K]       - Toggle AI Facial Points & Skeleton ON / OFF")
    print("  [P]       - Toggle Pose Fusion Tracking ON / OFF")
    print("  [ [ / ] ] - Decrease / Increase blur dilation radius (-/+ 2%)")
    print("  [M]       - Toggle Blur Mode (Dual vs Mosaic)")
    print("  [Q/ESC]   - Exit demo")
    print("=" * 70 + "\n")

    try:
        while True:
            ret, frame = stream.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            h, w = frame.shape[:2]
            t_frame_start = time.perf_counter()
            frame_idx += 1

            # Downscale frame for inference (640 width keeps speed high while preserving accuracy)
            infer_w = 640
            infer_h = int(h * (infer_w / float(w)))
            scale_factor = float(w) / float(infer_w)
            small_frame = cv2.resize(frame, (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)

            # Single-pass execution every frame
            # Run SCRFD every 2 frames, YOLO pose tracking every frame
            run_scrfd = (frame_idx % 2 == 1) or len(cached_detections) == 0

            latest_pose_results = []
            if pose_enabled and detector.pose_fusor:
                try:
                    # Run YOLOv8-pose ONCE on small frame
                    latest_pose_results = detector.pose_fusor.pose_model(small_frame, verbose=False, conf=conf_thresh)
                    cached_pose_results = latest_pose_results
                except Exception:
                    pass

            if run_scrfd:
                # Run SCRFD on downscaled frame
                try:
                    raw_dets = detector.detect(small_frame)
                except Exception:
                    raw_dets = []

                # Scale detections back to full resolution
                scaled_detections = []
                for d in raw_dets:
                    box = d["bbox"]
                    sx1, sy1, sx2, sy2 = [val * scale_factor for val in box]
                    scaled_lms = []
                    for lm in d.get("landmarks", []):
                        scaled_lms.append([lm[0] * scale_factor, lm[1] * scale_factor])
                    scaled_detections.append({
                        "bbox": [round(sx1, 1), round(sy1, 1), round(sx2, 1), round(sy2, 1)],
                        "score": d.get("score", 0.9),
                        "landmarks": scaled_lms,
                        "source": d.get("source", "scrfd_34g"),
                    })

                # Fuse with head keypoints from YOLO if pose is enabled
                if pose_enabled and detector.pose_fusor and cached_pose_results:
                    for res in cached_pose_results:
                        if res.keypoints is None or len(res.keypoints.data) == 0:
                            continue
                        for kp in res.keypoints.data.cpu().numpy():
                            head_kps = kp[:5]
                            shoulders = kp[5:7]
                            valid_head = head_kps[head_kps[:, 2] >= 0.38]
                            valid_shoulders = shoulders[shoulders[:, 2] >= 0.45]
                            sw = float(np.linalg.norm(valid_shoulders[0, :2] - valid_shoulders[1, :2])) if len(valid_shoulders) == 2 else 0.0

                            if len(valid_head) >= 2:
                                cx = float(np.mean(valid_head[:, 0])) * scale_factor
                                cy = float(np.mean(valid_head[:, 1])) * scale_factor
                                span = float(max(np.ptp(valid_head[:, 0]), np.ptp(valid_head[:, 1]), 16.0)) * scale_factor
                                rad = max(span * 1.35, sw * 0.38 * scale_factor, 22.0)
                            elif len(valid_head) == 1 and sw > 20.0:
                                cx = float(valid_head[0, 0]) * scale_factor
                                cy = float(valid_head[0, 1]) * scale_factor
                                rad = max(sw * 0.40 * scale_factor, 24.0)
                            else:
                                continue

                            hx1 = int(max(0, cx - rad))
                            hy1 = int(max(0, cy - rad * 1.25))
                            hx2 = int(min(w, cx + rad))
                            hy2 = int(min(h, cy + rad * 1.15))
                            h_box = [hx1, hy1, hx2, hy2]

                            # Match against SCRFD boxes
                            matched = any(
                                detector.pose_fusor._iou(h_box, d["bbox"]) > 0.25
                                for d in scaled_detections
                            )
                            if not matched:
                                scaled_detections.append({
                                    "bbox": h_box,
                                    "score": 0.82,
                                    "landmarks": [],
                                    "source": "pose_fusor_head",
                                })

                cached_detections = scaled_detections

            detections_to_render = cached_detections
            detected_count = len(detections_to_render)

            # 2. Apply Obfuscation if enabled
            if redaction_enabled and detected_count > 0:
                dilated_boxes = [dilate_bbox(d["bbox"], (h, w), ratio=dilation_ratio) for d in detections_to_render]
                display_frame = apply_obfuscation(frame, dilated_boxes, mode=blur_mode)
            else:
                display_frame = frame.copy()

            # 3. Render AI Keypoints & Pose Skeleton if enabled (reusing cached single-pass pose result)
            if keypoints_enabled:
                display_frame = draw_ai_landmarks(
                    display_frame,
                    detections_to_render,
                    cached_pose_results,
                    scale_factor=scale_factor,
                )

            # Compute smooth FPS
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
                keypoints_enabled=keypoints_enabled,
                dilation_ratio=dilation_ratio,
                mode=blur_mode,
            )

            cv2.imshow(window_name, hud_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                logger.info("Exiting live demo.")
                break
            elif key == ord(" "):
                redaction_enabled = not redaction_enabled
                logger.info("Redaction toggled: %s", "ENABLED" if redaction_enabled else "DISABLED")
            elif key in (ord("k"), ord("K")):
                keypoints_enabled = not keypoints_enabled
                logger.info("Keypoints overlay toggled: %s", "ENABLED" if keypoints_enabled else "DISABLED")
            elif key in (ord("p"), ord("P")):
                pose_enabled = not pose_enabled
                logger.info("Pose tracking toggled: %s", "ENABLED" if pose_enabled else "DISABLED")
            elif key == ord("["):
                dilation_ratio = max(0.04, round(dilation_ratio - 0.02, 2))
                logger.info("Dilation reduced to: %.2f", dilation_ratio)
            elif key == ord("]"):
                dilation_ratio = min(0.40, round(dilation_ratio + 0.02, 2))
                logger.info("Dilation increased to: %.2f", dilation_ratio)
            elif key in (ord("m"), ord("M")):
                blur_mode = "mosaic" if blur_mode == "gaussian_or_mosaic" else "gaussian_or_mosaic"
                logger.info("Blur mode changed to: %s", blur_mode)

    finally:
        stream.release()
        cv2.destroyAllWindows()
        logger.info("Webcam stream released and windows closed.")


if __name__ == "__main__":
    run_live_demo()
