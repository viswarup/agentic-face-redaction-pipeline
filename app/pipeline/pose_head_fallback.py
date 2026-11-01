"""Body & Head Keypoint Detection (YOLOv8n-Pose) Fusion Layer."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
from ultralytics import YOLO

from app.config import config

logger = logging.getLogger("anonymizer.pose")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")


class PoseHeadFusor:
    """
    Fuses YOLOv8n-Pose body/shoulder/cranial keypoints with SCRFD-34G face detections
    to recover motion-blurred, heavily occluded, and steep-profile faces.
    """

    def __init__(
        self,
        model_name: Optional[str | Path] = None,
        conf_thresh: float = 0.35,
    ):
        weights_candidates = [
            Path(model_name) if model_name else None,
            config.BASE_DIR / "weights" / "yolov8n-pose.pt",
            config.BASE_DIR / "yolov8n-pose.pt",
            Path("yolov8n-pose.pt"),
        ]
        resolved_path = "yolov8n-pose.pt"
        for cand in weights_candidates:
            if cand and cand.exists():
                resolved_path = str(cand)
                break

        self.pose_model = YOLO(resolved_path)
        self.conf_thresh = conf_thresh
        logger.info("Initialized PoseHeadFusor with %s (conf_thresh=%.2f)", resolved_path, self.conf_thresh)

    @staticmethod
    def _iou(boxA: list[float] | tuple[float, ...], boxB: list[float] | tuple[float, ...]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter = max(0.0, xB - xA) * max(0.0, yB - yA)
        areaA = max(1.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        areaB = max(1.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
        return float(inter / (areaA + areaB - inter))

    def infer_head_boxes(self, frame_bgr: np.ndarray) -> list[dict[str, Any]]:
        """
        Infer cranial/head bounding boxes from body and head keypoints.
        """
        h, w = frame_bgr.shape[:2]
        results = self.pose_model(frame_bgr, verbose=False, conf=self.conf_thresh)[0]
        pose_heads: list[dict[str, Any]] = []

        if results.keypoints is None or len(results.keypoints.data) == 0:
            return pose_heads

        for kp in results.keypoints.data.cpu().numpy():
            # COCO keypoints: 0=nose, 1=L_eye, 2=R_eye, 3=L_ear, 4=R_ear, 5=L_shoulder, 6=R_shoulder
            head_kps = kp[:5]
            shoulders = kp[5:7]

            valid_head = head_kps[head_kps[:, 2] >= 0.38]
            valid_shoulders = shoulders[shoulders[:, 2] >= 0.45]

            # Estimate shoulder width if both shoulders are visible
            shoulder_width = 0.0
            if len(valid_shoulders) == 2:
                shoulder_width = float(np.linalg.norm(valid_shoulders[0, :2] - valid_shoulders[1, :2]))

            if len(valid_head) >= 2:
                cx = float(np.mean(valid_head[:, 0]))
                cy = float(np.mean(valid_head[:, 1]))
                kp_span = float(max(np.ptp(valid_head[:, 0]), np.ptp(valid_head[:, 1]), 16.0))
                radius = max(kp_span * 1.35, shoulder_width * 0.38, 18.0)
                conf = float(np.mean(valid_head[:, 2]))
            elif len(valid_head) == 1 and shoulder_width > 25.0:
                # Severe occlusion fallback anchored by single head keypoint + shoulder scale
                cx, cy = float(valid_head[0, 0]), float(valid_head[0, 1])
                radius = max(shoulder_width * 0.40, 20.0)
                conf = float(valid_head[0, 2]) * 0.85
            else:
                continue

            x1 = int(max(0, cx - radius))
            y1 = int(max(0, cy - radius * 1.25))
            x2 = int(min(w, cx + radius))
            y2 = int(min(h, cy + radius * 1.15))

            if (x2 - x1) >= 12 and (y2 - y1) >= 12:
                pose_heads.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "score": round(conf, 3),
                        "source": "pose_head_fallback",
                        "landmarks": [],
                    }
                )

        return pose_heads

    def fuse_with_scrfd(
        self, frame_bgr: np.ndarray, scrfd_detections: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Merges SCRFD-34G detections with YOLOv8n-Pose head boxes:
          - If Pose head overlaps an SCRFD box (IoU >= 0.20), expand to their union to prevent partial leaks.
          - If Pose head has no matching SCRFD box (IoU < 0.20), add it as a recovered missed face.
        """
        pose_heads = self.infer_head_boxes(frame_bgr)
        fused = [dict(d, source=d.get("source", "scrfd_34g")) for d in scrfd_detections]

        for ph in pose_heads:
            p_box = ph["bbox"]
            matched = False
            for det in fused:
                if self._iou(p_box, det["bbox"]) >= 0.20:
                    # Union box prevents tight-box residual profile/chin leaks
                    s_box = det["bbox"]
                    det["bbox"] = [
                        min(s_box[0], p_box[0]),
                        min(s_box[1], p_box[1]),
                        max(s_box[2], p_box[2]),
                        max(s_box[3], p_box[3]),
                    ]
                    det["source"] = "scrfd_pose_union"
                    matched = True
                    break

            if not matched:
                fused.append(ph)

        return fused
