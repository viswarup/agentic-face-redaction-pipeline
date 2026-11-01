"""Primary Face Detector module strictly running SCRFD-34G (Fail-Closed Architecture)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

from app.config import config

logger = logging.getLogger("anonymizer.detector")


class PrimaryFaceDetector:
    """
    Primary face detection engine running strictly on SCRFD-34G at high recall.
    Implements a strict fail-closed design: no fallback models or mock detectors are permitted.
    If SCRFD-34G fails to initialize or encounters a runtime error, an explicit
    RuntimeError("MODEL_OR_PROCESSING_ERROR: ...") is raised.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        conf_thresh: Optional[float] = None,
        model_path: Optional[str | Path] = None,
        providers: Optional[list[str]] = None,
        input_size: tuple[int, int] = (640, 640),
        use_pose_fallback: bool = False,
    ):
        self.model_name = "scrfd_34g" + ("+yolov8_pose" if use_pose_fallback else "")
        self.conf_thresh = conf_thresh if conf_thresh is not None else config.PRIMARY_CONF_THRESH
        self.providers = providers or ["CPUExecutionProvider"]
        self.input_size = input_size
        self.loaded_weight_path: Optional[str] = None
        self._model: Any = None
        self.use_pose_fallback = use_pose_fallback
        self.pose_fusor: Optional[Any] = None

        if self.use_pose_fallback:
            from app.pipeline.pose_head_fallback import PoseHeadFusor
            self.pose_fusor = PoseHeadFusor()

        self._initialize_detector(model_path)

    def _initialize_detector(self, explicit_path: Optional[str | Path]) -> None:
        """Resolve and load the SCRFD-34G ONNX model. Fails closed if not found."""
        from insightface.model_zoo import get_model

        candidates: list[Path] = []
        if explicit_path:
            candidates.append(Path(explicit_path))

        project_root = config.BASE_DIR
        candidates.append(project_root / "weights" / "scrfd_34g.onnx")

        loaded = False
        last_error: Optional[Exception] = None

        for path in candidates:
            if path.exists() and path.is_file():
                try:
                    logger.info("Initializing SCRFD-34G model from %s", path)
                    model = get_model(str(path), providers=self.providers)
                    model.prepare(ctx_id=-1, input_size=self.input_size)
                    self._model = model
                    self.loaded_weight_path = str(path)
                    loaded = True
                    logger.info("Successfully initialized SCRFD-34G detector from %s", self.loaded_weight_path)
                    break
                except Exception as ex:
                    last_error = ex
                    logger.error("Failed to load SCRFD-34G checkpoint %s: %s", path, ex)

        if not loaded or self._model is None:
            err_msg = (
                f"MODEL_OR_PROCESSING_ERROR: SCRFD-34G weights not found or failed to load. "
                f"Checked candidates: {[str(c) for c in candidates]}. Last error: {last_error}"
            )
            logger.error("Fail-closed initialization: %s", err_msg)
            raise RuntimeError(err_msg)

    def detect(self, image_bgr: np.ndarray, conf_thresh: Optional[float] = None) -> list[dict[str, Any]]:
        """
        Run SCRFD-34G face detection on a single BGR image frame.

        Returns:
            list of dicts: [
                {
                    "bbox": [x1, y1, x2, y2],
                    "score": float,
                    "landmarks": [[lx, ly], ...]
                },
                ...
            ]
        """
        if image_bgr is None or image_bgr.size == 0:
            return []

        if self._model is None:
            raise RuntimeError("MODEL_OR_PROCESSING_ERROR: SCRFD-34G model is not initialized.")

        h, w = image_bgr.shape[:2]
        threshold = conf_thresh if conf_thresh is not None else self.conf_thresh
        detections: list[dict[str, Any]] = []

        try:
            try:
                bboxes, kpss = self._model.detect(image_bgr, det_thresh=threshold)
            except TypeError:
                bboxes, kpss = self._model.detect(image_bgr)
        except Exception as e:
            logger.exception("SCRFD-34G detection inference failure: %s", e)
            raise RuntimeError(f"MODEL_OR_PROCESSING_ERROR: SCRFD-34G inference failed: {e}") from e

        if bboxes is None or len(bboxes) == 0:
            return []

        for i, box in enumerate(bboxes):
            score = float(box[4])
            if score < threshold:
                continue

            x1 = float(np.clip(box[0], 0, w))
            y1 = float(np.clip(box[1], 0, h))
            x2 = float(np.clip(box[2], 0, w))
            y2 = float(np.clip(box[3], 0, h))

            if x2 <= x1 or y2 <= y1:
                continue

            landmarks: list[list[float]] = []
            if kpss is not None and i < len(kpss) and kpss[i] is not None:
                landmarks = [[float(pt[0]), float(pt[1])] for pt in kpss[i]]

            detections.append(
                {
                    "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    "score": round(score, 4),
                    "landmarks": landmarks,
                    "source": "scrfd_34g",
                }
            )

        if self.use_pose_fallback and self.pose_fusor:
            detections = self.pose_fusor.fuse_with_scrfd(image_bgr, detections)

        return detections
