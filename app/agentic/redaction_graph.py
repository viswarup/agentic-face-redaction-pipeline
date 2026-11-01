"""
Agentic Self-Healing Face Redaction State Graph.

Implements an Evaluator-Optimizer / Reflection Loop architecture:
  1. Perception & Telemetry: Assesses scene luminance, blur variance, dynamic range, and shadow ratio.
  2. Dynamic Tool Calling: Adaptively triggers CLAHE illumination enhancement, tunes detection thresholds,
     and integrates skeletal keypoint cranial inference.
  3. Obfuscation Rendering: Applies heavy irreversible dual-layer mosaic/Gaussian blur.
  4. Independent Evaluator (Audit): Runs RetinaFace QA Verifier on the obfuscated output to catch residual leaks.
  5. Reflection & Self-Correction (Optimizer): Diagnoses leak type (Peripheral vs Blindspot),
     applies targeted corrections (expansion, box injection, exposure boost), and re-audits.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

from app.config import config
from app.pipeline.detector import PrimaryFaceDetector
from app.pipeline.pose_head_fallback import PoseHeadFusor
from app.pipeline.redactor import apply_obfuscation, dilate_bbox
from app.pipeline.tracker_interpolator import compute_iou
from app.pipeline.verifier import RetinaFaceVerifier

logger = logging.getLogger("agentic.graph")


@dataclass
class SceneTelemetry:
    mean_luminance: float = 0.0
    blur_variance: float = 0.0
    contrast_std: float = 0.0
    dark_pixel_ratio: float = 0.0
    is_low_light: bool = False
    is_blurry: bool = False
    is_high_contrast: bool = False


@dataclass
class CorrectionAction:
    iteration: int
    action_type: str  # "DILATION_EXPANSION", "BLINDSPOT_INJECTION", "LOW_LIGHT_ADAPTATION"
    reason: str
    target_box: list[float]
    resulting_box: list[float]


@dataclass
class AgentState:
    raw_image: np.ndarray
    telemetry: SceneTelemetry = field(default_factory=SceneTelemetry)
    active_boxes: list[list[float]] = field(default_factory=list)
    box_dilation_ratios: list[float] = field(default_factory=list)
    dilated_boxes: list[list[int]] = field(default_factory=list)
    redacted_image: Optional[np.ndarray] = None
    audit_leaks: list[dict[str, Any]] = field(default_factory=list)
    correction_history: list[CorrectionAction] = field(default_factory=list)
    iterations_performed: int = 0
    audit_passed: bool = False
    total_latency_ms: float = 0.0


class AgenticRedactionGraph:
    """
    Self-Healing Face Redaction Agent modeled as a State Graph with closed-loop feedback.
    """

    def __init__(
        self,
        max_iterations: int = 2,
        base_conf_thresh: float = 0.25,
        verifier_conf_thresh: float = 0.30,
        default_dilation: float = 0.28,
    ):
        self.max_iterations = max_iterations
        self.base_conf_thresh = base_conf_thresh
        self.verifier_conf_thresh = verifier_conf_thresh
        self.default_dilation = default_dilation

        logger.info("Initializing Agentic Redaction Graph components...")
        self.primary_detector = PrimaryFaceDetector(
            conf_thresh=base_conf_thresh, use_pose_fallback=True
        )
        self.verifier = RetinaFaceVerifier(conf_thresh=verifier_conf_thresh)
        logger.info("Agentic Redaction Graph successfully initialized.")

    def perceive_telemetry(self, bgr_img: np.ndarray) -> SceneTelemetry:
        """Node 1: Perception & Environmental Telemetry."""
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        total_px = max(1, h * w)

        mean_lum = float(np.mean(gray))
        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        contrast_std = float(np.std(gray))
        dark_ratio = float(np.sum(gray < 35)) / float(total_px)

        is_low_light = mean_lum < 55.0 or dark_ratio > 0.35
        is_blurry = blur_var < 85.0
        is_high_contrast = contrast_std > 65.0

        return SceneTelemetry(
            mean_luminance=round(mean_lum, 2),
            blur_variance=round(blur_var, 2),
            contrast_std=round(contrast_std, 2),
            dark_pixel_ratio=round(dark_ratio, 3),
            is_low_light=is_low_light,
            is_blurry=is_blurry,
            is_high_contrast=is_high_contrast,
        )

    def tool_enhance_visibility(self, bgr_img: np.ndarray, telemetry: SceneTelemetry) -> np.ndarray:
        """Tool: Adaptive Contrast and Illumination Enhancement (CLAHE)."""
        if not telemetry.is_low_light:
            return bgr_img

        lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        return enhanced_bgr

    def tool_dynamic_detect(
        self, bgr_img: np.ndarray, telemetry: SceneTelemetry
    ) -> tuple[list[dict[str, Any]], float]:
        """Tool: Perception-Driven Dynamic Threshold Face Detector."""
        thresh = self.base_conf_thresh
        if telemetry.is_low_light:
            thresh = min(thresh, 0.18)
        if telemetry.is_blurry:
            thresh = min(thresh, 0.18)

        # Detect using Primary Detector (with Pose Fusion enabled)
        detections = self.primary_detector.detect(bgr_img, conf_thresh=thresh)
        return detections, thresh

    def render_obfuscation(
        self, raw_img: np.ndarray, dilated_boxes: list[list[int]]
    ) -> np.ndarray:
        """Node: Heavy Irreversible Redaction Renderer."""
        return apply_obfuscation(raw_img, dilated_boxes, mode="gaussian_or_mosaic")

    def audit_evaluator(
        self, redacted_img: np.ndarray
    ) -> list[dict[str, Any]]:
        """Node: Independent RetinaFace QA Audit Evaluator."""
        audit_detections = self.verifier.detect(redacted_img, conf_thresh=self.verifier_conf_thresh)
        return audit_detections

    def run(self, raw_bgr: np.ndarray) -> AgentState:
        """
        Execute the full Agentic Self-Healing Redaction Graph on an input image.
        """
        start_time = time.perf_counter()
        h, w = raw_bgr.shape[:2]

        # 1. Perception
        telemetry = self.perceive_telemetry(raw_bgr)

        # 2. Adaptive Illumination Tool
        eval_img = raw_bgr
        if telemetry.is_low_light:
            eval_img = self.tool_enhance_visibility(raw_bgr, telemetry)

        # 3. Dynamic Perception-Tuned Primary Detection (with Pose Fusion)
        primary_dets, active_thresh = self.tool_dynamic_detect(eval_img, telemetry)
        active_boxes: list[list[float]] = [list(d["bbox"]) for d in primary_dets]
        box_ratios: list[float] = [self.default_dilation] * len(active_boxes)

        state = AgentState(
            raw_image=raw_bgr,
            telemetry=telemetry,
            active_boxes=active_boxes,
            box_dilation_ratios=box_ratios,
            dilated_boxes=[],
            redacted_image=None,
            audit_leaks=[],
            correction_history=[],
            iterations_performed=0,
            audit_passed=False,
            total_latency_ms=0.0,
        )

        if telemetry.is_low_light:
            state.correction_history.append(
                CorrectionAction(
                    iteration=0,
                    action_type="LOW_LIGHT_ADAPTATION",
                    reason=f"Telemetry detected dark scene (lum={telemetry.mean_luminance}, dark_ratio={telemetry.dark_pixel_ratio}) -> Applied CLAHE + lowered threshold to {active_thresh}",
                    target_box=[0.0, 0.0, float(w), float(h)],
                    resulting_box=[0.0, 0.0, float(w), float(h)],
                )
            )

        # 4. Evaluator-Optimizer / Reflection Loop
        for iteration in range(self.max_iterations + 1):
            state.iterations_performed = iteration

            # Dilate all active boxes with their assigned dilation ratio
            dilated = []
            for b, r in zip(state.active_boxes, state.box_dilation_ratios):
                dilated.append(dilate_bbox(b, (h, w), ratio=r))
            state.dilated_boxes = dilated

            # Render Redaction
            state.redacted_image = self.render_obfuscation(raw_bgr, state.dilated_boxes)

            # Independent Audit on Redacted Output
            audit_leaks = self.audit_evaluator(state.redacted_image)
            state.audit_leaks = audit_leaks

            # If no faces/features detected leaking through blur, audit passes!
            if len(audit_leaks) == 0:
                state.audit_passed = True
                break

            # Stop if reached max iteration bound
            if iteration >= self.max_iterations:
                break

            # Reflection & Self-Correction Node
            for leak in audit_leaks:
                leak_box = leak["bbox"]

                # Diagnose leak type: check overlap with existing boxes
                overlaps = [compute_iou(leak_box, b) for b in state.active_boxes]
                max_iou = max(overlaps) if overlaps else 0.0
                best_idx = int(np.argmax(overlaps)) if overlaps else -1

                if max_iou >= 0.15 and best_idx >= 0:
                    # Type A: Peripheral Leak - the face was detected, but boundary leaked
                    orig_ratio = state.box_dilation_ratios[best_idx]
                    new_ratio = min(0.45, orig_ratio + 0.10)
                    state.box_dilation_ratios[best_idx] = new_ratio
                    state.correction_history.append(
                        CorrectionAction(
                            iteration=iteration,
                            action_type="DILATION_EXPANSION",
                            reason=f"Verifier detected peripheral leak (IoU={max_iou:.2f}) -> Expanded dilation ratio from {orig_ratio:.2f} to {new_ratio:.2f}",
                            target_box=state.active_boxes[best_idx],
                            resulting_box=state.active_boxes[best_idx],
                        )
                    )
                else:
                    # Type B: Missed Face Blindspot - completely undetected face caught by verifier
                    injected_box = [
                        max(0.0, float(leak_box[0])),
                        max(0.0, float(leak_box[1])),
                        min(float(w), float(leak_box[2])),
                        min(float(h), float(leak_box[3])),
                    ]
                    state.active_boxes.append(injected_box)
                    state.box_dilation_ratios.append(self.default_dilation)
                    state.correction_history.append(
                        CorrectionAction(
                            iteration=iteration,
                            action_type="BLINDSPOT_INJECTION",
                            reason=f"Verifier caught blindspot face leak (conf={leak.get('score', 0):.2f}) -> Injected new redaction target",
                            target_box=leak_box,
                            resulting_box=injected_box,
                        )
                    )

        state.total_latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        return state
