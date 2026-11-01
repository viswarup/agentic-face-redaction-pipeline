"""Tracking and Bidirectional Trajectory Interpolation Module."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from app.config import config

logger = logging.getLogger("anonymizer.tracker")


def compute_iou(box1: list[float] | np.ndarray, box2: list[float] | np.ndarray) -> float:
    """Compute Intersection-over-Union (IoU) between two [x1, y1, x2, y2] bounding boxes."""
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def compute_iou_cost_matrix(
    boxes_a: list[list[float]] | np.ndarray,
    boxes_b: list[list[float]] | np.ndarray,
) -> np.ndarray:
    """Compute (1.0 - IoU) cost matrix between two sets of bounding boxes."""
    n_a = len(boxes_a)
    n_b = len(boxes_b)
    cost = np.ones((n_a, n_b), dtype=np.float32)
    for i in range(n_a):
        for j in range(n_b):
            cost[i, j] = 1.0 - compute_iou(boxes_a[i], boxes_b[j])
    return cost


class KalmanBoxTracker:
    """
    Linear Kalman Filter state estimator for tracking [x1, y1, x2, y2] bounding boxes
    with constant velocity dynamics.
    """

    count = 0

    def __init__(self, bbox: list[float], score: float = 1.0):
        KalmanBoxTracker.count += 1
        self.track_id = KalmanBoxTracker.count

        # State: [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
        self.x = np.zeros(8, dtype=np.float32)
        self.x[:4] = bbox

        self.F = np.eye(8, dtype=np.float32)
        self.F[:4, 4:] = np.eye(4, dtype=np.float32)

        self.H = np.zeros((4, 8), dtype=np.float32)
        self.H[:4, :4] = np.eye(4, dtype=np.float32)

        self.P = np.eye(8, dtype=np.float32) * 10.0
        self.P[4:, 4:] *= 100.0  # High initial uncertainty in velocity

        self.Q = np.eye(8, dtype=np.float32) * 1.0
        self.Q[4:, 4:] *= 0.1

        self.R = np.eye(4, dtype=np.float32) * 1.0

        self.score = score
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

    def predict(self) -> list[float]:
        """Predict bounding box state for next time step."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.get_state()

    def update(self, bbox: list[float], score: float = 1.0) -> None:
        """Correct state with new measurement bbox."""
        z = np.array(bbox, dtype=np.float32)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(8, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

        self.score = score
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

    def get_state(self) -> list[float]:
        """Return current bounding box [x1, y1, x2, y2]."""
        x1, y1, x2, y2 = self.x[:4]
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return [float(x1), float(y1), float(x2), float(y2)]


class VideoFaceTracker:
    """
    ByteTrack-inspired multi-stage association tracker using Kalman Filters and IoU matching.
    """

    def __init__(
        self,
        high_thresh: float = 0.30,
        low_thresh: float = 0.15,
        match_thresh: float = 0.70,  # 1.0 - IoU cost threshold (IoU >= 0.30)
        max_age: int = 10,
    ):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.max_age = max_age
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Update tracker with detections for current frame.
        Each detection is a dict with 'bbox' and 'score'.

        Returns:
            list of dicts: [
                {
                    "track_id": int,
                    "bbox": [x1, y1, x2, y2],
                    "score": float,
                    "source": "detection"
                },
                ...
            ]
        """
        self.frame_count += 1

        # 1. Predict new locations of existing trackers
        for trk in self.trackers:
            trk.predict()

        # 2. Partition detections into high and low confidence groups
        high_dets = [d for d in detections if d["score"] >= self.high_thresh]
        low_dets = [d for d in detections if self.low_thresh <= d["score"] < self.high_thresh]

        # 3. First association: active trackers with high confidence detections
        unmatched_trackers: list[int] = list(range(len(self.trackers)))
        unmatched_high_dets: list[int] = list(range(len(high_dets)))
        matched_pairs_1: list[tuple[int, int]] = []

        if self.trackers and high_dets:
            trk_boxes = [t.get_state() for t in self.trackers]
            det_boxes = [d["bbox"] for d in high_dets]
            cost_matrix = compute_iou_cost_matrix(trk_boxes, det_boxes)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] <= self.match_thresh:
                    matched_pairs_1.append((r, c))

            matched_r = {p[0] for p in matched_pairs_1}
            matched_c = {p[1] for p in matched_pairs_1}
            unmatched_trackers = [i for i in range(len(self.trackers)) if i not in matched_r]
            unmatched_high_dets = [j for j in range(len(high_dets)) if j not in matched_c]

        # Update stage 1 matches
        for t_idx, d_idx in matched_pairs_1:
            det = high_dets[d_idx]
            self.trackers[t_idx].update(det["bbox"], det["score"])

        # 4. Second association: remaining unmatched trackers with low confidence detections
        matched_pairs_2: list[tuple[int, int]] = []
        if unmatched_trackers and low_dets:
            sub_trk_boxes = [self.trackers[i].get_state() for i in unmatched_trackers]
            low_boxes = [d["bbox"] for d in low_dets]
            cost_matrix_low = compute_iou_cost_matrix(sub_trk_boxes, low_boxes)

            row_ind, col_ind = linear_sum_assignment(cost_matrix_low)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix_low[r, c] <= self.match_thresh:
                    original_trk_idx = unmatched_trackers[r]
                    matched_pairs_2.append((original_trk_idx, c))

            matched_r2 = {p[0] for p in matched_pairs_2}
            unmatched_trackers = [i for i in unmatched_trackers if i not in matched_r2]

        # Update stage 2 matches
        for t_idx, d_idx in matched_pairs_2:
            det = low_dets[d_idx]
            self.trackers[t_idx].update(det["bbox"], det["score"])

        # 5. Initialize new trackers for remaining unmatched high confidence detections
        for d_idx in unmatched_high_dets:
            det = high_dets[d_idx]
            self.trackers.append(KalmanBoxTracker(det["bbox"], det["score"]))

        # 6. Remove dead trackers that exceeded max_age
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        # 7. Collect active outputs for this frame (updated in this frame or coasting shortly)
        outputs: list[dict[str, Any]] = []
        for t in self.trackers:
            if t.time_since_update == 0:
                outputs.append(
                    {
                        "track_id": t.track_id,
                        "bbox": [round(v, 2) for v in t.get_state()],
                        "score": round(float(t.score), 4),
                        "source": "detection",
                    }
                )

        return outputs


class TwoPassTrajectorySmoother:
    """
    Offline two-pass trajectory smoother:
    1. Bidirectional linear gap interpolation for tracks missing up to max_gap (12) frames.
    2. Forward and backward coasting by coast_frames (3) at track birth and death
       when faces appear/disappear near the interior of the camera view.
    """

    def __init__(
        self,
        max_gap: int = 12,
        coast_frames: int = 3,
        interior_margin: int = 25,
    ):
        self.max_gap = max_gap
        self.coast_frames = coast_frames
        self.interior_margin = interior_margin

    def _is_near_interior(self, bbox: list[float], frame_w: int, frame_h: int) -> bool:
        """Return True if bounding box is inside the interior (not touching frame edges)."""
        x1, y1, x2, y2 = bbox
        m = self.interior_margin
        return (x1 > m) and (y1 > m) and (x2 < frame_w - m) and (y2 < frame_h - m)

    def smooth(
        self,
        raw_frame_detections: list[list[dict[str, Any]]],
        frame_shape: tuple[int, int],
    ) -> list[list[dict[str, Any]]]:
        """
        Apply bidirectional gap interpolation and entry/exit edge coasting.

        Args:
            raw_frame_detections: per-frame list of detected/tracked boxes.
            frame_shape: (height, width) of the video frames.

        Returns:
            smoothed_frame_detections: per-frame list of boxes including interpolated coasters.
        """
        total_frames = len(raw_frame_detections)
        if total_frames == 0:
            return []

        frame_h, frame_w = frame_shape[:2]

        # Group detections by track_id: tracks[track_id][frame_idx] = box_info
        tracks: dict[int, dict[int, dict[str, Any]]] = {}
        for f_idx, frame_boxes in enumerate(raw_frame_detections):
            for b in frame_boxes:
                tid = b.get("track_id")
                if tid is None:
                    continue
                if tid not in tracks:
                    tracks[tid] = {}
                tracks[tid][f_idx] = dict(b)

        # Pass 1: Bidirectional Linear Gap Interpolation
        for tid, frame_map in tracks.items():
            sorted_frames = sorted(frame_map.keys())
            if len(sorted_frames) < 2:
                continue

            for idx in range(len(sorted_frames) - 1):
                t1 = sorted_frames[idx]
                t2 = sorted_frames[idx + 1]
                gap = t2 - t1
                if 1 < gap <= self.max_gap:
                    b1 = frame_map[t1]["bbox"]
                    b2 = frame_map[t2]["bbox"]
                    s1 = frame_map[t1].get("score", 0.5)
                    s2 = frame_map[t2].get("score", 0.5)

                    for t in range(t1 + 1, t2):
                        alpha = (t - t1) / float(gap)
                        interp_box = [
                            round((1.0 - alpha) * b1[k] + alpha * b2[k], 2)
                            for k in range(4)
                        ]
                        interp_score = round((1.0 - alpha) * s1 + alpha * s2, 4)
                        frame_map[t] = {
                            "track_id": tid,
                            "bbox": interp_box,
                            "score": interp_score,
                            "source": "interpolated_coaster",
                        }

        # Pass 2: Interior Birth & Death Coasting
        for tid, frame_map in tracks.items():
            sorted_frames = sorted(frame_map.keys())
            if not sorted_frames:
                continue

            t_birth = sorted_frames[0]
            t_death = sorted_frames[-1]

            birth_box = frame_map[t_birth]["bbox"]
            death_box = frame_map[t_death]["bbox"]
            birth_score = frame_map[t_birth].get("score", 0.5)
            death_score = frame_map[t_death].get("score", 0.5)

            # Coast backward at birth if in interior
            if self._is_near_interior(birth_box, frame_w, frame_h):
                for t in range(max(0, t_birth - self.coast_frames), t_birth):
                    if t not in frame_map:
                        frame_map[t] = {
                            "track_id": tid,
                            "bbox": list(birth_box),
                            "score": birth_score,
                            "source": "coast_backward",
                        }

            # Coast forward at death if in interior
            if self._is_near_interior(death_box, frame_w, frame_h):
                for t in range(t_death + 1, min(total_frames, t_death + self.coast_frames + 1)):
                    if t not in frame_map:
                        frame_map[t] = {
                            "track_id": tid,
                            "bbox": list(death_box),
                            "score": death_score,
                            "source": "coast_forward",
                        }

        # Reconstruct per-frame detections
        smoothed_results: list[list[dict[str, Any]]] = [[] for _ in range(total_frames)]
        for tid, frame_map in tracks.items():
            for f_idx, box_info in frame_map.items():
                if 0 <= f_idx < total_frames:
                    smoothed_results[f_idx].append(box_info)

        # Sort boxes within each frame by track_id
        for f_idx in range(total_frames):
            smoothed_results[f_idx].sort(key=lambda x: x.get("track_id", 0))

        return smoothed_results
