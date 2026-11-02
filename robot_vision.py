"""Heuristic road/obstacle overlay for a single downward-looking robot camera.

This is a visual aid, not a safety system. It estimates a drivable surface from
the colour/texture of the lower image region and tracks contrasting non-ground
regions. It can fail on shadows, wet floors, glass, low texture, motion blur and
unusual lighting. Never use it as the only obstacle stop.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass(frozen=True)
class RobotSettings:
    horizon_fraction: float = 0.30
    seed_bottom_fraction: float = 0.22
    seed_side_fraction: float = 0.25
    min_obstacle_area_fraction: float = 0.0015
    max_obstacle_area_fraction: float = 0.35
    smooth_alpha: float = 0.70
    max_missed_frames: int = 6

    def __post_init__(self):
        if not 0.05 <= self.horizon_fraction <= 0.70:
            raise ValueError('horizon_fraction must be between 0.05 and 0.70')
        if not 0.05 <= self.seed_bottom_fraction <= 0.50:
            raise ValueError('seed_bottom_fraction must be between 0.05 and 0.50')
        if not 0.05 <= self.seed_side_fraction <= 0.45:
            raise ValueError('seed_side_fraction must be between 0.05 and 0.45')
        if not 0 < self.min_obstacle_area_fraction < self.max_obstacle_area_fraction <= 0.8:
            raise ValueError('invalid obstacle area fractions')
        if not 0 < self.smooth_alpha <= 1:
            raise ValueError('smooth_alpha must be in (0, 1]')
        if self.max_missed_frames < 0:
            raise ValueError('max_missed_frames must be non-negative')


@dataclass
class ObstacleTrack:
    track_id: int
    bbox: tuple
    centroid: tuple
    missed: int = 0
    hits: int = 1


@dataclass
class RobotInfo:
    road_fraction: float
    obstacle_count: int
    center_blocked: bool
    tracks: list = field(default_factory=list)


def _bbox_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class RoadObstacleDetector:
    """Stateful per-camera detector. Call process() from one thread only."""

    def __init__(self, settings=None):
        self.settings = settings or RobotSettings()
        self._smooth_road = None
        self._tracks = []
        self._next_id = 1

    def reset(self):
        self._smooth_road = None
        self._tracks.clear()
        self._next_id = 1

    def process(self, frame, sensitivity=50):
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError('expected a BGR frame')
        s = self.settings
        sensitivity = int(max(1, min(100, sensitivity)))
        out = frame.copy()
        h, w = out.shape[:2]
        y0 = int(h * s.horizon_fraction)
        roi = out[y0:, :]
        rh, rw = roi.shape[:2]
        if rh < 8 or rw < 8:
            return out, RobotInfo(0.0, 0, False, [])

        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
        seed_y0 = int(rh * (1.0 - s.seed_bottom_fraction))
        seed_x0 = int(rw * s.seed_side_fraction)
        seed_x1 = max(seed_x0 + 1, int(rw * (1.0 - s.seed_side_fraction)))
        seed = lab[seed_y0:, seed_x0:seed_x1].reshape(-1, 3)
        if seed.shape[0] < 16:
            return out, RobotInfo(0.0, 0, False, [])
        median = np.median(seed, axis=0)
        mad = np.median(np.abs(seed - median), axis=0)
        mad = np.maximum(mad, 2.0)
        # Higher sensitivity makes the ground model stricter, so more regions
        # become candidate obstacles.
        threshold = 6.0 - 0.04 * sensitivity
        distance = np.sqrt(np.sum(((lab - median) / mad) ** 2, axis=2))
        road_color = (distance < threshold).astype(np.uint8)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        high = 260 - 2 * sensitivity
        edges = cv2.Canny(blurred, high // 2, high)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        # Keep colour-based ground, but remove strong texture from the seed model.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        road_mask = cv2.morphologyEx(road_color, cv2.MORPH_CLOSE, kernel)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(road_mask, connectivity=8)
        keep = np.zeros(num, bool)
        bottom_labels = np.unique(labels[max(0, rh - 8):, seed_x0:seed_x1])
        for label in bottom_labels:
            if label != 0:
                keep[label] = True
        if not keep.any() and num > 1:
            keep[1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))] = True
        road_mask = keep[labels].astype(np.uint8) * 255
        road_mask[edges > 0] = 0
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)

        if self._smooth_road is None or self._smooth_road.shape != road_mask.shape:
            self._smooth_road = road_mask.astype(np.float32) / 255.0
        else:
            self._smooth_road = (s.smooth_alpha * road_mask.astype(np.float32) / 255.0
                                 + (1.0 - s.smooth_alpha) * self._smooth_road)
        road = (self._smooth_road > 0.45).astype(np.uint8) * 255

        green = np.zeros_like(roi)
        green[:, :] = (0, 180, 0)
        mask = road > 0
        roi[mask] = cv2.addWeighted(roi, 0.55, green, 0.45, 0)[mask]
        contours, _ = cv2.findContours(road, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(roi, contours, -1, (0, 255, 180), 2)

        obstacle_mask = cv2.bitwise_not(road)
        obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_OPEN,
                                         cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        obstacle_mask = cv2.dilate(obstacle_mask, np.ones((5, 5), np.uint8), iterations=1)
        contours, _ = cv2.findContours(obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = max(24.0, s.min_obstacle_area_fraction * rw * rh)
        max_area = s.max_obstacle_area_fraction * rw * rh
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            if bw < 6 or bh < 6:
                continue
            detections.append(((x, y + y0, bw, bh), (x + bw / 2.0, y + y0 + bh / 2.0)))
        tracks = self._update_tracks(detections, w, h)

        center_blocked = False
        for track in tracks:
            x, y, bw, bh = track.bbox
            cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
            side = 'L' if track.centroid[0] < w * 0.38 else ('R' if track.centroid[0] > w * 0.62 else 'C')
            label = f'#{track.track_id} {side}'
            cv2.putText(out, label, (x, max(14, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            if x < w * 0.65 and x + bw > w * 0.35 and y + bh > h * 0.55:
                center_blocked = True

        road_fraction = float(np.count_nonzero(road) / (rw * rh))
        status = 'CENTER OBSTACLE' if center_blocked else ('OBSTACLES NEAR' if tracks else 'PATH CLEAR (VISUAL)')
        cv2.putText(out, status, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 255) if center_blocked else (0, 220, 220), 2, cv2.LINE_AA)
        cv2.putText(out, 'VISUAL AID - NOT A SAFETY STOP', (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return out, RobotInfo(road_fraction, len(tracks), center_blocked, list(tracks))

    def _update_tracks(self, detections, width, height):
        assigned = set()
        updated = []
        max_distance = max(24.0, 0.12 * max(width, height))
        for track in self._tracks:
            best, best_score = None, 0.0
            for index, (bbox, centroid) in enumerate(detections):
                if index in assigned:
                    continue
                distance = np.hypot(centroid[0] - track.centroid[0], centroid[1] - track.centroid[1])
                score = _bbox_iou(track.bbox, bbox) + max(0.0, 1.0 - distance / max_distance)
                if score > best_score:
                    best, best_score = index, score
            if best is not None and best_score > 0.35:
                bbox, centroid = detections[best]
                assigned.add(best)
                track.bbox, track.centroid, track.missed, track.hits = bbox, centroid, 0, track.hits + 1
                updated.append(track)
            else:
                track.missed += 1
                if track.missed <= self.settings.max_missed_frames:
                    updated.append(track)
        for index, (bbox, centroid) in enumerate(detections):
            if index not in assigned:
                track = ObstacleTrack(self._next_id, bbox, centroid)
                self._next_id += 1
                updated.append(track)
        self._tracks = updated
        return [track for track in self._tracks if track.missed == 0 or track.hits >= 2]
