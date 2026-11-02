"""Human skeleton overlay: NanoDet-Plus person detector + RTMPose-t keypoints.

Both networks are small ONNX files executed by cv2.dnn, so the packaged Windows
executable needs no extra runtime, no GPU and no network access. The detector
finds people, the pose network returns 17 COCO keypoints per person, and a light
IoU tracker keeps the identifiers and colours stable between frames.

This is a visual estimate, not a measurement: keypoints drift on motion blur,
occlusion, unusual poses, strong backlight and when a person is partly outside
the frame. Never use it as a safety sensor.
"""
import os
import sys
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

DETECTOR_FILE = 'nanodet_plus_m_320.onnx'
POSE_FILE = 'rtmpose_t_256x192.onnx'

# COCO 17-keypoint order used by RTMPose.
KEYPOINT_NAMES = (
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
)
# (from, to, BGR colour): left side green, right side orange, torso cyan.
SKELETON = (
    (0, 1, (0, 220, 0)), (0, 2, (0, 160, 255)),
    (1, 3, (0, 220, 0)), (2, 4, (0, 160, 255)),
    (5, 7, (0, 220, 0)), (7, 9, (0, 220, 0)),
    (6, 8, (0, 160, 255)), (8, 10, (0, 160, 255)),
    (5, 6, (255, 220, 0)), (5, 11, (255, 220, 0)),
    (6, 12, (255, 220, 0)), (11, 12, (255, 220, 0)),
    (11, 13, (0, 220, 0)), (13, 15, (0, 220, 0)),
    (12, 14, (0, 160, 255)), (14, 16, (0, 160, 255)),
)
TRACK_COLORS = (
    (255, 128, 0), (0, 200, 255), (255, 0, 200), (0, 255, 128),
    (128, 0, 255), (255, 255, 0), (0, 128, 255), (200, 200, 200),
)

_DETECTOR_SIZE = 320
_DETECTOR_STRIDES = (8, 16, 32, 64)
_REG_MAX = 7
_DETECTOR_MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
_DETECTOR_STD = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)
_POSE_W, _POSE_H = 192, 256
_POSE_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_POSE_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
_SIMCC_SPLIT = 2.0


class PoseModelError(RuntimeError):
    """Raised when the bundled ONNX files are missing or cannot be loaded."""


def resource_path(*parts):
    """Resolve a bundled data file both from source and from a PyInstaller exe."""
    base = getattr(sys, '_MEIPASS', None) or os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, *parts)


def models_present():
    return (os.path.isfile(resource_path('models', DETECTOR_FILE))
            and os.path.isfile(resource_path('models', POSE_FILE)))


@dataclass(frozen=True)
class PoseSettings:
    score_threshold: float = 0.40
    nms_threshold: float = 0.60
    keypoint_threshold: float = 0.30
    max_people: int = 4
    min_box_pixels: int = 24
    smooth_alpha: float = 0.55
    max_missed_frames: int = 5
    detect_interval: int = 2
    draw_boxes: bool = True

    def __post_init__(self):
        if not 0.05 <= self.score_threshold <= 0.95:
            raise ValueError('score_threshold must be between 0.05 and 0.95')
        if not 0.1 <= self.nms_threshold <= 0.9:
            raise ValueError('nms_threshold must be between 0.1 and 0.9')
        if not 0.0 <= self.keypoint_threshold <= 1.5:
            raise ValueError('keypoint_threshold must be between 0.0 and 1.5')
        if not 1 <= self.max_people <= 16:
            raise ValueError('max_people must be between 1 and 16')
        if self.min_box_pixels < 4:
            raise ValueError('min_box_pixels must be at least 4')
        if not 0 < self.smooth_alpha <= 1:
            raise ValueError('smooth_alpha must be in (0, 1]')
        if self.max_missed_frames < 0:
            raise ValueError('max_missed_frames must be non-negative')
        if not 1 <= self.detect_interval <= 30:
            raise ValueError('detect_interval must be between 1 and 30')


@dataclass
class PersonPose:
    track_id: int
    bbox: tuple
    score: float
    keypoints: np.ndarray
    missed: int = 0
    hits: int = 1

    @property
    def visible_keypoints(self):
        return int(np.count_nonzero(self.keypoints[:, 2] > 0))


@dataclass
class PoseInfo:
    people: list = field(default_factory=list)
    inference_ms: float = 0.0
    status: str = ''

    @property
    def person_count(self):
        return len(self.people)


def _bbox_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _detector_grids():
    global _GRIDS
    if _GRIDS is None:
        grids = []
        for stride in _DETECTOR_STRIDES:
            n = _DETECTOR_SIZE // stride
            xv, yv = np.meshgrid(np.arange(n) * stride, np.arange(n) * stride)
            grids.append(np.stack((xv.ravel(), yv.ravel()), axis=-1).astype(np.float32))
        _GRIDS = tuple(grids)
    return _GRIDS


_GRIDS = None


def _person_boxes(output, width, height, scale, score_threshold, nms_threshold):
    """Decode NanoDet-Plus distributions into person boxes in source pixels."""
    preds = output[0] if output.ndim == 3 else output
    project = np.arange(_REG_MAX + 1, dtype=np.float32)
    boxes, scores = [], []
    offset = 0
    for stride, grid in zip(_DETECTOR_STRIDES, _detector_grids()):
        count = len(grid)
        cls = preds[offset:offset + count, :80]
        dist = preds[offset:offset + count, 80:]
        offset += count
        if dist.shape[1] != 4 * (_REG_MAX + 1):
            raise PoseModelError('Неожиданный формат выхода детектора людей.')
        exp = np.exp(dist.reshape(-1, _REG_MAX + 1))
        softmax = exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-9)
        distance = (softmax @ project).reshape(-1, 4) * stride
        boxes.append(np.stack([grid[:, 0] - distance[:, 0], grid[:, 1] - distance[:, 1],
                               grid[:, 0] + distance[:, 2], grid[:, 1] + distance[:, 3]], axis=-1))
        scores.append(cls[:, 0])  # COCO class 0 is 'person'
    boxes = np.concatenate(boxes) / scale
    scores = np.concatenate(scores)
    keep = scores > score_threshold
    boxes, scores = boxes[keep], scores[keep]
    if len(boxes) == 0:
        return []
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height - 1)
    wh = boxes.copy()
    wh[:, 2] = boxes[:, 2] - boxes[:, 0]
    wh[:, 3] = boxes[:, 3] - boxes[:, 1]
    valid = (wh[:, 2] > 1) & (wh[:, 3] > 1)
    boxes, wh, scores = boxes[valid], wh[valid], scores[valid]
    if len(boxes) == 0:
        return []
    indices = cv2.dnn.NMSBoxes(wh.tolist(), scores.tolist(), score_threshold, nms_threshold)
    indices = np.array(indices).ravel()
    order = indices[np.argsort(-scores[indices])]
    return [(boxes[i], float(scores[i])) for i in order]


def _pose_affine(bbox):
    """MMPose top-down crop: expand the box, keep the 192x256 aspect ratio."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    aspect = _POSE_W / _POSE_H
    if bw > aspect * bh:
        bh = bw / aspect
    else:
        bw = bh * aspect
    longest = max(bw, bh) * 1.25
    crop_w, crop_h = longest * aspect, longest
    scale = _POSE_W / crop_w
    return np.array([[scale, 0, _POSE_W / 2.0 - cx * scale],
                     [0, scale, _POSE_H / 2.0 - cy * scale]], dtype=np.float32)


class SkeletonDetector:
    """Stateful per-camera pose estimator. Call process() from one thread only."""

    def __init__(self, settings=None, detector_path=None, pose_path=None):
        self.settings = settings or PoseSettings()
        self._detector_path = detector_path or resource_path('models', DETECTOR_FILE)
        self._pose_path = pose_path or resource_path('models', POSE_FILE)
        self._detector = None
        self._pose = None
        self._tracks = []
        self._next_id = 1
        self._frame_index = 0
        self._last_boxes = []

    def load(self):
        if self._detector is not None:
            return
        for path in (self._detector_path, self._pose_path):
            if not os.path.isfile(path):
                raise PoseModelError('Не найдена модель ' + os.path.basename(path)
                                     + '. Переустановите программу или скачайте релиз заново.')
        try:
            self._detector = cv2.dnn.readNetFromONNX(self._detector_path)
            self._pose = cv2.dnn.readNetFromONNX(self._pose_path)
        except cv2.error as exc:
            self._detector = self._pose = None
            raise PoseModelError('Не удалось загрузить модель скелета: ' + str(exc)[:200]) from exc

    def reset(self):
        self._tracks = []
        self._next_id = 1
        self._frame_index = 0
        self._last_boxes = []

    def process(self, frame, sensitivity=50):
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError('expected a BGR frame')
        self.load()
        started = time.perf_counter()
        settings = self.settings
        sensitivity = int(max(1, min(100, sensitivity)))
        # Higher sensitivity lowers both thresholds, so more people and more
        # joints survive, at the cost of extra false positives.
        score_threshold = float(np.clip(settings.score_threshold + 0.20 - 0.004 * sensitivity, 0.05, 0.95))
        keypoint_threshold = float(np.clip(settings.keypoint_threshold + 0.15 - 0.003 * sensitivity, 0.05, 1.5))

        out = frame.copy()
        height, width = out.shape[:2]
        people = []
        if height >= 16 and width >= 16:
            # Detection costs more than pose estimation, so between detections
            # the previous boxes are reused; a lost person is re-found within
            # detect_interval frames.
            if self._frame_index % settings.detect_interval == 0 or not self._last_boxes:
                self._last_boxes = self._detect_people(frame, width, height, score_threshold, settings)
            self._frame_index += 1
            for bbox, score in self._last_boxes[:settings.max_people]:
                keypoints = self._keypoints(frame, bbox)
                if keypoints is not None:
                    keypoints[keypoints[:, 2] < keypoint_threshold, 2] = 0.0
                    people.append((bbox, score, keypoints))
        tracks = self._update_tracks(people)
        self._draw(out, tracks)
        elapsed = (time.perf_counter() - started) * 1000.0
        status = f'PEOPLE: {len(tracks)}' if tracks else 'NO PEOPLE'
        cv2.putText(out, status, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 220, 255) if tracks else (200, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(out, 'POSE ESTIMATE - NOT A SAFETY SENSOR', (8, height - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return out, PoseInfo(list(tracks), elapsed, status)

    def _detect_people(self, frame, width, height, score_threshold, settings):
        scale = min(_DETECTOR_SIZE / width, _DETECTOR_SIZE / height)
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        canvas = np.zeros((_DETECTOR_SIZE, _DETECTOR_SIZE, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        normalized = (canvas.astype(np.float32) - _DETECTOR_MEAN) / _DETECTOR_STD
        self._detector.setInput(cv2.dnn.blobFromImage(normalized))
        output = self._detector.forward()
        detections = _person_boxes(output, width, height, scale, score_threshold, settings.nms_threshold)
        result = []
        for box, score in detections:
            x1, y1, x2, y2 = (float(v) for v in box)
            if x2 - x1 < settings.min_box_pixels or y2 - y1 < settings.min_box_pixels:
                continue
            result.append(((x1, y1, x2, y2), score))
        return result

    def _keypoints(self, frame, bbox):
        affine = _pose_affine(bbox)
        crop = cv2.warpAffine(frame, affine, (_POSE_W, _POSE_H), flags=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - _POSE_MEAN) / _POSE_STD
        self._pose.setInput(np.ascontiguousarray(rgb.transpose(2, 0, 1)[None]))
        try:
            simcc_x, simcc_y = self._pose.forward(['simcc_x', 'simcc_y'])
        except cv2.error as exc:
            raise PoseModelError('Модель скелета вернула ошибку: ' + str(exc)[:200]) from exc
        if simcc_x.shape[1] != 17 or simcc_y.shape[1] != 17:
            raise PoseModelError('Модель скелета должна возвращать 17 точек.')
        xs, ys = simcc_x[0], simcc_y[0]
        px, py = xs.argmax(axis=1) / _SIMCC_SPLIT, ys.argmax(axis=1) / _SIMCC_SPLIT
        confidence = np.minimum(xs.max(axis=1), ys.max(axis=1))
        inverse = cv2.invertAffineTransform(affine)
        points = np.stack([px, py, np.ones(len(px))], axis=-1) @ inverse.T
        return np.concatenate([points, confidence[:, None]], axis=1).astype(np.float32)

    def _update_tracks(self, people):
        assigned, updated = set(), []
        for track in self._tracks:
            best, best_iou = None, 0.0
            for index, (bbox, score, keypoints) in enumerate(people):
                if index in assigned:
                    continue
                overlap = _bbox_iou(track.bbox, bbox)
                if overlap > best_iou:
                    best, best_iou = index, overlap
            if best is not None and best_iou >= 0.25:
                bbox, score, keypoints = people[best]
                assigned.add(best)
                alpha = self.settings.smooth_alpha
                if track.keypoints.shape == keypoints.shape:
                    both = (track.keypoints[:, 2] > 0) & (keypoints[:, 2] > 0)
                    smoothed = keypoints.copy()
                    smoothed[both, :2] = (alpha * keypoints[both, :2]
                                          + (1.0 - alpha) * track.keypoints[both, :2])
                    keypoints = smoothed
                track.bbox, track.score, track.keypoints = bbox, score, keypoints
                track.missed, track.hits = 0, track.hits + 1
                updated.append(track)
            else:
                track.missed += 1
                if track.missed <= self.settings.max_missed_frames:
                    updated.append(track)
        for index, (bbox, score, keypoints) in enumerate(people):
            if index not in assigned:
                updated.append(PersonPose(self._next_id, bbox, score, keypoints))
                self._next_id += 1
        self._tracks = updated
        return [track for track in self._tracks if track.missed == 0]

    def _draw(self, frame, tracks):
        height, width = frame.shape[:2]
        thickness = 2 if min(width, height) >= 240 else 1
        radius = 3 if min(width, height) >= 240 else 2
        for track in tracks:
            color = TRACK_COLORS[(track.track_id - 1) % len(TRACK_COLORS)]
            points = track.keypoints
            for start, end, limb_color in SKELETON:
                if points[start, 2] <= 0 or points[end, 2] <= 0:
                    continue
                a = (int(round(points[start, 0])), int(round(points[start, 1])))
                b = (int(round(points[end, 0])), int(round(points[end, 1])))
                cv2.line(frame, a, b, limb_color, thickness, cv2.LINE_AA)
            for x, y, confidence in points:
                if confidence <= 0:
                    continue
                cv2.circle(frame, (int(round(x)), int(round(y))), radius, color, -1, cv2.LINE_AA)
            x1, y1, x2, y2 = (int(round(v)) for v in track.bbox)
            if self.settings.draw_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
            side = 'L' if (x1 + x2) / 2 < width * 0.38 else ('R' if (x1 + x2) / 2 > width * 0.62 else 'C')
            label = f'#{track.track_id} {side} {track.score:.2f}'
            cv2.putText(frame, label, (max(0, x1), max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)
