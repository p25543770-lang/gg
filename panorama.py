"""Live two-camera composition and a fixed, calibrated right-to-left homography.

This is image stitching, not depth estimation or hardware synchronization.
A single homography cannot remove parallax between separated cameras: nearby
objects can double or bend at the seam. The unwarped side-by-side view remains
available. No video recording is performed here.
"""
from dataclasses import dataclass
import os
import tempfile

import cv2
import numpy as np


class PanoramaError(ValueError):
    """An actionable panorama configuration or calibration error."""


def _image_size(frame):
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise PanoramaError('Ожидается цветное изображение BGR с тремя каналами.')
    height, width = frame.shape[:2]
    if width < 64 or height < 64:
        raise PanoramaError('Для настройки панорамы нужен кадр не меньше 64×64.')
    return width, height


def _valid_size(size):
    return (isinstance(size, (tuple, list)) and len(size) == 2
            and all(isinstance(v, (int, np.integer)) and 1 <= v <= 16384 for v in size))


@dataclass(frozen=True)
class PanoramaSettings:
    view: str = 'side_by_side'
    max_width: int = 1600
    calibration_width: int = 800
    sync_tolerance_ms: float = 66.0
    swap_cameras: bool = False

    def __post_init__(self):
        if self.view not in ('side_by_side', 'panorama'):
            raise PanoramaError('Выберите режим «Рядом» или «Панорама».')
        if not isinstance(self.max_width, int) or not 640 <= self.max_width <= 3840:
            raise PanoramaError('Ширина общего кадра должна быть от 640 до 3840 пикселей.')
        if not isinstance(self.calibration_width, int) or not 320 <= self.calibration_width <= 1600:
            raise PanoramaError('Ширина настройки должна быть от 320 до 1600 пикселей.')
        if not np.isfinite(self.sync_tolerance_ms) or not 5 <= self.sync_tolerance_ms <= 250:
            raise PanoramaError('Допуск синхронизации должен быть от 5 до 250 мс.')
        if not isinstance(self.swap_cameras, bool):
            raise PanoramaError('Неверный параметр порядка камер.')


@dataclass(frozen=True)
class PanoramaCalibration:
    """Maps original right-image pixels into original left-image coordinates."""
    homography: np.ndarray
    left_size: tuple
    right_size: tuple
    inliers: int = 0
    matches: int = 0
    rms_px: float | None = None

    def __post_init__(self):
        h = np.asarray(self.homography, dtype=np.float64)
        if h.shape != (3, 3) or not np.all(np.isfinite(h)) or abs(h[2, 2]) < 1e-12:
            raise PanoramaError('Некорректная матрица совмещения H.')
        if not _valid_size(self.left_size) or not _valid_size(self.right_size):
            raise PanoramaError('Некорректный размер кадров в настройке панорамы.')
        if self.inliers < 0 or self.matches < 0 or self.inliers > self.matches:
            raise PanoramaError('Некорректная статистика совпадений.')
        if self.rms_px is not None and (not np.isfinite(self.rms_px) or self.rms_px < 0):
            raise PanoramaError('Некорректная ошибка совмещения.')
        object.__setattr__(self, 'homography', h / h[2, 2])
        object.__setattr__(self, 'left_size', tuple(map(int, self.left_size)))
        object.__setattr__(self, 'right_size', tuple(map(int, self.right_size)))


def _project(points, matrix):
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    q = np.column_stack((p, np.ones(len(p)))) @ np.asarray(matrix).T
    if np.any(np.abs(q[:, 2]) < 1e-10):
        raise PanoramaError('Совмещение вырождено. Повторите настройку на неподвижной текстурной сцене.')
    return q[:, :2] / q[:, 2:3]


def _corners(size):
    width, height = size
    return np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float64)


def _validate_homography(matrix, left_size, right_size):
    projected = _project(_corners(right_size), matrix)
    if not np.all(np.isfinite(projected)) or np.max(np.abs(projected)) > 50000:
        raise PanoramaError('Совмещение даёт слишком большое искажение. Оставьте камеры неподвижными и повторите.')
    lw, lh = left_size
    rw, rh = right_size
    minimum, maximum = projected.min(axis=0), projected.max(axis=0)
    span = maximum - minimum
    if np.any(span < (0.2 * rw, 0.2 * rh)) or np.any(span > (3.5 * rw, 3.5 * rh)):
        raise PanoramaError('Неправдоподобный масштаб панорамы. Нужны разные, хорошо различимые детали в общей зоне.')
    overlap_w = min(lw, maximum[0]) - max(0, minimum[0])
    overlap_h = min(lh, maximum[1]) - max(0, minimum[1])
    if overlap_w < 0.04 * min(lw, rw) or overlap_h < 0.15 * min(lh, rh):
        raise PanoramaError('Камеры почти не перекрываются. Уменьшите разворот камер или выберите режим «Рядом». '
                            'Если камеры перепутаны местами, включите «Поменять камеры местами».')
    if float(np.mean(projected[:, 0])) < lw * 0.52:
        raise PanoramaError('Порядок камер не подходит для панорамы. Включите «Поменять камеры местами» и повторите.')
    # Reject folds, extreme local scales and a nearly singular perspective warp.
    for y in np.linspace(0, rh - 1, 5):
        for x in np.linspace(0, rw - 1, 5):
            denominator = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
            if denominator <= 0.05:
                raise PanoramaError('Совмещение переворачивает кадр. Повторите настройку без движения камер.')
            px, py = _project([[x, y]], matrix)[0]
            j0 = (matrix[0, 0] * denominator - px * matrix[2, 0]) / denominator ** 2
            j1 = (matrix[0, 1] * denominator - px * matrix[2, 1]) / denominator ** 2
            j2 = (matrix[1, 0] * denominator - py * matrix[2, 0]) / denominator ** 2
            j3 = (matrix[1, 1] * denominator - py * matrix[2, 1]) / denominator ** 2
            determinant = j0 * j3 - j1 * j2
            a = np.array([j0, j2])
            b = np.array([j1, j3])
            lengths = np.linalg.norm(a) * np.linalg.norm(b)
            if not 0.08 <= determinant <= 12 or lengths < 1e-10:
                raise PanoramaError('Слишком сильное искажение панорамы. Проверьте перекрытие и неподвижность камер.')
            if abs(float(a @ b)) / lengths > 0.90:
                raise PanoramaError('Совмещение схлопывает изображение. Выберите сцену с деталями на разных высотах.')
    area = lw * lh + rw * rh
    canvas_w = max(lw, maximum[0]) - min(0, minimum[0])
    canvas_h = max(lh, maximum[1]) - min(0, minimum[1])
    if canvas_w > 2.5 * (lw + rw) or canvas_h > 2.5 * max(lh, rh) or canvas_w * canvas_h > 3 * area:
        raise PanoramaError('Панорама получается неоправданно большой. Повторите настройку склейки.')


def calibrate_panorama(left_frame, right_frame, settings=None):
    """Estimate and validate a fixed H from raw, simultaneous camera images."""
    settings = settings or PanoramaSettings()
    if settings.swap_cameras:
        left_frame, right_frame = right_frame, left_frame
    left_size, right_size = _image_size(left_frame), _image_size(right_frame)
    scales = []
    gray_images, masks, working_sizes = [], [], []
    for index, (frame, size) in enumerate(((left_frame, left_size), (right_frame, right_size))):
        width, height = size
        scale = min(1.0, settings.calibration_width / width)
        working = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))),
                             interpolation=cv2.INTER_AREA) if scale < 1 else frame
        scales.append(np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], np.float64))
        working_sizes.append((working.shape[1], working.shape[0]))
        gray_images.append(cv2.cvtColor(working, cv2.COLOR_BGR2GRAY))
        mask = np.zeros(working.shape[:2], np.uint8)
        wh, ww = mask.shape[1], mask.shape[0]
        # The left camera supplies the left part of the panorama. Broad masks
        # still support large overlap while suppressing implausible matches.
        x0, x1 = (int(ww * 0.15), ww) if index == 0 else (0, int(ww * 0.85))
        mask[:, x0:x1] = 255
        masks.append(mask)
    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.025, edgeThreshold=12)
    left_points, left_descriptors = sift.detectAndCompute(gray_images[0], masks[0])
    right_points, right_descriptors = sift.detectAndCompute(gray_images[1], masks[1])
    if left_descriptors is None or right_descriptors is None or min(len(left_points), len(right_points)) < 20:
        raise PanoramaError('Мало деталей в общей зоне. Добавьте свет и временно положите текстурные предметы '
                            'в перекрытие камер. Однородный пол не подходит для настройки.')
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(right_descriptors, left_descriptors, k=2)
    backward = matcher.knnMatch(left_descriptors, right_descriptors, k=2)
    reverse = {pairs[0].queryIdx: pairs[0].trainIdx for pairs in backward if len(pairs) >= 1}
    good = [pairs[0] for pairs in forward if len(pairs) == 2
            and pairs[0].distance < 0.75 * pairs[1].distance
            and reverse.get(pairs[0].trainIdx) == pairs[0].queryIdx]
    if len(good) < 20:
        raise PanoramaError(f'Надёжных совпадений мало: {len(good)}, нужно хотя бы 20. '
                            'Проверьте общее поле зрения, фокус и освещение; не двигайте камеры во время настройки.')
    source = np.float32([right_points[m.queryIdx].pt for m in good])
    target = np.float32([left_points[m.trainIdx].pt for m in good])
    matrix, inlier_mask = cv2.findHomography(source, target, cv2.RANSAC, 3.0,
                                           maxIters=5000, confidence=0.995)
    if matrix is None or inlier_mask is None:
        raise PanoramaError('Не удалось совместить камеры. Покажите текстурную неподвижную сцену в обоих кадрах.')
    keep = inlier_mask.ravel().astype(bool)
    inliers = int(np.count_nonzero(keep))
    if inliers < max(20, int(0.20 * len(good))):
        raise PanoramaError(f'Совпадения противоречивы: {inliers} из {len(good)}. '
                            'Уберите движущиеся предметы и повторите настройку.')
    source, target = source[keep], target[keep]
    residual = np.linalg.norm(_project(source, matrix) - target, axis=1)
    rms = float(np.sqrt(np.mean(residual ** 2)))
    if rms > 2.5:
        raise PanoramaError(f'Ошибка совмещения слишком велика: {rms:.2f} px. Улучшите освещение и повторите.')
    # Coverage is measured in the working image, where the matched points live.
    ww, wh = working_sizes[0]
    cells = {(min(2, int(x / ww * 3)), min(2, int(y / wh * 3))) for x, y in target}
    if len(cells) < 3 or np.ptp(target[:, 0]) < ww * 0.08:
        raise PanoramaError('Совпадения сосредоточены в одном маленьком пятне. Нужны детали в разных частях общей зоны.')
    original_h = np.linalg.inv(scales[0]) @ matrix @ scales[1]
    original_h /= original_h[2, 2]
    _validate_homography(original_h, left_size, right_size)
    return PanoramaCalibration(original_h, left_size, right_size, inliers, len(good), rms)


class PanoramaGeometry:
    """Precomputed output canvas and feathering weights for one calibration/size."""

    def __init__(self, calibration, max_width=1600):
        _validate_homography(calibration.homography, calibration.left_size, calibration.right_size)
        left_corners = _corners(calibration.left_size)
        right_corners = _project(_corners(calibration.right_size), calibration.homography)
        corners = np.vstack((left_corners, right_corners))
        # Sub-pixel calibration noise should not add a whole empty border pixel.
        minimum = np.floor(corners.min(axis=0) + 0.001).astype(int)
        maximum = np.ceil(corners.max(axis=0) + 0.999).astype(int)
        native_size = maximum - minimum
        max_height = max(360, int(max_width * 0.75))
        scale = min(1.0, max_width / native_size[0], max_height / native_size[1])
        self.image_size = tuple(max(1, int(round(v * scale))) for v in native_size)
        transform = np.array([[scale, 0, -minimum[0] * scale],
                              [0, scale, -minimum[1] * scale], [0, 0, 1]], np.float64)
        self.left_matrix = transform
        self.right_matrix = transform @ calibration.homography
        self.left_size, self.right_size = calibration.left_size, calibration.right_size
        white = (np.full((h, w), 255, np.uint8) for w, h in (self.left_size, self.right_size))
        self.left_mask = cv2.warpPerspective(next(white), self.left_matrix, self.image_size,
                                            flags=cv2.INTER_NEAREST) > 0
        self.right_mask = cv2.warpPerspective(next(white), self.right_matrix, self.image_size,
                                             flags=cv2.INTER_NEAREST) > 0
        self.right_only = self.right_mask & ~self.left_mask
        overlap = self.left_mask & self.right_mask
        self.overlap_box = None
        if np.any(overlap):
            ys, xs = np.where(overlap)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            self.overlap_box = (int(y0), int(y1), int(x0), int(x1))
            # Pad the overlap crop with zeros: distanceTransform needs a zero
            # border, otherwise an all-foreground crop can return FLT_MAX.
            def feather(mask):
                crop = mask[y0:y1, x0:x1].astype(np.uint8)
                padded = cv2.copyMakeBorder(crop, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
                return cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1].astype(np.float32)
            dl, dr = feather(self.left_mask), feather(self.right_mask)
            self.alpha = (dl / np.maximum(dl + dr, 1e-6)).astype(np.float32)[:, :, None]
            self.beta = 1.0 - self.alpha

    def render(self, left, right):
        if _image_size(left) != self.left_size or _image_size(right) != self.right_size:
            raise PanoramaError('Разрешение камер изменилось. Повторите настройку панорамы или загрузите настройку '
                                'для текущего разрешения. Режим «Рядом» работает без настройки.')
        warped_left = cv2.warpPerspective(left, self.left_matrix, self.image_size, flags=cv2.INTER_LINEAR)
        warped_right = cv2.warpPerspective(right, self.right_matrix, self.image_size, flags=cv2.INTER_LINEAR)
        output = warped_left.copy()
        output[self.right_only] = warped_right[self.right_only]
        if self.overlap_box is not None:
            y0, y1, x0, x1 = self.overlap_box
            a, b = warped_left[y0:y1, x0:x1], warped_right[y0:y1, x0:x1]
            blended = np.rint(a.astype(np.float32) * self.alpha + b.astype(np.float32) * self.beta).astype(np.uint8)
            region = output[y0:y1, x0:x1]
            both = self.left_mask[y0:y1, x0:x1] & self.right_mask[y0:y1, x0:x1]
            region[both] = blended[both]
        return output


def compose_side_by_side(left, right, max_width=1600):
    """Preserve each camera's aspect ratio; equalize height without stretching."""
    lw, lh = _image_size(left)
    rw, rh = _image_size(right)
    separator = 2
    available = max(2, max_width - separator)
    height = min(lh, rh, max(360, int(max_width * 0.75)))
    height = max(1, min(height, int(available / (lw / lh + rw / rh))))
    sizes = [(max(1, int(lw * height / lh)), height), (max(1, int(rw * height / rh)), height)]
    # Integer rounding must never exceed the requested output width.
    while sizes[0][0] + sizes[1][0] > available:
        if sizes[0][0] >= sizes[1][0] and sizes[0][0] > 1:
            sizes[0] = (sizes[0][0] - 1, height)
        elif sizes[1][0] > 1:
            sizes[1] = (sizes[1][0] - 1, height)
        else:
            break
    canvas = np.zeros((height, sizes[0][0] + separator + sizes[1][0], 3), np.uint8)
    canvas[:, sizes[0][0]:sizes[0][0] + separator] = (55, 55, 55)
    for frame, (width, _), x0 in ((left, sizes[0], 0), (right, sizes[1], sizes[0][0] + separator)):
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        canvas[:, x0:x0 + width] = resized
    return canvas


def save_panorama(calibration, path):
    path = os.fspath(path)
    extension = os.path.splitext(path)[1].lower()
    if extension not in ('.yml', '.yaml', '.json'):
        raise PanoramaError('Сохраните настройку панорамы в YAML (.yml) или JSON (.json).')
    directory = os.path.dirname(os.path.abspath(path))
    descriptor, temporary = tempfile.mkstemp(prefix='.panorama-', suffix=extension, dir=directory)
    os.close(descriptor)
    storage = cv2.FileStorage(temporary, cv2.FILE_STORAGE_WRITE)
    try:
        if not storage.isOpened():
            raise PanoramaError('Не удалось создать файл настройки панорамы.')
        storage.write('format_version', 1)
        storage.write('kind', 'purple-corners-panorama')
        storage.write('homography', calibration.homography)
        storage.write('left_width', calibration.left_size[0])
        storage.write('left_height', calibration.left_size[1])
        storage.write('right_width', calibration.right_size[0])
        storage.write('right_height', calibration.right_size[1])
        storage.write('inliers', int(calibration.inliers))
        storage.write('matches', int(calibration.matches))
        if calibration.rms_px is not None:
            storage.write('rms_px', float(calibration.rms_px))
        storage.release()
        os.replace(temporary, path)
    finally:
        if storage.isOpened():
            storage.release()
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_panorama(path):
    extension = os.path.splitext(os.fspath(path))[1].lower()
    if extension not in ('.yml', '.yaml', '.json'):
        raise PanoramaError('Выберите настройку панорамы YAML или JSON, а не калибровку глубины stereo.')
    storage = cv2.FileStorage(os.fspath(path), cv2.FILE_STORAGE_READ)
    try:
        if not storage.isOpened():
            raise PanoramaError('Не удалось открыть файл настройки панорамы.')
        version = storage.getNode('format_version')
        if version.isNone() or not version.isInt() or version.real() != 1:
            raise PanoramaError('Неизвестная версия настройки панорамы.')
        matrix = storage.getNode('homography')
        if matrix.isNone() or matrix.mat() is None:
            raise PanoramaError('В файле нет матрицы H. Это не настройка панорамы этой программы.')
        def integer(name):
            node = storage.getNode(name)
            if node.isNone() or not node.isInt():
                raise PanoramaError(f'В настройке панорамы нет целого поля {name}.')
            return int(node.real())
        def optional_integer(name):
            node = storage.getNode(name)
            return int(node.real()) if not node.isNone() and node.isInt() else 0
        rms = storage.getNode('rms_px')
        return PanoramaCalibration(matrix.mat(), (integer('left_width'), integer('left_height')),
                                   (integer('right_width'), integer('right_height')),
                                   optional_integer('inliers'), optional_integer('matches'),
                                   float(rms.real()) if not rms.isNone() and rms.isReal() else None)
    finally:
        storage.release()
