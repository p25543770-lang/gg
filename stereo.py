"""Stereo calibration, rectification and disparity/depth without any Tk calls.

Calibration translations are normalized to metres when their unit is known.
An unknown unit still permits disparity, but never produces a distance in metres.
"""
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np


class StereoError(ValueError):
    """A user-actionable calibration or stereo-processing error."""


_UNIT_FACTORS = {'m': 1.0, 'cm': 0.01, 'mm': 0.001}


def _matrix(value, name):
    try:
        result = np.asarray(value, dtype=np.float64).copy()
    except (TypeError, ValueError) as exc:
        raise StereoError(f'{name}: нужна числовая матрица.') from exc
    if not np.all(np.isfinite(result)):
        raise StereoError(f'{name}: значения должны быть конечными числами.')
    return result


@dataclass(frozen=True)
class StereoCalibration:
    K_left: np.ndarray
    D_left: np.ndarray
    K_right: np.ndarray
    D_right: np.ndarray
    R: np.ndarray
    T: np.ndarray
    image_size: tuple
    translation_unit: str = 'm'
    rms: float = None
    square_size_m: float = None

    def __post_init__(self):
        for name in ('K_left', 'D_left', 'K_right', 'D_right', 'R', 'T'):
            object.__setattr__(self, name, _matrix(getattr(self, name), name))
        for name in ('K_left', 'K_right', 'R'):
            if getattr(self, name).shape != (3, 3):
                raise StereoError(f'{name}: ожидался размер 3×3.')
        for name in ('K_left', 'K_right'):
            matrix = getattr(self, name)
            if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or abs(matrix[2, 2] - 1) > 1e-6:
                raise StereoError(f'{name}: неверные фокусное расстояние или последняя строка.')
        for name in ('D_left', 'D_right'):
            coeffs = getattr(self, name)
            if coeffs.ndim not in (1, 2) or coeffs.size not in (4, 5, 8, 12, 14):
                raise StereoError(f'{name}: ожидалось 4, 5, 8, 12 или 14 коэффициентов.')
            object.__setattr__(self, name, coeffs.reshape(1, -1))
        if self.T.shape not in ((3,), (3, 1), (1, 3)):
            raise StereoError('T: нужен вектор из трёх координат.')
        object.__setattr__(self, 'T', self.T.reshape(3, 1))
        if not np.allclose(self.R.T @ self.R, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(self.R), 1, atol=1e-5):
            raise StereoError('R должна быть матрицей поворота.')
        if self.translation_unit not in (*_UNIT_FACTORS, 'unknown'):
            raise StereoError('Неизвестная единица translation_unit; допустимы m, cm, mm, unknown.')
        if self.translation_unit != 'unknown':
            object.__setattr__(self, 'T', self.T * _UNIT_FACTORS[self.translation_unit])
            object.__setattr__(self, 'translation_unit', 'm')
        try:
            width, height = map(int, self.image_size)
        except (TypeError, ValueError) as exc:
            raise StereoError('image_size: нужны целые ширина и высота.') from exc
        if width < 32 or height < 32 or (width, height) != tuple(self.image_size):
            raise StereoError('Неверный размер кадров калибровки.')
        object.__setattr__(self, 'image_size', (width, height))
        if np.linalg.norm(self.T) <= 1e-9:
            raise StereoError('Нулевая базовая линия: камеры должны находиться в разных точках.')
        if self.translation_unit == 'm' and not 0.001 <= self.baseline_m <= 10:
            raise StereoError('Базовая линия вне диапазона 1 мм – 10 м. Проверьте размер клетки и единицы T.')
        for name in ('rms', 'square_size_m'):
            value = getattr(self, name)
            if value is not None:
                value = float(value)
                if not np.isfinite(value) or value < 0:
                    raise StereoError(f'{name}: нужно неотрицательное конечное число.')
                object.__setattr__(self, name, value)

    @property
    def baseline_m(self):
        return float(np.linalg.norm(self.T)) if self.translation_unit == 'm' else None


def load_calibration(path, override_unit='auto'):
    """Read an OpenCV YAML/JSON file; never deserialize Python objects."""
    if override_unit not in ('auto', *_UNIT_FACTORS, 'unknown'):
        raise StereoError('Неверная единица базовой линии.')
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    try:
        if not storage.isOpened():
            raise StereoError('Не удалось открыть файл калибровки YAML/JSON.')

        def matrix(*names):
            for name in names:
                node = storage.getNode(name)
                if not node.empty():
                    value = node.mat()
                    if value is not None:
                        return value
            raise StereoError(f'В файле нет матрицы {names[0]}.')

        def number(*names):
            for name in names:
                node = storage.getNode(name)
                if not node.empty() and (node.isInt() or node.isReal()):
                    return float(node.real())
            return None

        width, height = number('image_width', 'width'), number('image_height', 'height')
        if width is None or height is None:
            size = matrix('image_size')
            if size.size != 2:
                raise StereoError('В файле нужны image_width и image_height.')
            width, height = size.ravel()[:2]
        unit_node = storage.getNode('translation_unit')
        file_unit = unit_node.string() if not unit_node.empty() and unit_node.isString() else 'unknown'
        unit = file_unit if override_unit == 'auto' else override_unit
        return StereoCalibration(
            K_left=matrix('K1', 'K_left', 'M1'), D_left=matrix('D1', 'D_left'),
            K_right=matrix('K2', 'K_right', 'M2'), D_right=matrix('D2', 'D_right'),
            R=matrix('R'), T=matrix('T'), image_size=(int(width), int(height)),
            translation_unit=unit, rms=number('stereo_rms', 'rms'),
            square_size_m=number('square_size_m'),
        )
    finally:
        storage.release()


def save_calibration(calibration, path):
    """Atomically save a portable OpenCV calibration, with explicit T units."""
    path = Path(path)
    if path.suffix.lower() not in ('.yml', '.yaml', '.json'):
        raise StereoError('Сохраните калибровку в файл .yml, .yaml или .json.')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False) as handle:
            temporary = handle.name
        storage = cv2.FileStorage(temporary, cv2.FILE_STORAGE_WRITE)
        try:
            if not storage.isOpened():
                raise StereoError('Не удалось создать файл калибровки.')
            storage.write('format_version', 1)
            storage.write('image_width', calibration.image_size[0])
            storage.write('image_height', calibration.image_size[1])
            storage.write('translation_unit', calibration.translation_unit)
            for name in ('K_left', 'D_left', 'K_right', 'D_right', 'R', 'T'):
                key = {'K_left': 'K1', 'K_right': 'K2', 'D_left': 'D1', 'D_right': 'D2'}.get(name, name)
                storage.write(key, getattr(calibration, name))
            for name in ('rms', 'square_size_m'):
                if getattr(calibration, name) is not None:
                    storage.write(name, getattr(calibration, name))
        finally:
            storage.release()
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def validate_board(board_size, square_size_m):
    try:
        columns, rows = map(int, board_size)
        square = float(square_size_m)
    except (TypeError, ValueError) as exc:
        raise StereoError('Укажите целые числа внутренних углов и размер клетки в мм.') from exc
    if not 2 <= columns <= 30 or not 2 <= rows <= 30 or columns == rows:
        raise StereoError('Нужно от 2 до 30 углов по каждой стороне; стороны должны различаться, например 9×6.')
    if not np.isfinite(square) or not 0.001 <= square <= 1:
        raise StereoError('Сторона клетки должна быть от 1 до 1000 мм.')
    return (columns, rows), square


def detect_chessboard(frame, board_size):
    """Return ordered subpixel internal corners, or None if the board is absent."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    scale = min(1.0, 960 / gray.shape[1])
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(small, board_size, flags=flags)
    if not found:
        return None
    corners = corners.astype(np.float32) / scale
    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.001))
    return corners


def calibrate_from_corners(left_points, right_points, board_size, square_size_m, image_size):
    """Calibrate a fixed stereo rig from at least ten varied, matched board views."""
    board_size, square = validate_board(board_size, square_size_m)
    if len(left_points) != len(right_points) or len(left_points) < 10:
        raise StereoError('Нужно не менее 10 пар снимков доски; рекомендуется 15–30.')
    if len(left_points) > 60:
        raise StereoError('Слишком много пар снимков (максимум 60).')
    grid = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    grid[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    grid *= square
    objects = [grid.copy() for _ in left_points]
    left, right = [], []
    for a, b in zip(left_points, right_points):
        a, b = _matrix(a, 'Углы левой камеры'), _matrix(b, 'Углы правой камеры')
        if a.shape != (grid.shape[0], 1, 2) or b.shape != a.shape:
            raise StereoError('Число углов не соответствует выбранной доске или снимки не совпадают.')
        left.append(a.astype(np.float32))
        right.append(b.astype(np.float32))
    try:
        _, k1, d1, _, _ = cv2.calibrateCamera(objects, left, image_size, None, None)
        _, k2, d2, _, _ = cv2.calibrateCamera(objects, right, image_size, None, None)
        calibrated = cv2.stereoCalibrate(
            objects, left, right, k1, d1, k2, d2, image_size,
            flags=cv2.CALIB_FIX_INTRINSIC,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-6),
        )
        # OpenCV 4.x returns E/F (and sometimes per-view errors) after T.
        rms, k1, d1, k2, d2, rotation, translation = calibrated[:7]
    except cv2.error as exc:
        raise StereoError('Калибровка не удалась. Снимите доску заново с другими наклонами и положениями.') from exc
    if not np.isfinite(rms) or rms > 5:
        raise StereoError('Слишком большая ошибка калибровки. Проверьте порядок камер, доску и неподвижность креплений.')
    return StereoCalibration(k1, d1, k2, d2, rotation, translation, image_size,
                             translation_unit='m', rms=float(rms), square_size_m=square)


@dataclass(frozen=True)
class StereoSettings:
    max_width: int = 640
    num_disparities: int = 128
    sync_tolerance_ms: float = 33.0
    min_depth_m: float = 0.2
    max_depth_m: float = 10.0
    view: str = 'depth'

    def __post_init__(self):
        if self.max_width < 160 or self.max_width > 1920:
            raise StereoError('Ширина обработки должна быть от 160 до 1920 пикселей.')
        if self.num_disparities < 16 or self.num_disparities > 256 or self.num_disparities % 16:
            raise StereoError('Диапазон диспаратности должен быть кратен 16, от 16 до 256.')
        values = (self.sync_tolerance_ms, self.min_depth_m, self.max_depth_m)
        if not all(np.isfinite(v) for v in values) or not 1 <= self.sync_tolerance_ms <= 500:
            raise StereoError('Допуск синхронизации должен быть от 1 до 500 мс.')
        if self.min_depth_m <= 0 or self.max_depth_m <= self.min_depth_m:
            raise StereoError('Нужно 0 < минимальная глубина < максимальная глубина.')
        if self.view not in ('depth', 'disparity'):
            raise StereoError('Доступны карта глубины и диспаратность.')


@dataclass
class StereoResult:
    disparity: np.ndarray
    depth_m: np.ndarray | None
    color_rgb: np.ndarray
    sync_skew_ms: float
    elapsed_ms: float
    valid_fraction: float
    baseline_m: float | None
    image_size: tuple
    view: str
    created_at: float


class StereoProcessor:
    """Reusable rectification maps and SGBM matcher for one calibration/settings pair."""
    def __init__(self, calibration, settings=None):
        self.calibration = calibration
        self.settings = settings or StereoSettings()
        self._prepared = None

    def _prepare(self, image_size):
        if image_size != self.calibration.image_size:
            raise StereoError(f'Калибровка для {self.calibration.image_size[0]}×{self.calibration.image_size[1]}, '
                              f'а камеры дают {image_size[0]}×{image_size[1]}. Выберите прежнее разрешение или откалибруйте заново.')
        if self._prepared is not None:
            return self._prepared
        c, s = self.calibration, self.settings
        width, height = image_size
        scale = min(1.0, s.max_width / width)
        size = (max(32, round(width * scale)), max(32, round(height * scale)))
        sx, sy = size[0] / width, size[1] / height
        scaling = np.diag([sx, sy, 1.0])
        k1, k2 = scaling @ c.K_left, scaling @ c.K_right
        r1, r2, p1, p2, q, roi1, roi2 = cv2.stereoRectify(
            k1, c.D_left, k2, c.D_right, size, c.R, c.T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
        )
        map1x, map1y = cv2.initUndistortRectifyMap(k1, c.D_left, r1, p1, size, cv2.CV_16SC2)
        map2x, map2y = cv2.initUndistortRectifyMap(k2, c.D_right, r2, p2, size, cv2.CV_16SC2)
        x, y = max(roi1[0], roi2[0]), max(roi1[1], roi2[1])
        right, bottom = min(roi1[0] + roi1[2], roi2[0] + roi2[2]), min(roi1[1] + roi1[3], roi2[1] + roi2[3])
        roi = np.zeros(size[::-1], bool)
        if right > x and bottom > y:
            roi[y:bottom, x:right] = True
        if not roi.any():
            raise StereoError('После выравнивания нет общей области изображения. Проверьте калибровку.')
        ones = np.full(size[::-1], 255, np.uint8)
        border = (cv2.remap(ones, map1x, map1y, cv2.INTER_NEAREST) > 250) & (
            cv2.remap(ones, map2x, map2y, cv2.INTER_NEAREST) > 250)
        roi &= border
        block = 5
        num = min(s.num_disparities, max(16, ((size[0] - 1) // 16) * 16))
        matcher = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=num, blockSize=block,
            P1=8 * block * block, P2=32 * block * block,
            disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100,
            speckleRange=2, preFilterCap=31, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        self._prepared = (size, map1x, map1y, map2x, map2y, matcher, q, roi, num)
        return self._prepared

    @staticmethod
    def _color(values, valid, near, far, inverse=False):
        image = np.zeros((*values.shape, 3), np.uint8)
        if not valid.any():
            return image
        if inverse:
            normalized = (1.0 / np.maximum(values[valid], 1e-6) - 1.0 / far) / (1.0 / near - 1.0 / far)
        else:
            normalized = (values[valid] - near) / max(far - near, 1e-6)
        gray = np.zeros(values.shape, np.uint8)
        gray[valid] = np.clip(normalized * 255, 0, 255).astype(np.uint8)
        image[valid] = cv2.applyColorMap(gray[valid].reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)[:, ::-1]
        return image

    def process(self, left, right, sync_skew_ms=0.0):
        started = time.perf_counter()
        if left is None or right is None or left.shape != right.shape or left.ndim not in (2, 3):
            raise StereoError('Для стерео нужны два кадра одинакового размера.')
        height, width = left.shape[:2]
        size, m1x, m1y, m2x, m2y, matcher, q, roi, num = self._prepare((width, height))
        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY) if left.ndim == 3 else left
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY) if right.ndim == 3 else right
        if (width, height) != size:
            left_gray = cv2.resize(left_gray, size, interpolation=cv2.INTER_AREA)
            right_gray = cv2.resize(right_gray, size, interpolation=cv2.INTER_AREA)
        a = cv2.remap(left_gray, m1x, m1y, cv2.INTER_LINEAR)
        b = cv2.remap(right_gray, m2x, m2y, cv2.INTER_LINEAR)
        disparity = matcher.compute(a, b).astype(np.float32) / 16.0
        valid = roi & np.isfinite(disparity) & (disparity > 0) & (disparity < num)
        disparity[~valid] = np.nan
        depth = None
        if self.calibration.baseline_m is not None:
            # Equivalent to the Z component of cv2.reprojectImageTo3 for the
            # disparity-to-depth Q returned by stereoRectify, without depending
            # on the OpenCV 4.x/5.x spelling of that helper.
            numerator = disparity * np.float32(q[2, 2]) + np.float32(q[2, 3])
            denominator = disparity * np.float32(q[3, 2]) + np.float32(q[3, 3])
            with np.errstate(divide='ignore', invalid='ignore'):
                z = numerator / denominator
            depth = np.full(disparity.shape, np.nan, np.float32)
            usable = valid & np.isfinite(z) & (z > 0) & (denominator != 0)
            depth[usable] = z[usable]
            valid = usable
        view = self.settings.view
        if view == 'depth' and depth is not None:
            color = self._color(depth, valid, self.settings.min_depth_m, self.settings.max_depth_m, inverse=True)
        else:
            view = 'disparity'
            color = self._color(disparity, valid, 1.0, float(num))
        return StereoResult(
            disparity=disparity, depth_m=depth, color_rgb=color, sync_skew_ms=sync_skew_ms,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            valid_fraction=float(np.count_nonzero(valid) / max(1, np.count_nonzero(roi))),
            baseline_m=self.calibration.baseline_m, image_size=size, view=view,
            created_at=time.perf_counter(),
        )
