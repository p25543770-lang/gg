"""Background two-camera preview composition and panorama calibration."""
from dataclasses import dataclass
import queue
import threading
import time

import cv2
import numpy as np

from panorama import (
    PanoramaCalibration, PanoramaError, PanoramaGeometry, PanoramaSettings,
    calibrate_panorama, compose_side_by_side,
)


@dataclass(frozen=True)
class PanoramaResult:
    color_rgb: np.ndarray
    image_size: tuple
    view: str
    requested_view: str
    sync_skew_ms: float
    elapsed_ms: float
    created_at: float
    fallback: bool = False


@dataclass(frozen=True)
class PanoramaState:
    result: PanoramaResult | None
    status: str
    notice: str
    notice_id: int
    busy: bool
    calibration: PanoramaCalibration | None
    enabled: bool
    settings: PanoramaSettings


class PanoramaWorker:
    """One worker thread; the UI exchanges snapshots and commands only."""

    def __init__(self):
        self.lock = threading.Condition()
        self.commands = queue.Queue()
        self.stopping = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self._streams = (None, None)
        self._enabled = False
        self._settings = PanoramaSettings()
        self._calibration = None
        self._geometry = None
        self._result = None
        self._status = 'Откройте обе камеры на вкладке «Камеры».'
        self._notice = ''
        self._notice_id = 0
        self._busy = False
        self._revision = 0
        self._last_pair = None
        self._last_capture_pair = None

    def start(self):
        self.thread.start()

    def stop(self):
        with self.lock:
            self.stopping = True
            self.lock.notify_all()
        self.commands.put(None)

    def alive(self):
        return self.thread.is_alive()

    def _invalidate(self, clear_calibration=False):
        self._revision += 1
        self._geometry = None
        self._result = None
        self._last_pair = None
        self._last_capture_pair = None
        if clear_calibration:
            self._calibration = None

    def set_streams(self, left, right):
        with self.lock:
            streams = (left, right)
            if streams == self._streams:
                return
            self._streams = streams
            self._invalidate(clear_calibration=True)
            self._status = 'Откройте обе камеры на вкладке «Камеры».'
            self.lock.notify_all()

    def set_calibration(self, calibration):
        if calibration is not None and not isinstance(calibration, PanoramaCalibration):
            raise PanoramaError('Неверная настройка панорамы.')
        with self.lock:
            self._calibration = calibration
            self._invalidate()
            self._status = 'Настройка панорамы загружена.' if calibration else 'Панорама не настроена.'
            self.lock.notify_all()

    def configure(self, settings):
        if not isinstance(settings, PanoramaSettings):
            raise PanoramaError('Неверные параметры общего вида.')
        with self.lock:
            if settings == self._settings:
                return
            swap_changed = settings.swap_cameras != self._settings.swap_cameras
            width_changed = settings.max_width != self._settings.max_width
            self._settings = settings
            # Re-render the latest fresh pair after a view/size/sync change.
            self._result = None
            self._last_pair = None
            if swap_changed:
                self._invalidate(clear_calibration=True)
                self._notify('Порядок камер изменён. Настройте панораму заново.')
            elif width_changed:
                self._geometry = None
            self.lock.notify_all()

    def set_enabled(self, enabled):
        with self.lock:
            enabled = bool(enabled)
            if enabled == self._enabled:
                return
            self._enabled = enabled
            self._result = None
            self._last_pair = None
            self._status = 'Общий вид включён.' if enabled else 'Общий вид приостановлен.'
            self.lock.notify_all()

    def request_calibration(self):
        with self.lock:
            if self._busy:
                return False
            if any(stream is None for stream in self._streams):
                self._notify('Сначала откройте обе камеры.')
                return False
            self._busy = True
            self._revision += 1
            self.commands.put(('calibrate', self._revision, time.perf_counter()))
            self.lock.notify_all()
            return True

    def poll(self):
        with self.lock:
            return PanoramaState(
                result=self._result, status=self._status, notice=self._notice,
                notice_id=self._notice_id, busy=self._busy, calibration=self._calibration,
                enabled=self._enabled, settings=self._settings,
            )

    def _notify(self, message):
        self._notice = message
        self._notice_id += 1

    def _waiting(self, message):
        with self.lock:
            if self._status != message:
                self._status = message
                self._result = None

    def _fresh_pair(self, left_stream, right_stream, tolerance_ms, after=0.0, previous=None):
        if left_stream is None or right_stream is None:
            self._waiting('Откройте левую и правую камеры на вкладке «Камеры».')
            return None
        if not left_stream.alive() or not right_stream.alive():
            self._waiting('Одна из камер закрыта или завершилась. Откройте обе камеры заново.')
            return None
        left, right = left_stream.snapshot(), right_stream.snapshot()
        if left is None or right is None:
            self._waiting('Ожидание первых кадров от обеих камер…')
            return None
        left_frame, left_time, left_sequence = left
        right_frame, right_time, right_sequence = right
        skew = abs(left_time - right_time) * 1000.0
        if skew > tolerance_ms:
            self._waiting(f'Кадры рассинхронизированы: {skew:.1f} мс > {tolerance_ms:.0f} мс. '
                          'Увеличьте допуск только для почти неподвижной сцены.')
            return None
        age = time.perf_counter() - min(left_time, right_time)
        if age > max(0.5, tolerance_ms / 1000.0 * 4):
            self._waiting('Ожидание свежих кадров. Проверьте, что обе камеры продолжают съёмку.')
            return None
        if after and min(left_time, right_time) < after - tolerance_ms / 1000.0:
            self._waiting('Ожидание новой пары кадров для настройки панорамы…')
            return None
        pair_id = (left_sequence, right_sequence)
        if previous is not None and pair_id == previous:
            return None
        return left_frame, right_frame, skew, pair_id

    def _handle_command(self, command):
        kind, revision, requested_at = command
        try:
            with self.lock:
                left, right = self._streams
                settings = self._settings
            if kind != 'calibrate':
                return
            deadline = time.perf_counter() + 5.0
            pair = None
            while time.perf_counter() < deadline:
                with self.lock:
                    if self.stopping or revision != self._revision:
                        return
                    left, right = self._streams
                pair = self._fresh_pair(left, right, settings.sync_tolerance_ms,
                                        after=requested_at, previous=self._last_capture_pair)
                if pair is not None:
                    break
                with self.lock:
                    self.lock.wait(0.02)
            if pair is None:
                with self.lock:
                    self._notify('Не удалось получить свежую пару кадров. Проверьте камеры и допуск синхронизации.')
                return
            left_frame, right_frame, _, pair_id = pair
            calibration = calibrate_panorama(left_frame, right_frame, settings)
            with self.lock:
                if self.stopping or revision != self._revision:
                    return
                self._calibration = calibration
                self._geometry = None
                self._result = None
                self._last_pair = None
                self._last_capture_pair = pair_id
                self._settings = PanoramaSettings(
                    view='panorama', max_width=settings.max_width,
                    calibration_width=settings.calibration_width,
                    sync_tolerance_ms=settings.sync_tolerance_ms, swap_cameras=settings.swap_cameras,
                )
                self._notify(f'Панорама настроена: {calibration.inliers} надёжных совпадений, '
                             f'ошибка {calibration.rms_px:.2f} px. Сохраните настройку, если камеры закреплены неподвижно.')
        except PanoramaError as exc:
            with self.lock:
                self._notify(str(exc))
        except cv2.error:
            with self.lock:
                self._notify('Ошибка настройки панорамы. Проверьте освещение, фокус и общее поле зрения камер.')
        except Exception:
            with self.lock:
                self._notify('Не удалось настроить панораму. Оставьте камеры неподвижными и повторите.')
        finally:
            with self.lock:
                self._busy = False
                self.lock.notify_all()

    def _process_once(self):
        with self.lock:
            if not self._enabled or self._busy:
                return
            left, right = self._streams
            settings, calibration = self._settings, self._calibration
            geometry, previous, revision = self._geometry, self._last_pair, self._revision
        pair = self._fresh_pair(left, right, settings.sync_tolerance_ms, previous=previous)
        if pair is None:
            return
        left_frame, right_frame, skew, pair_id = pair
        if settings.swap_cameras:
            left_frame, right_frame = right_frame, left_frame
        started = time.perf_counter()
        requested_view = settings.view
        fallback = False
        status = 'Камеры рядом.'
        try:
            if requested_view == 'panorama':
                if calibration is None:
                    fallback = True
                    status = 'Панорама не настроена: показаны камеры рядом. Нажмите «Настроить панораму».'
                    frame = compose_side_by_side(left_frame, right_frame, settings.max_width)
                    view = 'side_by_side'
                else:
                    if geometry is None:
                        geometry = PanoramaGeometry(calibration, settings.max_width)
                    frame = geometry.render(left_frame, right_frame)
                    view = 'panorama'
                    status = 'Панорама работает.'
            else:
                frame = compose_side_by_side(left_frame, right_frame, settings.max_width)
                view = 'side_by_side'
        except PanoramaError as exc:
            fallback = True
            status = f'Панорама недоступна: {exc} Показаны камеры рядом.'
            frame = compose_side_by_side(left_frame, right_frame, settings.max_width)
            view = 'side_by_side'
            geometry = None
        except cv2.error:
            self._waiting('Ошибка склейки кадров. Повторите настройку панорамы или выберите режим «Рядом».')
            return
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        result = PanoramaResult(
            color_rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            image_size=(frame.shape[1], frame.shape[0]), view=view, requested_view=requested_view,
            sync_skew_ms=skew, elapsed_ms=elapsed_ms, created_at=time.perf_counter(), fallback=fallback,
        )
        with self.lock:
            if self.stopping or revision != self._revision or not self._enabled:
                return
            self._geometry = geometry
            self._result = result
            self._last_pair = pair_id
            self._status = status

    def _run(self):
        while True:
            with self.lock:
                if self.stopping:
                    return
            try:
                command = self.commands.get_nowait()
            except queue.Empty:
                command = False
            if command is None:
                return
            if command is False:
                try:
                    self._process_once()
                except Exception:
                    self._waiting('Ошибка общего вида. Выберите режим «Рядом» или повторите настройку панорамы.')
            else:
                self._handle_command(command)
            with self.lock:
                if self.stopping:
                    return
                self.lock.wait(0.01 if self._enabled else 0.05)
