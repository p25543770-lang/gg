"""Background stereo matching and calibration commands, with no Tk calls."""
from dataclasses import dataclass
import queue
import threading
import time

import cv2

from stereo import (
    StereoCalibration, StereoError, StereoProcessor, StereoResult, StereoSettings,
    calibrate_from_corners, detect_chessboard, validate_board,
)


@dataclass(frozen=True)
class StereoState:
    result: StereoResult | None
    status: str
    notice: str
    notice_id: int
    sample_count: int
    busy: bool
    calibration: StereoCalibration | None
    enabled: bool
    settings: StereoSettings


class StereoWorker:
    """Owns one worker thread; the UI only exchanges immutable snapshots and commands."""

    def __init__(self):
        self.lock = threading.Condition()
        self.commands = queue.Queue()
        self.stopping = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self._streams = (None, None)
        self._enabled = False
        self._settings = StereoSettings()
        self._calibration = None
        self._processor = None
        self._result = None
        self._status = 'Откройте обе камеры и загрузите калибровку.'
        self._notice = ''
        self._notice_id = 0
        self._samples = []
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

    def _invalidate(self):
        self._revision += 1
        self._processor = None
        self._result = None
        self._last_pair = None
        self._last_capture_pair = None

    def set_streams(self, left, right):
        with self.lock:
            streams = (left, right)
            if streams == self._streams:
                return
            self._streams = streams
            self._samples.clear()
            self._invalidate()
            self._status = 'Откройте обе камеры и загрузите калибровку.'
            self.lock.notify_all()

    def set_calibration(self, calibration):
        if calibration is not None and not isinstance(calibration, StereoCalibration):
            raise StereoError('Неверный объект калибровки.')
        with self.lock:
            self._calibration = calibration
            self._samples.clear()
            self._invalidate()
            self._status = 'Калибровка загружена.' if calibration else 'Калибровка не загружена.'
            self.lock.notify_all()

    def configure(self, settings):
        if not isinstance(settings, StereoSettings):
            raise StereoError('Неверные параметры стерео.')
        with self.lock:
            if settings == self._settings:
                return
            self._settings = settings
            self._invalidate()
            self.lock.notify_all()

    def set_enabled(self, enabled):
        with self.lock:
            self._enabled = bool(enabled)
            self._invalidate()
            self._status = 'Стерео включено.' if enabled else 'Стерео выключено.'
            self.lock.notify_all()

    def clear_samples(self):
        with self.lock:
            self._samples.clear()
            self._revision += 1
            self._notice = 'Пары снимков доски очищены.'
            self._notice_id += 1
            self.lock.notify_all()

    def request_capture(self, board_size, square_size_mm):
        board, square = validate_board(board_size, float(square_size_mm) / 1000.0)
        with self.lock:
            if self._busy:
                return False
            if any(stream is None for stream in self._streams):
                self._notify('Сначала откройте обе камеры.')
                return False
            if len(self._samples) >= 60:
                self._notify('Достигнут предел 60 пар. Откалибруйте или очистите пары.')
                return False
            self._busy = True
            self.commands.put(('capture', self._revision, board, square, time.perf_counter()))
            self.lock.notify_all()
            return True

    def request_calibration(self):
        with self.lock:
            if self._busy:
                return False
            if len(self._samples) < 10:
                self._notify('Нужно не менее 10 пар снимков доски; рекомендуется 15–30.')
                return False
            self._busy = True
            self.commands.put(('calibrate', self._revision))
            self.lock.notify_all()
            return True

    def poll(self):
        with self.lock:
            return StereoState(
                result=self._result, status=self._status, notice=self._notice,
                notice_id=self._notice_id, sample_count=len(self._samples), busy=self._busy,
                calibration=self._calibration, enabled=self._enabled, settings=self._settings,
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
            self._waiting('Ожидание новой пары кадров для снимка доски…')
            return None
        pair_id = (left_sequence, right_sequence)
        if previous is not None and pair_id == previous:
            return None
        if left_frame.shape != right_frame.shape:
            self._waiting('Размеры кадров различаются. Выберите одинаковое разрешение для обеих камер.')
            return None
        return left_frame, right_frame, skew, pair_id

    def _handle_command(self, command):
        kind, revision = command[:2]
        try:
            with self.lock:
                left, right = self._streams
                tolerance = self._settings.sync_tolerance_ms
            if kind == 'capture':
                _, board, square, requested_at = command
                deadline = time.perf_counter() + 5.0
                while time.perf_counter() < deadline:
                    with self.lock:
                        if self.stopping or revision != self._revision:
                            return
                        left, right = self._streams
                    pair = self._fresh_pair(left, right, tolerance, after=requested_at,
                                            previous=self._last_capture_pair)
                    if pair is not None:
                        break
                    with self.lock:
                        self.lock.wait(0.02)
                else:
                    with self.lock:
                        self._notify('Не удалось получить свежую синхронную пару для доски. Проверьте камеры и допуск.')
                    return
                left_frame, right_frame, _, pair_id = pair
                left_corners = detect_chessboard(left_frame, board)
                right_corners = detect_chessboard(right_frame, board)
                with self.lock:
                    if self.stopping or revision != self._revision:
                        return
                    self._last_capture_pair = pair_id
                    if left_corners is None or right_corners is None:
                        missing = 'в обеих камерах' if left_corners is None and right_corners is None else (
                            'в левой камере' if left_corners is None else 'в правой камере')
                        self._notify(f'Доска не найдена {missing}. Покажите все внутренние углы целиком, без бликов и смаза.')
                        return
                    height, width = left_frame.shape[:2]
                    metadata = (board, square, (width, height))
                    if self._samples and self._samples[0][:3] != metadata:
                        self._notify('Параметры доски или разрешение изменились. Очистите пары и снимите доску заново.')
                        return
                    self._samples.append((board, square, (width, height), left_corners, right_corners))
                    self._notify(f'Пара доски снята: {len(self._samples)}. Меняйте наклон, расстояние и положение доски.')
            elif kind == 'calibrate':
                with self.lock:
                    if revision != self._revision or self.stopping:
                        return
                    samples = self._samples[:]
                if not samples:
                    return
                board, square, size = samples[0][:3]
                if any(sample[:3] != (board, square, size) for sample in samples):
                    raise StereoError('Снимки доски сделаны с разными параметрами. Очистите пары и повторите.')
                calibration = calibrate_from_corners(
                    [sample[3] for sample in samples], [sample[4] for sample in samples],
                    board, square, size,
                )
                with self.lock:
                    if self.stopping or revision != self._revision:
                        return
                    self._calibration = calibration
                    self._invalidate()
                    self._notify(f'Калибровка готова: RMS {calibration.rms:.3f} px, '
                                 f'базовая линия {calibration.baseline_m * 1000:.1f} мм. Сохраните файл.')
        except Exception as exc:
            with self.lock:
                message = str(exc) if isinstance(exc, StereoError) else (
                    'Ошибка обработки доски. Проверьте освещение, фокус и параметры шаблона.')
                self._notify(message)
        finally:
            with self.lock:
                self._busy = False
                self.lock.notify_all()

    def _process_once(self):
        with self.lock:
            if not self._enabled:
                return
            left, right = self._streams
            calibration, settings = self._calibration, self._settings
            revision, previous = self._revision, self._last_pair
            processor = self._processor
        if calibration is None:
            self._waiting('Загрузите калибровку или снимите доску и нажмите «Калибровать». Без неё глубина не считается.')
            return
        pair = self._fresh_pair(left, right, settings.sync_tolerance_ms, previous=previous)
        if pair is None:
            with self.lock:
                if self._status != 'Стерео работает.':
                    self._result = None
            return
        left_frame, right_frame, skew, pair_id = pair
        try:
            if processor is None:
                processor = StereoProcessor(calibration, settings)
            result = processor.process(left_frame, right_frame, skew)
        except StereoError as exc:
            self._waiting(str(exc))
            return
        except cv2.error:
            self._waiting('Ошибка стереообработки. Проверьте калибровку, размер кадров и параметры.')
            return
        with self.lock:
            if self.stopping or revision != self._revision or not self._enabled:
                return
            self._processor = processor
            self._result = result
            self._last_pair = pair_id
            self._status = 'Стерео работает.'

    def _run(self):
        while True:
            with self.lock:
                if self.stopping:
                    return
            try:
                command = self.commands.get_nowait()
            except queue.Empty:
                command = False
            if command is None:  # stop sentinel
                return
            if command is False:
                try:
                    self._process_once()
                except Exception:
                    self._waiting('Ошибка стереопотока. Выключите и включите стерео; проверьте параметры.')
            else:
                self._handle_command(command)
            with self.lock:
                if self.stopping:
                    return
                self.lock.wait(0.01 if self._enabled else 0.05)
