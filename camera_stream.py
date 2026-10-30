"""Asynchronous camera capture and latest-frame processing, with no Tk calls."""
import sys
import threading
import time

import cv2

from edges import highlight_edges


class CameraStream:
    def __init__(self, index, capture_factory=None, manual_exposure=False, resolution=(640, 480)):
        self.index = index
        self.capture_factory = capture_factory or cv2.VideoCapture
        self.manual_exposure = manual_exposure
        self.resolution = resolution
        self.notice = ''
        self.lock = threading.Condition()
        self.stopping = False
        self.state = 'opening'
        self.raw = None
        self.sequence = 0
        self.result = None
        self.sensitivity = 50
        self.size = (640, 360)
        self.capture_thread = threading.Thread(target=self._capture, daemon=True)
        self.process_thread = threading.Thread(target=self._process, daemon=True)

    def start(self):
        self.capture_thread.start()
        self.process_thread.start()

    def stop(self):
        with self.lock:
            self.stopping = True
            self.result = None
            self.lock.notify_all()

    def alive(self):
        return self.capture_thread.is_alive() or self.process_thread.is_alive()

    def configure(self, sensitivity, width, height):
        with self.lock:
            self.sensitivity = sensitivity
            if width > 1 and height > 1:
                self.size = (width, height)

    def poll(self):
        with self.lock:
            result, self.result = self.result, None
            return self.state, result, self.notice

    def _capture(self):
        camera = None
        old_auto = None
        old_exposure = None
        exposure_attempted = False
        try:
            camera = self.capture_factory(self.index, cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_ANY)
            if not camera.isOpened():
                raise RuntimeError('Не удалось открыть камеру. Попробуйте другой номер.')
            # Requests, not guarantees: camera, backend and USB bandwidth decide.
            camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            camera.set(cv2.CAP_PROP_FPS, 30)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if self.manual_exposure and sys.platform == 'win32':
                # DirectShow uses 0.25 for manual and exposure -5 for ~1/32 s.
                # Only change settings if the previous mode can be restored.
                old_auto = camera.get(cv2.CAP_PROP_AUTO_EXPOSURE)
                old_exposure = camera.get(cv2.CAP_PROP_EXPOSURE)
                if old_auto not in (0.25, 0.75):
                    self.notice = 'Не удалось прочитать режим экспозиции; оставлен текущий.'
                else:
                    exposure_attempted = True
                    manual_ok = camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                    exposure_ok = camera.set(cv2.CAP_PROP_EXPOSURE, -5) if manual_ok else False
                    if not (manual_ok and exposure_ok):
                        self.notice = 'Камера не приняла ручную экспозицию; FPS может быть ниже 30.'
                        self._restore_exposure(camera, old_auto, old_exposure)
                        exposure_attempted = False
            with self.lock:
                if self.stopping:
                    return
                self.state = 'open'
            while True:
                with self.lock:
                    if self.stopping:
                        return
                started = time.perf_counter()
                ok, frame = camera.read()
                acquired = time.perf_counter()
                if not ok or frame is None:
                    raise RuntimeError('Не удалось получить кадр. Проверьте подключение камеры.')
                with self.lock:
                    if self.stopping:
                        return
                    # Overwrite, don't queue: processing always takes the latest frame.
                    self.raw = (frame, started, acquired)
                    self.sequence += 1
                    self.lock.notify_all()
        except Exception as exc:
            with self.lock:
                if not self.stopping:
                    self.state = str(exc) or 'Ошибка камеры.'
                self.stopping = True
                self.lock.notify_all()
        finally:
            if camera is not None:
                if exposure_attempted:
                    self._restore_exposure(camera, old_auto, old_exposure)
                camera.release()

    @staticmethod
    def _restore_exposure(camera, old_auto, old_exposure):
        # Best effort: some drivers do not allow restoring properties on close.
        try:
            if old_auto == 0.25 and -20 <= old_exposure <= 0:
                camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                camera.set(cv2.CAP_PROP_EXPOSURE, old_exposure)
            camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, old_auto)
        except Exception:
            pass

    def _process(self):
        last_seq = 0
        try:
            while True:
                with self.lock:
                    while not self.stopping and self.sequence == last_seq:
                        self.lock.wait()
                    if self.stopping:
                        return
                    frame, started, acquired = self.raw
                    last_seq = self.sequence
                    sensitivity, (width, height) = self.sensitivity, self.size
                h, w = frame.shape[:2]
                scale = min(width / w, height / h, 1)
                if scale < 1:
                    frame = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
                highlight_edges(frame, sensitivity)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with self.lock:
                    if self.stopping:
                        return
                    self.result = (rgb, started, acquired, (w, h))
        except Exception:
            with self.lock:
                if not self.stopping:
                    self.state = 'Ошибка обработки кадра.'
                self.stopping = True
                self.lock.notify_all()
