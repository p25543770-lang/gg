"""Asynchronous camera capture and latest-frame processing, with no Tk calls."""
import sys
import threading
import time

import cv2

from edges import highlight_edges
from pose import PoseModelError, SkeletonDetector, PoseSettings
from robot_vision import RoadObstacleDetector, RobotSettings

MODES = ('edges', 'robot', 'skeleton')


class CameraStream:
    def __init__(self, index, capture_factory=None, resolution=(640, 480)):
        self.index = index
        self.capture_factory = capture_factory or cv2.VideoCapture
        self.resolution = resolution
        self.notice = ''
        self.lock = threading.Condition()
        self.stopping = False
        self.state = 'opening'
        self.raw = None
        self.processed = None
        self.sequence = 0
        self.result = None
        self.sensitivity = 50
        self.size = (640, 360)
        self.mode = 'edges'
        self.robot_settings = RobotSettings()
        self._robot_reset = True
        self.robot_detector = RoadObstacleDetector(self.robot_settings)
        self.pose_settings = PoseSettings()
        self._pose_reset = True
        self.pose_detector = SkeletonDetector(self.pose_settings)
        self.pose_error = ''
        self.pose_info = None
        self.capture_thread = threading.Thread(target=self._capture, daemon=True)
        self.process_thread = threading.Thread(target=self._process, daemon=True)

    def start(self):
        self.capture_thread.start()
        self.process_thread.start()

    def stop(self):
        with self.lock:
            self.stopping = True
            self.result = None
            self.processed = None
            self.lock.notify_all()

    def alive(self):
        return self.capture_thread.is_alive() or self.process_thread.is_alive()

    def configure(self, sensitivity, width, height, mode=None, robot_settings=None, pose_settings=None):
        with self.lock:
            self.sensitivity = sensitivity
            if width > 1 and height > 1:
                self.size = (width, height)
            if mode is not None and mode != self.mode:
                if mode not in MODES:
                    raise ValueError('mode must be one of ' + ', '.join(MODES))
                self.mode = mode
                self._robot_reset = True
                self._pose_reset = True
            if robot_settings is not None and robot_settings != self.robot_settings:
                self.robot_settings = robot_settings
                self._robot_reset = True
            if pose_settings is not None and pose_settings != self.pose_settings:
                self.pose_settings = pose_settings
                self._pose_reset = True

    def poll(self):
        with self.lock:
            result, self.result = self.result, None
            notice = self.notice
            if self.mode == 'skeleton' and self.pose_error:
                notice = (notice + ' ' + self.pose_error).strip()
            return self.state, result, notice

    def snapshot(self):
        """Latest untouched camera frame, acquisition time and sequence number.

        The frame is shared read-only with callers; preview processing must copy
        before drawing. Returns None until the camera is open and has a frame.
        """
        with self.lock:
            if self.state != 'open' or self.stopping or self.raw is None:
                return None
            frame, _started, acquired = self.raw
            return frame, acquired, self.sequence

    def snapshot_processed(self):
        """Latest processed frame (with the mode overlay), like snapshot().

        Same (frame, acquisition time, sequence) shape as snapshot(); the frame
        is shared read-only with callers. Returns None until the first frame
        has been processed, which keeps panorama pairing in sync with the
        camera sequence numbers.
        """
        with self.lock:
            if self.state != 'open' or self.stopping or self.processed is None:
                return None
            frame, acquired, sequence = self.processed
            return frame, acquired, sequence

    def _capture(self):
        camera = None
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
                camera.release()

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
                    mode, robot_settings = self.mode, self.robot_settings
                    pose_settings = self.pose_settings
                    robot_reset, pose_reset = self._robot_reset, self._pose_reset
                    self._robot_reset = False
                    self._pose_reset = False
                h, w = frame.shape[:2]
                scale = min(width / w, height / h, 1)
                if scale < 1:
                    frame = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
                else:
                    # snapshots share the captured array; edge drawing must not
                    # contaminate the stereo/calibration input.
                    frame = frame.copy()
                if mode == 'robot':
                    if robot_reset or self.robot_detector.settings != robot_settings:
                        self.robot_detector = RoadObstacleDetector(robot_settings)
                    frame, _robot_info = self.robot_detector.process(frame, sensitivity)
                elif mode == 'skeleton':
                    if self.pose_detector.settings != pose_settings:
                        self.pose_detector = SkeletonDetector(pose_settings)
                    elif pose_reset:
                        self.pose_detector.reset()
                    try:
                        frame, pose_info = self.pose_detector.process(frame, sensitivity)
                        error = ''
                    except PoseModelError as exc:
                        # Missing/corrupt model must not kill the preview.
                        pose_info, error = None, str(exc)
                        cv2.putText(frame, 'SKELETON MODEL UNAVAILABLE', (8, 24),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
                    with self.lock:
                        self.pose_error, self.pose_info = error, pose_info
                else:
                    highlight_edges(frame, sensitivity)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with self.lock:
                    if self.stopping:
                        return
                    self.result = (rgb, started, acquired, (w, h))
                    self.processed = (frame, acquired, last_seq)
        except Exception:
            with self.lock:
                if not self.stopping:
                    self.state = 'Ошибка обработки кадра.'
                self.stopping = True
                self.lock.notify_all()
