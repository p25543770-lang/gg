"""Synthetic stereo tests: depth scale, calibration files and worker synchronization."""
import os
import tempfile
import time
import unittest

import cv2
import numpy as np

from stereo import (
    StereoCalibration, StereoError, StereoProcessor, StereoSettings,
    calibrate_from_corners, load_calibration, save_calibration,
)
from stereo_worker import StereoWorker


def synthetic_pair(width=320, height=240, disparity=20):
    rng = np.random.default_rng(42)
    texture = rng.integers(0, 256, (height, width + 80), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (3, 3), 0)
    left = cv2.cvtColor(texture[:, 20:20 + width], cv2.COLOR_GRAY2BGR)
    right = cv2.cvtColor(texture[:, 20 + disparity:20 + disparity + width], cv2.COLOR_GRAY2BGR)
    return left, right


def ideal_calibration(width=320, height=240, baseline=0.1, unit='m'):
    k = np.array([[500.0, 0, width / 2], [0, 500.0, height / 2], [0, 0, 1]])
    translation = np.array([[-baseline], [0], [0]])
    return StereoCalibration(k, np.zeros(5), k, np.zeros(5), np.eye(3), translation,
                             (width, height), translation_unit=unit)


class FakeStream:
    def __init__(self, frame, timestamp=None, sequence=1):
        self.frame = frame
        self.timestamp = time.perf_counter() if timestamp is None else timestamp
        self.sequence = sequence
        self.running = True

    def alive(self):
        return self.running

    def snapshot(self):
        if not self.running:
            return None
        return self.frame, self.timestamp, self.sequence


class StereoProcessorTests(unittest.TestCase):
    def test_metric_depth_from_known_disparity(self):
        left, right = synthetic_pair(disparity=20)
        result = StereoProcessor(ideal_calibration(baseline=0.1),
                                 StereoSettings(max_width=320, num_disparities=64)).process(left, right)
        self.assertEqual(result.image_size, (320, 240))
        self.assertGreater(result.valid_fraction, 0.5)
        self.assertAlmostEqual(float(np.nanmedian(result.disparity)), 20.0, delta=0.6)
        # Z = fx * baseline / disparity = 500 * 0.1 / 20 = 2.5 m.
        self.assertAlmostEqual(float(np.nanmedian(result.depth_m)), 2.5, delta=0.1)
        self.assertEqual(result.color_rgb.shape, (240, 320, 3))

    def test_downscaling_preserves_metric_depth(self):
        left, right = synthetic_pair(disparity=20)
        result = StereoProcessor(ideal_calibration(baseline=0.1),
                                 StereoSettings(max_width=160, num_disparities=16)).process(left, right)
        self.assertEqual(result.image_size, (160, 120))
        self.assertAlmostEqual(float(np.nanmedian(result.depth_m)), 2.5, delta=0.2)

    def test_unknown_translation_unit_never_reports_metres(self):
        left, right = synthetic_pair()
        calibration = ideal_calibration(baseline=100, unit='unknown')
        result = StereoProcessor(calibration, StereoSettings(max_width=320, num_disparities=64)).process(left, right)
        self.assertIsNone(result.baseline_m)
        self.assertIsNone(result.depth_m)
        self.assertEqual(result.view, 'disparity')

    def test_resolution_mismatch_is_actionable(self):
        left, right = synthetic_pair(width=640, height=480)
        with self.assertRaisesRegex(StereoError, 'Калибровка для'):
            StereoProcessor(ideal_calibration(width=320, height=240)).process(left, right)

    def test_calibration_yaml_json_round_trip(self):
        calibration = ideal_calibration()
        with tempfile.TemporaryDirectory() as directory:
            for extension in ('.yml', '.json'):
                path = os.path.join(directory, 'calibration' + extension)
                save_calibration(calibration, path)
                loaded = load_calibration(path)
                self.assertEqual(loaded.image_size, calibration.image_size)
                self.assertAlmostEqual(loaded.baseline_m, calibration.baseline_m)
                np.testing.assert_allclose(loaded.K_left, calibration.K_left)
                np.testing.assert_allclose(loaded.T, calibration.T)

    def test_load_can_override_unknown_units(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'unknown.yml')
            storage = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
            storage.write('image_width', 320)
            storage.write('image_height', 240)
            storage.write('translation_unit', 'unknown')
            for name in ('K1', 'K2'):
                storage.write(name, np.array([[500.0, 0, 160], [0, 500.0, 120], [0, 0, 1]]))
            for name in ('D1', 'D2'):
                storage.write(name, np.zeros((1, 5)))
            storage.write('R', np.eye(3))
            storage.write('T', np.array([[-100.0], [0], [0]]))
            storage.release()
            self.assertIsNone(load_calibration(path).baseline_m)
            self.assertAlmostEqual(load_calibration(path, override_unit='mm').baseline_m, 0.1)

    def test_invalid_calibration_is_rejected(self):
        with self.assertRaises(StereoError):
            ideal_calibration(baseline=0)
        with self.assertRaises(StereoError):
            StereoSettings(num_disparities=63)
        with self.assertRaises(StereoError):
            StereoSettings(min_depth_m=5, max_depth_m=1)

    def test_calibration_from_synthetic_board_projections(self):
        columns, rows, square = 9, 6, 0.025
        grid = np.zeros((columns * rows, 3), np.float32)
        grid[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
        grid *= square
        k = np.array([[520.0, 0, 320], [0, 520.0, 240], [0, 0, 1]])
        distortion = np.zeros(5)
        rotation = np.eye(3)
        translation = np.array([[-0.06], [0], [0]])
        left_points, right_points = [], []
        for i in range(14):
            rvec = np.array([[0.12 * np.sin(i), 0.15 * np.cos(i), 0.08 * np.sin(i * 0.7)]])
            tvec = np.array([[0.02 * np.sin(i * 1.3)], [0.015 * np.cos(i * 0.9)], [0.45 + 0.05 * np.cos(i)]])
            left, _ = cv2.projectPoints(grid, rvec, tvec, k, distortion)
            right, _ = cv2.projectPoints(grid, rvec, rotation @ tvec + translation, k, distortion)
            left_points.append(left.astype(np.float32))
            right_points.append(right.astype(np.float32))
        calibration = calibrate_from_corners(left_points, right_points, (columns, rows), square, (640, 480))
        self.assertAlmostEqual(calibration.baseline_m, 0.06, delta=0.002)
        self.assertLess(calibration.rms, 0.2)


class StereoWorkerTests(unittest.TestCase):
    def wait_until(self, worker, predicate, timeout=4):
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            state = worker.poll()
            if predicate(state):
                return state
            time.sleep(0.01)
        self.fail('Timed out waiting for stereo worker; last status: ' + worker.poll().status)

    def test_worker_publishes_depth_and_rejects_skewed_frames(self):
        left, right = synthetic_pair()
        now = time.perf_counter()
        left_stream, right_stream = FakeStream(left, now), FakeStream(right, now)
        worker = StereoWorker()
        worker.start()
        try:
            worker.set_streams(left_stream, right_stream)
            worker.set_calibration(ideal_calibration())
            worker.configure(StereoSettings(max_width=320, num_disparities=64, sync_tolerance_ms=40))
            worker.set_enabled(True)
            state = self.wait_until(worker, lambda s: s.result is not None)
            self.assertAlmostEqual(float(np.nanmedian(state.result.depth_m)), 2.5, delta=0.1)

            right_stream.timestamp = left_stream.timestamp + 0.2
            right_stream.sequence += 1
            state = self.wait_until(worker, lambda s: 'рассинхронизированы' in s.status)
            self.assertIsNone(state.result)

            right_stream.timestamp = left_stream.timestamp = time.perf_counter()
            right_stream.sequence += 1
            self.wait_until(worker, lambda s: s.result is not None)
        finally:
            worker.stop()
            worker.thread.join(2)
        self.assertFalse(worker.alive())

    def test_capture_requires_two_cameras(self):
        worker = StereoWorker()
        worker.start()
        try:
            self.assertFalse(worker.request_capture((9, 6), 25))
            self.assertIn('Сначала откройте обе камеры', worker.poll().notice)
        finally:
            worker.stop()
            worker.thread.join(2)


if __name__ == '__main__':
    unittest.main()
