"""Synthetic checks for wide composition, panorama calibration and worker sync."""
import os
import tempfile
import time
import unittest

import cv2
import numpy as np

from panorama import (
    PanoramaCalibration, PanoramaError, PanoramaGeometry, PanoramaSettings,
    calibrate_panorama, compose_side_by_side, load_panorama, save_panorama,
)
from panorama_worker import PanoramaWorker


def translated_pair(width=480, height=360, shift=200):
    rng = np.random.default_rng(7)
    texture = rng.integers(0, 256, (height, width + shift + 40), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (3, 3), 0)
    left = cv2.cvtColor(texture[:, 20:20 + width], cv2.COLOR_GRAY2BGR)
    right = cv2.cvtColor(texture[:, 20 + shift:20 + shift + width], cv2.COLOR_GRAY2BGR)
    return left, right


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


class PanoramaCoreTests(unittest.TestCase):
    def test_side_by_side_preserves_aspect_and_width_limit(self):
        left = np.zeros((240, 320, 3), np.uint8)
        right = np.zeros((240, 640, 3), np.uint8)
        out = compose_side_by_side(left, right, max_width=640)
        self.assertLessEqual(out.shape[1], 640)
        height = out.shape[0]
        left_width = int(320 * height / 240)
        self.assertAlmostEqual(left_width / height, 320 / 240, delta=0.02)
        self.assertTrue(np.all(out[:, left_width:left_width + 2] == (55, 55, 55)))

    def test_calibration_recovers_translation_and_renders_wider_canvas(self):
        left, right = translated_pair(shift=200)
        calibration = calibrate_panorama(left, right, PanoramaSettings(calibration_width=480))
        self.assertGreater(calibration.inliers, 100)
        self.assertLess(calibration.rms_px, 0.5)
        self.assertAlmostEqual(calibration.homography[0, 2], 200.0, delta=1.0)
        geometry = PanoramaGeometry(calibration, max_width=1200)
        out = geometry.render(left, right)
        self.assertGreaterEqual(out.shape[1], 680)
        self.assertLessEqual(out.shape[1], 682)
        self.assertGreaterEqual(out.shape[0], 360)
        self.assertLessEqual(out.shape[0], 362)
        # An ideal translation gives the exact native canvas and preserves the
        # non-overlap part of the right image without blending it away.
        ideal = PanoramaCalibration(np.array([[1.0, 0, 200], [0, 1.0, 0], [0, 0, 1]]),
                                    (480, 360), (480, 360))
        ideal_geometry = PanoramaGeometry(ideal, max_width=1200)
        self.assertEqual(ideal_geometry.image_size, (680, 360))
        ideal_out = ideal_geometry.render(left, right)
        np.testing.assert_array_equal(ideal_out[40:320, 500:660], right[40:320, 300:460])

    def test_swap_cameras_is_applied_before_calibration(self):
        left, right = translated_pair(shift=160)
        normal = calibrate_panorama(left, right, PanoramaSettings(calibration_width=480))
        swapped = calibrate_panorama(right, left, PanoramaSettings(calibration_width=480, swap_cameras=True))
        np.testing.assert_allclose(normal.homography, swapped.homography, atol=1e-6)

    def test_uniform_scene_is_rejected_with_actionable_message(self):
        frame = np.full((240, 320, 3), 120, np.uint8)
        with self.assertRaisesRegex(PanoramaError, 'Мало деталей'):
            calibrate_panorama(frame, frame)

    def test_resolution_change_requires_new_calibration(self):
        left, right = translated_pair(shift=160)
        calibration = calibrate_panorama(left, right, PanoramaSettings(calibration_width=480))
        geometry = PanoramaGeometry(calibration)
        with self.assertRaisesRegex(PanoramaError, 'Разрешение камер изменилось'):
            geometry.render(cv2.resize(left, (320, 240)), right)

    def test_yaml_json_round_trip_and_invalid_file(self):
        left, right = translated_pair(shift=160)
        calibration = calibrate_panorama(left, right, PanoramaSettings(calibration_width=480))
        with tempfile.TemporaryDirectory() as directory:
            for extension in ('.yml', '.json'):
                path = os.path.join(directory, 'panorama' + extension)
                save_panorama(calibration, path)
                loaded = load_panorama(path)
                self.assertEqual(loaded.left_size, calibration.left_size)
                self.assertEqual(loaded.right_size, calibration.right_size)
                np.testing.assert_allclose(loaded.homography, calibration.homography)
            bad = os.path.join(directory, 'bad.yml')
            with open(bad, 'w', encoding='utf-8') as handle:
                handle.write('%YAML:1.0\nfoo: 1\n')
            with self.assertRaises(PanoramaError):
                load_panorama(bad)

    def test_extreme_homography_is_rejected(self):
        with self.assertRaises(PanoramaError):
            PanoramaGeometry(PanoramaCalibration(np.array([[1, 0, 5000], [0, 1, 0], [0.01, 0, 1]]),
                                                 (320, 240), (320, 240)))


class PanoramaWorkerTests(unittest.TestCase):
    def wait_until(self, worker, predicate, timeout=5):
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            state = worker.poll()
            if predicate(state):
                return state
            time.sleep(0.01)
        self.fail('Timed out waiting for panorama worker; last status: ' + worker.poll().status)

    def test_worker_publishes_side_by_side_and_panorama_fallback(self):
        left, right = translated_pair(width=320, height=240, shift=120)
        worker = PanoramaWorker()
        worker.set_streams(FakeStream(left), FakeStream(right))
        worker.configure(PanoramaSettings(max_width=640, sync_tolerance_ms=100))
        worker.set_enabled(True)
        worker.start()
        try:
            state = self.wait_until(worker, lambda s: s.result is not None)
            self.assertEqual(state.result.view, 'side_by_side')
            self.assertLessEqual(state.result.image_size[0], 640)
            self.assertGreater(state.result.image_size[0], 600)
            worker.configure(PanoramaSettings(view='panorama', max_width=640, sync_tolerance_ms=100))
            state = self.wait_until(worker, lambda s: s.result is not None and s.result.fallback)
            self.assertEqual(state.result.view, 'side_by_side')
            self.assertIn('не настроена', state.status)
        finally:
            worker.stop()
            worker.thread.join(2)
        self.assertFalse(worker.alive())

    def test_worker_calibrates_from_fresh_pair_and_rejects_skew(self):
        left, right = translated_pair(width=320, height=240, shift=120)
        now = time.perf_counter()
        left_stream, right_stream = FakeStream(left, now), FakeStream(right, now)
        worker = PanoramaWorker()
        worker.set_streams(left_stream, right_stream)
        worker.configure(PanoramaSettings(max_width=640, calibration_width=320, sync_tolerance_ms=50))
        worker.set_enabled(True)
        worker.start()
        try:
            self.wait_until(worker, lambda s: s.result is not None)
            self.assertTrue(worker.request_calibration())
            state = self.wait_until(worker, lambda s: s.calibration is not None and not s.busy, timeout=8)
            self.assertEqual(state.settings.view, 'panorama')
            state = self.wait_until(worker, lambda s: s.result is not None and s.result.view == 'panorama')
            self.assertGreater(state.result.image_size[0], 320)
            # A stale/skewed pair must not be presented as a synchronized panorama.
            right_stream.timestamp = time.perf_counter() - 1.0
            right_stream.sequence += 1
            self.wait_until(worker, lambda s: s.result is None and 'рассинхронизированы' in s.status)
        finally:
            worker.stop()
            worker.thread.join(2)

    def test_changing_streams_clears_calibration(self):
        left, right = translated_pair(width=320, height=240, shift=120)
        worker = PanoramaWorker()
        worker.set_streams(FakeStream(left), FakeStream(right))
        worker.set_calibration(PanoramaCalibration(np.eye(3), (320, 240), (320, 240)))
        self.assertIsNotNone(worker.poll().calibration)
        worker.set_streams(FakeStream(left), FakeStream(right))
        self.assertIsNone(worker.poll().calibration)


if __name__ == '__main__':
    unittest.main()
