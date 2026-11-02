"""Synthetic checks for the robot visual-aid overlay (not a safety validation)."""
import unittest

import cv2
import numpy as np

from camera_stream import CameraStream
from robot_vision import RoadObstacleDetector, RobotSettings


def make_scene():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:] = (110, 110, 110)  # uniform ground
    cv2.rectangle(frame, (130, 120), (190, 220), (20, 20, 220), -1)  # red obstacle
    return frame


class RobotVisionTests(unittest.TestCase):
    def test_uniform_frame_has_no_obstacles(self):
        detector = RoadObstacleDetector()
        frame = np.full((160, 200, 3), 120, dtype=np.uint8)
        out, info = detector.process(frame, 50)
        self.assertEqual(out.shape, frame.shape)
        self.assertEqual(info.obstacle_count, 0)
        self.assertFalse(info.center_blocked)
        self.assertGreater(info.road_fraction, 0.5)

    def test_obstacle_is_detected_and_center_blocked(self):
        detector = RoadObstacleDetector()
        frame = make_scene()
        out, info = detector.process(frame, 60)
        self.assertGreaterEqual(info.obstacle_count, 1)
        self.assertTrue(info.center_blocked)
        self.assertTrue(np.any(np.all(out == (0, 0, 255), axis=2)))
        # The road is tinted green by alpha blending, so test the dominant hue.
        greenish = (out[:, :, 1].astype(int) > out[:, :, 0] + 20) & (out[:, :, 1].astype(int) > out[:, :, 2] + 20)
        self.assertGreater(int(np.count_nonzero(greenish)), 1000)

    def test_track_id_persists_for_same_obstacle(self):
        detector = RoadObstacleDetector()
        frame = make_scene()
        _, first = detector.process(frame, 60)
        _, second = detector.process(frame, 60)
        self.assertTrue(first.tracks and second.tracks)
        first_ids = {track.track_id for track in first.tracks}
        second_ids = {track.track_id for track in second.tracks}
        self.assertTrue(first_ids & second_ids)

    def test_invalid_settings_rejected(self):
        with self.assertRaises(ValueError):
            RobotSettings(horizon_fraction=0.9)
        with self.assertRaises(ValueError):
            RobotSettings(min_obstacle_area_fraction=0.5, max_obstacle_area_fraction=0.4)

    def test_stream_robot_mode_does_not_mutate_snapshot(self):
        class SceneCamera:
            def __init__(self):
                self.frame = make_scene()
                self.released = False

            def isOpened(self):
                return True

            def set(self, *_):
                return True

            def read(self):
                return True, self.frame.copy()

            def release(self):
                self.released = True

        camera = SceneCamera()
        stream = CameraStream(0, lambda *_: camera)
        stream.configure(60, 320, 240, mode='robot')
        stream.start()
        try:
            import time
            deadline = time.perf_counter() + 2
            result = None
            while result is None and time.perf_counter() < deadline:
                _, result, _ = stream.poll()
                time.sleep(0.005)
            self.assertIsNotNone(result)
            snapshot = stream.snapshot()
            self.assertIsNotNone(snapshot)
            raw = snapshot[0]
            # The raw snapshot must still contain the original red obstacle,
            # not the green/red overlay drawn by the robot mode.
            self.assertTrue(np.any(np.all(raw == (20, 20, 220), axis=2)))
            self.assertFalse(np.any(np.all(raw == (0, 180, 0), axis=2)))
        finally:
            stream.stop()
            stream.capture_thread.join(2)
            stream.process_thread.join(2)
        self.assertTrue(camera.released)


if __name__ == '__main__':
    unittest.main()
