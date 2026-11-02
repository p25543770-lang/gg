"""Checks for the bundled human-skeleton overlay (an estimate, not a sensor)."""
import os
import unittest

import cv2
import numpy as np

from camera_stream import CameraStream
from pose import (
    KEYPOINT_NAMES, PoseModelError, PoseSettings, SkeletonDetector,
    models_present, resource_path,
)

cv2.setNumThreads(1)
SAMPLE = resource_path('assets', 'pose_sample.jpg')


def sample_frame(size=(320, 320)):
    image = cv2.imread(SAMPLE)
    if image is None:
        raise AssertionError('bundled pose sample is missing: ' + SAMPLE)
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def overlay_pixels(frame):
    """Count pixels painted in the exact skeleton limb colours (BGR)."""
    colors = ((0, 220, 0), (0, 160, 255), (255, 220, 0))
    mask = np.zeros(frame.shape[:2], dtype=bool)
    for color in colors:
        mask |= np.all(frame == color, axis=2)
    return int(np.count_nonzero(mask))


class PoseSettingsTests(unittest.TestCase):
    def test_invalid_settings_rejected(self):
        for kwargs in ({'score_threshold': 0.0}, {'nms_threshold': 1.0}, {'max_people': 0},
                       {'max_people': 99}, {'smooth_alpha': 0.0}, {'max_missed_frames': -1},
                       {'keypoint_threshold': 2.0}, {'min_box_pixels': 1}, {'detect_interval': 0}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    PoseSettings(**kwargs)

    def test_keypoint_layout_matches_coco(self):
        self.assertEqual(len(KEYPOINT_NAMES), 17)
        self.assertEqual(KEYPOINT_NAMES[0], 'nose')
        self.assertEqual(KEYPOINT_NAMES[-1], 'right_ankle')


class SkeletonDetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not models_present():
            raise unittest.SkipTest('bundled ONNX models are missing')

    def test_person_is_found_with_seventeen_keypoints(self):
        detector = SkeletonDetector()
        frame = sample_frame()
        out, info = detector.process(frame, 50)
        self.assertEqual(out.shape, frame.shape)
        self.assertEqual(info.person_count, 1)
        person = info.people[0]
        self.assertEqual(person.keypoints.shape, (17, 3))
        self.assertGreater(person.score, 0.5)
        self.assertGreaterEqual(person.visible_keypoints, 8)
        self.assertIn('PEOPLE: 1', info.status)
        self.assertGreater(info.inference_ms, 0.0)
        # The nose must land on the head, in the upper part of this portrait.
        nose = person.keypoints[0]
        self.assertGreater(nose[2], 0.3)
        self.assertTrue(120 < nose[0] < 300, nose)
        self.assertTrue(20 < nose[1] < 160, nose)

    def test_overlay_is_drawn_without_mutating_the_input(self):
        detector = SkeletonDetector()
        frame = sample_frame()
        original = frame.copy()
        out, _info = detector.process(frame, 50)
        self.assertTrue(np.array_equal(frame, original), 'process() must not modify the camera frame')
        self.assertGreater(overlay_pixels(out), 200)
        self.assertLess(overlay_pixels(frame), 50)

    def test_blank_frame_reports_no_people(self):
        detector = SkeletonDetector()
        frame = np.full((240, 320, 3), 127, dtype=np.uint8)
        out, info = detector.process(frame, 50)
        self.assertEqual(info.person_count, 0)
        self.assertEqual(info.status, 'NO PEOPLE')
        self.assertEqual(overlay_pixels(out), 0)

    def test_track_id_is_stable_and_smoothing_keeps_points_close(self):
        detector = SkeletonDetector()
        frame = sample_frame()
        _, first = detector.process(frame, 50)
        _, second = detector.process(frame, 50)
        self.assertEqual(first.people[0].track_id, second.people[0].track_id)
        delta = np.abs(first.people[0].keypoints[:, :2] - second.people[0].keypoints[:, :2])
        self.assertLess(float(delta.max()), 8.0)
        detector.reset()
        _, after_reset = detector.process(frame, 50)
        self.assertEqual(after_reset.people[0].track_id, 1)

    def test_max_people_limits_the_number_of_skeletons(self):
        frame = sample_frame()
        # Two copies of the portrait side by side produce two detections.
        pair = np.concatenate([frame, frame], axis=1)
        one = SkeletonDetector(PoseSettings(max_people=1))
        two = SkeletonDetector(PoseSettings(max_people=2))
        _, info_one = one.process(pair, 50)
        _, info_two = two.process(pair, 50)
        self.assertEqual(info_one.person_count, 1)
        self.assertEqual(info_two.person_count, 2)
        self.assertNotEqual(info_two.people[0].track_id, info_two.people[1].track_id)

    def test_high_sensitivity_keeps_at_least_as_many_keypoints(self):
        frame = sample_frame()
        _, low = SkeletonDetector().process(frame, 1)
        _, high = SkeletonDetector().process(frame, 100)
        self.assertGreaterEqual(high.people[0].visible_keypoints, low.people[0].visible_keypoints)

    def test_tiny_and_invalid_frames(self):
        detector = SkeletonDetector()
        tiny = np.zeros((8, 8, 3), dtype=np.uint8)
        out, info = detector.process(tiny, 50)
        self.assertEqual(out.shape, tiny.shape)
        self.assertEqual(info.person_count, 0)
        with self.assertRaises(ValueError):
            detector.process(np.zeros((20, 20), dtype=np.uint8), 50)

    def test_missing_model_raises_a_readable_error(self):
        detector = SkeletonDetector(detector_path=os.path.join(resource_path('models'), 'absent.onnx'))
        with self.assertRaises(PoseModelError) as caught:
            detector.process(sample_frame(), 50)
        self.assertIn('модель', str(caught.exception).lower())


class StreamSkeletonTests(unittest.TestCase):
    def test_stream_accepts_skeleton_mode_and_rejects_unknown_modes(self):
        stream = CameraStream(0, lambda *_: None)
        stream.configure(50, 320, 240, mode='skeleton')
        self.assertEqual(stream.mode, 'skeleton')
        stream.configure(50, 320, 240, mode='skeleton', pose_settings=PoseSettings(max_people=2))
        self.assertEqual(stream.pose_settings.max_people, 2)
        self.assertTrue(stream._pose_reset)
        with self.assertRaises(ValueError):
            stream.configure(50, 320, 240, mode='depth')

    def test_missing_model_does_not_break_the_preview(self):
        class FakeCamera:
            def __init__(self):
                self.released = False

            def isOpened(self):
                return True

            def set(self, *_args):
                return True

            def read(self):
                return True, sample_frame((160, 160))

            def release(self):
                self.released = True

        stream = CameraStream(0, lambda *_: FakeCamera())
        stream.pose_detector = SkeletonDetector(detector_path=resource_path('models', 'absent.onnx'))
        stream.configure(50, 160, 160, mode='skeleton')
        stream.start()
        try:
            deadline = __import__('time').perf_counter() + 15
            while __import__('time').perf_counter() < deadline:
                state, result, notice = stream.poll()
                if result is not None:
                    self.assertEqual(state, 'open')
                    self.assertIn('модель', notice.lower())
                    self.assertEqual(result[0].shape, (160, 160, 3))
                    break
                __import__('time').sleep(0.01)
            else:
                self.fail('stream produced no frame with a missing pose model')
        finally:
            stream.stop()
            stream.capture_thread.join(5)
            stream.process_thread.join(5)
        self.assertFalse(stream.alive())


if __name__ == '__main__':
    unittest.main()
