"""Exercise independent capture, latest-frame handling, pacing and cleanup."""
import threading
import time
import unittest

import cv2
import numpy as np

from camera_stream import CameraStream


class FakeCamera:
    def __init__(self, delay=0.005, fail=False):
        self.delay = delay
        self.fail = fail
        self.count = 0
        self.released = False
        self.properties = {}
        self.lock = threading.Lock()

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.properties[prop] = value
        return True

    def read(self):
        time.sleep(self.delay)
        if self.fail:
            return False, None
        with self.lock:
            self.count += 1
            count = self.count
        return True, np.full((64, 64, 3), count % 256, dtype=np.uint8)

    def release(self):
        self.released = True


class StreamTests(unittest.TestCase):
    def stop(self, *streams):
        for stream in streams:
            stream.stop()
        for stream in streams:
            stream.capture_thread.join(2)
            stream.process_thread.join(2)
            self.assertFalse(stream.alive())

    def test_fast_stream_not_blocked_by_slow_camera_and_uses_latest_frames(self):
        fast, slow = FakeCamera(), FakeCamera(delay=0.12)
        streams = [CameraStream(0, lambda *_: fast), CameraStream(1, lambda *_: slow)]
        for stream in streams:
            stream.start()
        times = []
        most_recent = None
        try:
            deadline = time.perf_counter() + 1.2
            while time.perf_counter() < deadline:
                state, result = streams[0].poll()
                self.assertIn(state, ('opening', 'open'))
                if result is not None:
                    times.append(time.perf_counter())
                    most_recent = int(result[0][0, 0, 0])
                time.sleep(0.002)
            self.assertGreaterEqual(len(times), 25)
            self.assertLessEqual(len(times), 38)
            with fast.lock:
                current = fast.count % 256
            self.assertLess((current - most_recent) % 256, 15)
            self.assertEqual(fast.properties[cv2.CAP_PROP_FPS], 30)
            self.assertEqual(fast.properties[cv2.CAP_PROP_FOURCC], cv2.VideoWriter_fourcc(*'MJPG'))
            self.assertEqual(streams[1].poll()[0], 'open')
        finally:
            self.stop(*streams)
        self.assertTrue(fast.released and slow.released)

    def test_failed_camera_does_not_stop_other(self):
        failed, good = FakeCamera(fail=True), FakeCamera()
        streams = [CameraStream(0, lambda *_: failed), CameraStream(1, lambda *_: good)]
        for stream in streams:
            stream.start()
        try:
            deadline = time.perf_counter() + 2
            while streams[0].alive() and time.perf_counter() < deadline:
                time.sleep(0.01)
            self.assertFalse(streams[0].alive())
            self.assertIn('Не удалось получить кадр', streams[0].poll()[0])
            self.assertTrue(streams[1].alive())
        finally:
            self.stop(*streams)
        self.assertTrue(failed.released and good.released)


if __name__ == '__main__':
    unittest.main()
