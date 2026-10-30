"""Tests of edge overlay independent of a display or a physical camera."""
import unittest

import numpy as np

from edges import highlight_edges


class EdgeTests(unittest.TestCase):
    def test_uniform_frame_has_no_edges(self):
        frame = np.full((100, 100, 3), 90, dtype=np.uint8)
        original = frame.copy()
        self.assertIs(highlight_edges(frame, 50), frame)
        np.testing.assert_array_equal(frame, original)

    def test_rectangle_edges_are_purple_not_image_corners(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[25:75, 25:75] = 255
        highlight_edges(frame, 50)
        purple = np.all(frame == (180, 0, 140), axis=2)
        for area in (purple[20:30, 30:70], purple[70:80, 30:70],
                     purple[30:70, 20:30], purple[30:70, 70:80]):
            self.assertTrue(area.any())
        self.assertFalse(purple[0, 0])
        np.testing.assert_array_equal(frame[50, 50], (255, 255, 255))

    def test_sensitivity_reveals_weak_edges(self):
        original = np.zeros((100, 100, 3), dtype=np.uint8)
        original[25:75, 25:75] = 40
        low = highlight_edges(original.copy(), 1)
        high = highlight_edges(original.copy(), 100)
        self.assertFalse(np.any(np.all(low == (180, 0, 140), axis=2)))
        self.assertTrue(np.any(np.all(high == (180, 0, 140), axis=2)))


if __name__ == '__main__':
    unittest.main()
