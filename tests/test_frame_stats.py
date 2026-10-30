"""Frame counters can be tested with deterministic timestamps."""
import unittest

from frame_stats import FrameStats


class FrameStatsTests(unittest.TestCase):
    def test_counter_and_rolling_window(self):
        stats = FrameStats(window=3)
        self.assertEqual(stats.text(), 'FPS: —  |  В приложении: —  |  Чтение: —')
        stats.add(0.97, 0.99, 1.0)
        self.assertEqual(stats.text(), 'FPS: —  |  В приложении: 10.0 мс  |  Чтение: 20.0 мс')
        stats.add(1.07, 1.09, 1.1)
        stats.add(1.16, 1.19, 1.2)
        self.assertEqual(stats.text(), 'FPS: 10.0  |  В приложении: 10.0 мс  |  Чтение: 23.3 мс')
        stats.add(1.25, 1.29, 1.3)
        self.assertEqual(stats.text(), 'FPS: 10.0  |  В приложении: 10.0 мс  |  Чтение: 30.0 мс')

    def test_instances_are_independent_and_reset(self):
        left, right = FrameStats(), FrameStats()
        left.add(1.0, 1.01, 1.02)
        right.add(2.0, 2.03, 2.05)
        left.reset()
        self.assertEqual(left.text(), 'FPS: —  |  В приложении: —  |  Чтение: —')
        self.assertEqual(right.text(), 'FPS: —  |  В приложении: 20.0 мс  |  Чтение: 30.0 мс')

    def test_equal_timestamps_do_not_divide_by_zero(self):
        stats = FrameStats()
        stats.add(1.0, 1.0, 1.0)
        stats.add(1.0, 1.0, 1.0)
        self.assertIn('FPS: —', stats.text())


if __name__ == '__main__':
    unittest.main()
