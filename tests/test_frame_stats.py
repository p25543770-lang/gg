"""Frame counters can be tested with deterministic timestamps."""
import unittest

from frame_stats import FrameStats


class FrameStatsTests(unittest.TestCase):
    def test_counter_and_rolling_window(self):
        stats = FrameStats(window=3)
        self.assertEqual(stats.text(), 'FPS: —  |  Чтение + обработка + ожидание: —')
        stats.add(0.98, 1.0)
        self.assertEqual(stats.text(), 'FPS: —  |  Чтение + обработка + ожидание: 20.0 мс')
        stats.add(1.08, 1.1)
        stats.add(1.17, 1.2)
        self.assertEqual(stats.text(), 'FPS: 10.0  |  Чтение + обработка + ожидание: 23.3 мс')
        stats.add(1.26, 1.3)
        self.assertEqual(stats.text(), 'FPS: 10.0  |  Чтение + обработка + ожидание: 30.0 мс')

    def test_instances_are_independent_and_reset(self):
        left, right = FrameStats(), FrameStats()
        left.add(1.0, 1.02)
        right.add(2.0, 2.05)
        left.reset()
        self.assertEqual(left.text(), 'FPS: —  |  Чтение + обработка + ожидание: —')
        self.assertEqual(right.text(), 'FPS: —  |  Чтение + обработка + ожидание: 50.0 мс')

    def test_equal_timestamps_do_not_divide_by_zero(self):
        stats = FrameStats()
        stats.add(1.0, 1.0)
        stats.add(1.0, 1.0)
        self.assertIn('FPS: —', stats.text())


if __name__ == '__main__':
    unittest.main()
