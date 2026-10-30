"""Rolling preview frame rate and capture/processing time statistics."""
from collections import deque


class FrameStats:
    def __init__(self, window=30):
        self.timestamps = deque(maxlen=window)
        self.durations = deque(maxlen=window)

    def reset(self):
        self.timestamps.clear()
        self.durations.clear()

    def add(self, start, end):
        """Record a successfully prepared frame, using monotonic seconds."""
        self.timestamps.append(end)
        self.durations.append(max(0, end - start))

    def text(self):
        if not self.durations:
            return 'FPS: —  |  Получение + обработка: —'
        fps = '—'
        if len(self.timestamps) > 1:
            elapsed = self.timestamps[-1] - self.timestamps[0]
            if elapsed > 0:
                fps = f'{(len(self.timestamps) - 1) / elapsed:.1f}'
        ms = sum(self.durations) * 1000 / len(self.durations)
        return f'FPS: {fps}  |  Получение + обработка: {ms:.1f} мс'
