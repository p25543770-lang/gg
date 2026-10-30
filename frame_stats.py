"""Rolling preview frame rate and separate timing statistics."""
from collections import deque


class FrameStats:
    def __init__(self, window=30):
        self.timestamps = deque(maxlen=window)
        self.read_waits = deque(maxlen=window)
        self.in_app = deque(maxlen=window)

    def reset(self):
        self.timestamps.clear()
        self.read_waits.clear()
        self.in_app.clear()

    def add(self, read_start, acquired, displayed):
        """Record timestamps from before read, after read, and UI submission."""
        self.timestamps.append(displayed)
        self.read_waits.append(max(0, acquired - read_start))
        self.in_app.append(max(0, displayed - acquired))

    def text(self):
        if not self.timestamps:
            return 'FPS: —  |  В приложении: —  |  Чтение: —'
        fps = '—'
        if len(self.timestamps) > 1:
            elapsed = self.timestamps[-1] - self.timestamps[0]
            if elapsed > 0:
                fps = f'{(len(self.timestamps) - 1) / elapsed:.1f}'
        app_ms = sum(self.in_app) * 1000 / len(self.in_app)
        read_ms = sum(self.read_waits) * 1000 / len(self.read_waits)
        return f'FPS: {fps}  |  В приложении: {app_ms:.1f} мс  |  Чтение: {read_ms:.1f} мс'
