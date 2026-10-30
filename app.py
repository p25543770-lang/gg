"""Two live camera previews with purple object edges."""
import sys
import time
import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from camera_stream import CameraStream
from frame_stats import FrameStats


class CameraPanel(ttk.LabelFrame):
    def __init__(self, parent, app, title, default_index):
        super().__init__(parent, text=title, padding=8)
        self.app = app
        self.stream = None
        self.open_index = None
        self.timer = None
        self.photo = None
        self.frame_stats = FrameStats()

        controls = ttk.Frame(self)
        controls.pack(fill='x')
        ttk.Label(controls, text='Номер камеры:').pack(side='left')
        self.index = tk.StringVar(value=str(default_index))
        ttk.Spinbox(controls, from_=0, to=20, width=4, textvariable=self.index).pack(side='left', padx=5)
        ttk.Button(controls, text='Открыть', command=self.open_camera).pack(side='left', padx=2)
        ttk.Button(controls, text='Закрыть', command=self.release).pack(side='left', padx=2)

        # Do not let the current image force the two panels to grow indefinitely.
        preview_area = ttk.Frame(self, width=320, height=240)
        preview_area.pack(fill='both', expand=True, pady=(8, 0))
        preview_area.pack_propagate(False)
        self.preview = ttk.Label(preview_area, anchor='center', text='Нажмите «Открыть»')
        self.preview.pack(fill='both', expand=True)
        self.metrics = tk.StringVar(value=self.frame_stats.text())
        ttk.Label(self, textvariable=self.metrics).pack(fill='x', pady=(8, 0))
        self.status = tk.StringVar(value='Выберите номер камеры.')
        ttk.Label(self, textvariable=self.status, wraplength=450).pack(fill='x', pady=(8, 0))

    def open_camera(self):
        try:
            index = int(self.index.get())
            if not 0 <= index <= 20:
                raise ValueError
        except ValueError:
            self.status.set('Введите номер камеры от 0 до 20.')
            return
        if self.app.camera_busy(index):
            self.status.set('Эта камера уже открыта или ещё закрывается. Выберите другой номер.')
            return
        self.release()
        self.open_index = index
        self.stream = CameraStream(index)
        self.app.streams.append(self.stream)
        self.stream.start()
        self.status.set(f'Открытие камеры {index}…')
        self.update_frame()

    def update_frame(self):
        self.timer = None
        if self.stream is None:
            return
        self.stream.configure(self.app.sensitivity.get(), self.preview.winfo_width(), self.preview.winfo_height())
        state, result = self.stream.poll()
        if state not in ('open', 'opening'):
            self.release()
            self.status.set(state)
            return
        if state == 'open':
            self.status.set(f'Камера {self.open_index} открыта.')
        if result is not None:
            rgb, started = result
            self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.preview.configure(image=self.photo, text='')
            self.frame_stats.add(started, time.perf_counter())
            self.metrics.set(self.frame_stats.text())
        # Short polling interval; frame production is capped separately at 30 FPS.
        self.timer = self.app.root.after(10, self.update_frame)

    def release(self):
        if self.timer is not None:
            self.app.root.after_cancel(self.timer)
            self.timer = None
        if self.stream is not None:
            self.stream.stop()
            self.stream = None
        self.open_index = None
        self.preview.configure(image='', text='Камера не открыта')
        self.photo = None
        self.frame_stats.reset()
        self.metrics.set(self.frame_stats.text())
        self.status.set('Камера закрыта.')


class CameraApp:
    def __init__(self, root):
        self.root = root
        self.streams = []
        root.title('Камеры — фиолетовые контуры')
        root.protocol('WM_DELETE_WINDOW', self.close)

        controls = ttk.Frame(root, padding=10)
        controls.pack(fill='x')
        ttk.Label(controls, text='Чувствительность контуров (обе камеры):').pack(side='left')
        self.sensitivity = tk.IntVar(value=50)
        ttk.Scale(controls, from_=1, to=100, variable=self.sensitivity, orient='horizontal').pack(side='left', fill='x', expand=True, padx=5)

        views = ttk.Frame(root, padding=(10, 0, 10, 10))
        views.pack(fill='both', expand=True)
        views.columnconfigure((0, 1), weight=1, uniform='cameras')
        views.rowconfigure(0, weight=1)
        self.panels = [CameraPanel(views, self, 'Левая камера', 0),
                       CameraPanel(views, self, 'Правая камера', 1)]
        for column, panel in enumerate(self.panels):
            panel.grid(row=0, column=column, sticky='nsew', padx=(0, 5) if column == 0 else (5, 0))

    def camera_busy(self, index):
        # A stopped capture may still be blocked in the driver; don't reopen it
        # until its owner thread has released the device.
        self.streams = [stream for stream in self.streams if stream.alive()]
        return any(stream.index == index for stream in self.streams)

    def close(self):
        for panel in self.panels:
            panel.release()
        self.root.destroy()


def self_test():
    """Exercise two asynchronous packaged previews without physical cameras."""
    import numpy as np

    cameras = {}

    class FakeCamera:
        def __init__(self, index, backend):
            self.index = index
            self.released = False
            self.fail_read = False
            cameras[index] = self

        def isOpened(self):
            return True

        def set(self, prop, value):
            return True

        def read(self):
            time.sleep(0.02)
            if self.fail_read:
                return False, None
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            frame[25:75, 25:75] = 255
            return True, frame

        def release(self):
            self.released = True

    def wait_until(predicate):
        deadline = time.perf_counter() + 5
        while not predicate():
            root.update()
            if time.perf_counter() >= deadline:
                raise AssertionError('Timed out waiting for camera preview')
            time.sleep(0.01)

    root = tk.Tk()
    root.withdraw()
    app = CameraApp(root)
    original_capture = cv2.VideoCapture
    cv2.VideoCapture = FakeCamera
    streams = []
    try:
        left, right = app.panels
        left.open_camera()
        right.open_camera()
        streams = app.streams[:]
        wait_until(lambda: len(left.frame_stats.timestamps) >= 2 and len(right.frame_stats.timestamps) >= 2)
        assert left.photo is not None and right.photo is not None
        assert 'FPS: —' not in left.metrics.get() and 'FPS: —' not in right.metrics.get()
        right.index.set('0')
        right.open_camera()
        assert right.stream is streams[1] and left.stream is streams[0]
        cameras[0].fail_read = True
        wait_until(lambda: left.stream is None)
        assert not left.frame_stats.timestamps
        assert right.stream is streams[1]
        count = len(right.frame_stats.timestamps)
        wait_until(lambda: len(right.frame_stats.timestamps) > count)
    finally:
        app.close()
        for stream in streams:
            stream.capture_thread.join(2)
            stream.process_thread.join(2)
        cv2.VideoCapture = original_capture
    assert cameras[0].released and cameras[1].released


if __name__ == '__main__':
    if '--self-test' in sys.argv:
        try:
            self_test()
        except Exception:
            import traceback
            with open('self-test-error.txt', 'w', encoding='utf-8') as error_file:
                traceback.print_exc(file=error_file)
            sys.exit(1)
        sys.exit(0)
    root = tk.Tk()
    root.geometry('1200x650')
    CameraApp(root)
    root.mainloop()
