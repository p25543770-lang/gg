"""Two live camera previews with purple object edges."""
import sys
import time
import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from edges import highlight_edges
from frame_stats import FrameStats


class CameraPanel(ttk.LabelFrame):
    def __init__(self, parent, app, title, default_index):
        super().__init__(parent, text=title, padding=8)
        self.app = app
        self.camera = None
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
        if any(panel is not self and panel.open_index == index for panel in self.app.panels):
            self.status.set('Эта камера уже открыта в соседней панели. Выберите другой номер.')
            return
        self.release()
        # DirectShow avoids slow Media Foundation device initialization on Windows.
        self.camera = cv2.VideoCapture(index, cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_ANY)
        if not self.camera.isOpened():
            self.release()
            self.status.set('Не удалось открыть камеру. Попробуйте другой номер.')
            return
        self.open_index = index
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.status.set(f'Камера {index} открыта.')
        self.update_frame()

    def update_frame(self):
        self.timer = None
        if self.camera is None:
            return
        started = time.perf_counter()
        ok, frame = self.camera.read()
        if not ok:
            self.release()
            self.status.set('Не удалось получить кадр. Проверьте подключение камеры.')
            return
        h, w = frame.shape[:2]
        highlight_edges(frame, self.app.sensitivity.get())
        # Fit each image in its own panel without changing the aspect ratio.
        available_w = max(self.preview.winfo_width(), 1)
        available_h = max(self.preview.winfo_height(), 1)
        scale = min(available_w / w, available_h / h, 1)
        frame = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.preview.configure(image=self.photo, text='')
        self.frame_stats.add(started, time.perf_counter())
        self.metrics.set(self.frame_stats.text())
        self.timer = self.app.root.after(30, self.update_frame)

    def release(self):
        if self.timer is not None:
            self.app.root.after_cancel(self.timer)
            self.timer = None
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.open_index = None
        self.preview.configure(image='', text='Камера не открыта')
        self.photo = None
        self.frame_stats.reset()
        self.metrics.set(self.frame_stats.text())
        self.status.set('Камера закрыта.')


class CameraApp:
    def __init__(self, root):
        self.root = root
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

    def close(self):
        for panel in self.panels:
            panel.release()
        self.root.destroy()


def self_test():
    """Exercise both packaged previews independently, without physical cameras."""
    import numpy as np

    class FakeCamera:
        def __init__(self, index, backend):
            self.index = index
            self.released = False
            self.fail_read = False
            self.frame = None

        def isOpened(self):
            return True

        def set(self, prop, value):
            return True

        def read(self):
            if self.fail_read:
                return False, None
            self.frame = np.zeros((100, 100, 3), dtype=np.uint8)
            self.frame[25:75, 25:75] = 255
            return True, self.frame

        def release(self):
            self.released = True

    root = tk.Tk()
    root.withdraw()
    app = CameraApp(root)
    original_capture = cv2.VideoCapture
    cv2.VideoCapture = FakeCamera
    try:
        left, right = app.panels
        left.open_camera()
        right.open_camera()
        left_camera, right_camera = left.camera, right.camera
        assert left_camera.index == 0 and right_camera.index == 1
        for panel, camera in ((left, left_camera), (right, right_camera)):
            assert tuple(camera.frame[0, 0]) == (0, 0, 0)
            assert tuple(camera.frame[50, 50]) == (255, 255, 255)
            assert np.any(np.all(camera.frame[20:30, 30:70] == (180, 0, 140), axis=2))
            assert panel.photo is not None and panel.timer is not None
            assert len(panel.frame_stats.timestamps) == 1
            assert 'FPS: —' in panel.metrics.get() and 'мс' in panel.metrics.get()
        root.after_cancel(right.timer)
        right.timer = None
        right.update_frame()
        assert len(right.frame_stats.timestamps) == 2
        assert right.metrics.get() == right.frame_stats.text()
        assert len(left.frame_stats.timestamps) == 1
        # Refuse to open the same camera twice without disturbing either stream.
        right.index.set('0')
        right.open_camera()
        assert left.camera is left_camera and right.camera is right_camera
        # Failure of one stream should not release the other.
        root.after_cancel(left.timer)
        left.timer = None
        left_camera.fail_read = True
        left.update_frame()
        assert left_camera.released and left.camera is None
        assert not left.frame_stats.timestamps and left.metrics.get() == left.frame_stats.text()
        assert right.camera is right_camera and right.timer is not None
        assert len(right.frame_stats.timestamps) == 2
    finally:
        cv2.VideoCapture = original_capture
        app.close()
    assert right_camera.released


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
