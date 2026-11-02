"""Two live camera previews with purple object edges."""
import sys
import time
import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from camera_stream import CameraStream
from frame_stats import FrameStats
from stereo_ui import StereoPanel

# Avoid oversubscribing CPU cores while two cameras process frames concurrently.
cv2.setNumThreads(1)


class CameraPanel(ttk.LabelFrame):
    def __init__(self, parent, app, title, default_index):
        super().__init__(parent, text=title, padding=8)
        self.app = app
        self.stream = None
        self.open_index = None
        self.requested_resolution = None
        self.actual_resolution = None
        self.timer = None
        self.reopen_timer = None
        self.photo = None
        self.frame_stats = FrameStats()

        controls = ttk.Frame(self)
        controls.pack(fill='x')
        ttk.Label(controls, text='Номер камеры:').pack(side='left')
        self.index = tk.StringVar(value=str(default_index))
        ttk.Spinbox(controls, from_=0, to=20, width=4, textvariable=self.index).pack(side='left', padx=5)
        ttk.Button(controls, text='Открыть', command=self.open_camera).pack(side='left', padx=2)
        ttk.Button(controls, text='Закрыть', command=self.release).pack(side='left', padx=2)
        resolution_controls = ttk.Frame(self)
        resolution_controls.pack(fill='x', pady=(5, 0))
        ttk.Label(resolution_controls, text='Разрешение:').pack(side='left')
        self.resolution = tk.StringVar(value='640×480')
        ttk.Combobox(resolution_controls, textvariable=self.resolution, width=12, state='readonly',
                     values=('320×240', '640×480', '1280×720', '1920×1080')).pack(side='left', padx=5)
        ttk.Button(resolution_controls, text='Применить', command=self.apply_resolution).pack(side='left', padx=5)

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
        if self.reopen_timer is not None:
            self.app.root.after_cancel(self.reopen_timer)
            self.reopen_timer = None
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
        self.requested_resolution = tuple(map(int, self.resolution.get().split('×')))
        self.stream = CameraStream(index, resolution=self.requested_resolution)
        self.app.streams.append(self.stream)
        self.stream.start()
        self.app.sync_stereo()
        self.status.set(f'Открытие камеры {index}…')
        self.update_frame()

    def update_frame(self):
        self.timer = None
        if self.stream is None:
            return
        self.stream.configure(self.app.sensitivity.get(), self.preview.winfo_width(), self.preview.winfo_height(),
                              mode=self.app.vision_mode())
        state, result, notice = self.stream.poll()
        if state not in ('open', 'opening'):
            self.release()
            self.status.set(state)
            return
        if result is not None:
            rgb, started, acquired, self.actual_resolution = result
            self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.preview.configure(image=self.photo, text='')
            self.frame_stats.add(started, acquired, time.perf_counter())
            self.metrics.set(self.frame_stats.text())
        if state == 'open':
            status = f'Камера {self.open_index} открыта.'
            if self.actual_resolution:
                w, h = self.actual_resolution
                status += f' Кадр: {w}×{h}.'
                if self.actual_resolution != self.requested_resolution:
                    status += ' Камера выбрала другое разрешение.'
            if notice:
                status += f' {notice}'
            if self.status.get() != status:
                self.status.set(status)
        # Keep the UI responsive and pick up the most recent processed frame.
        self.timer = self.app.root.after(2, self.update_frame)

    def apply_resolution(self):
        if self.stream is None:
            self.status.set('Выберите разрешение и нажмите «Открыть».')
            return
        index = self.open_index
        self.release()
        self.index.set(str(index))
        self.status.set('Перезапуск камеры с новым разрешением…')
        self._reopen_when_released(index, time.perf_counter() + 5)

    def _reopen_when_released(self, index, deadline):
        self.reopen_timer = None
        if not self.app.camera_busy(index):
            self.index.set(str(index))
            self.open_camera()
        elif time.perf_counter() >= deadline:
            self.status.set('Камера ещё закрывается. Попробуйте открыть её снова.')
        else:
            self.reopen_timer = self.app.root.after(25, self._reopen_when_released, index, deadline)

    def release(self):
        if self.reopen_timer is not None:
            self.app.root.after_cancel(self.reopen_timer)
            self.reopen_timer = None
        if self.timer is not None:
            self.app.root.after_cancel(self.timer)
            self.timer = None
        if self.stream is not None:
            self.stream.stop()
            self.stream = None
        self.open_index = None
        self.requested_resolution = None
        self.actual_resolution = None
        self.preview.configure(image='', text='Камера не открыта')
        self.photo = None
        self.frame_stats.reset()
        self.metrics.set(self.frame_stats.text())
        self.status.set('Камера закрыта.')
        self.app.sync_stereo()


class CameraApp:
    def __init__(self, root):
        self.root = root
        self.streams = []
        root.title('Камеры — фиолетовые контуры и стерео')
        root.protocol('WM_DELETE_WINDOW', self.close)

        controls = ttk.Frame(root, padding=10)
        controls.pack(fill='x')
        ttk.Label(controls, text='Режим:').pack(side='left')
        self.mode_label = tk.StringVar(value='Фиолетовые контуры')
        self.mode_box = ttk.Combobox(controls, textvariable=self.mode_label, width=24, state='readonly',
                                     values=('Фиолетовые контуры', 'Дорога / препятствия'))
        self.mode_box.pack(side='left', padx=(5, 12))
        self.mode_box.bind('<<ComboboxSelected>>', lambda _event: self.on_mode_changed())
        ttk.Label(controls, text='Чувствительность (обе камеры):').pack(side='left')
        self.sensitivity = tk.IntVar(value=50)
        ttk.Scale(controls, from_=1, to=100, variable=self.sensitivity, orient='horizontal').pack(side='left', fill='x', expand=True, padx=5)
        self.robot_hint = ttk.Label(root, padding=(10, 0, 10, 6),
                                    text='Робот: зелёным — предполагаемый пол/дорога, красным — препятствия. Это визуальная подсказка, не аварийная система.')
        self.robot_hint.pack(fill='x')
        self.robot_hint.pack_forget()

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True)
        cameras_tab = ttk.Frame(self.notebook)
        self.notebook.add(cameras_tab, text='Камеры и контуры')
        views = ttk.Frame(cameras_tab, padding=(10, 0, 10, 10))
        views.pack(fill='both', expand=True)
        views.columnconfigure((0, 1), weight=1, uniform='cameras')
        views.rowconfigure(0, weight=1)
        self.panels = [CameraPanel(views, self, 'Левая камера', 0),
                       CameraPanel(views, self, 'Правая камера', 1)]
        for column, panel in enumerate(self.panels):
            panel.grid(row=0, column=column, sticky='nsew', padx=(0, 5) if column == 0 else (5, 0))
        self.stereo = StereoPanel(self.notebook, self)
        self.notebook.add(self.stereo, text='Стерео / глубина')
        self.sync_stereo()

    def vision_mode(self):
        return 'robot' if self.mode_label.get() == 'Дорога / препятствия' else 'edges'

    def on_mode_changed(self):
        if self.vision_mode() == 'robot':
            self.robot_hint.pack(fill='x', before=self.notebook)
        else:
            self.robot_hint.pack_forget()
        for panel in getattr(self, 'panels', []):
            if panel.stream is not None:
                panel.stream.configure(self.sensitivity.get(), panel.preview.winfo_width(), panel.preview.winfo_height(),
                                       mode=self.vision_mode())

    def sync_stereo(self):
        if hasattr(self, 'stereo') and hasattr(self, 'panels'):
            self.stereo.update_streams(self.panels[0].stream, self.panels[1].stream)

    def camera_busy(self, index):
        # A stopped capture may still be blocked in the driver; don't reopen it
        # until its owner thread has released the device.
        self.streams = [stream for stream in self.streams if stream.alive()]
        return any(stream.index == index for stream in self.streams)

    def close(self):
        for panel in self.panels:
            panel.release()
        self.stereo.close()
        # Give capture threads a bounded chance to restore camera settings.
        for stream in self.streams:
            stream.capture_thread.join(timeout=0.5)
        self.root.destroy()


def self_test():
    """Exercise two asynchronous packaged previews and stereo without physical cameras."""
    import numpy as np

    from stereo import StereoCalibration, StereoSettings

    cameras = {}
    texture = np.random.default_rng(42).integers(0, 256, (100, 140), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (3, 3), 0)

    class FakeCamera:
        def __init__(self, index, backend):
            self.index = index
            self.released = False
            self.fail_read = False
            self.properties = {}
            cameras[index] = self

        def isOpened(self):
            return True

        def set(self, prop, value):
            self.properties[prop] = value
            return True

        def get(self, prop):
            return 0.75 if prop == cv2.CAP_PROP_AUTO_EXPOSURE else -5

        def read(self):
            time.sleep(0.02)
            if self.fail_read:
                return False, None
            # A textured synthetic stereo pair: the right image is shifted by
            # 8 pixels, so a calibrated matcher should recover a finite depth.
            offset = 20 + (8 if self.index == 1 else 0)
            frame = cv2.cvtColor(texture[:, offset:offset + 100], cv2.COLOR_GRAY2BGR)
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
        right.resolution.set('1280×720')
        left.open_camera()
        right.open_camera()
        streams = app.streams[:]
        wait_until(lambda: len(left.frame_stats.timestamps) >= 2 and len(right.frame_stats.timestamps) >= 2)
        assert left.photo is not None and right.photo is not None
        assert cameras[0].properties[cv2.CAP_PROP_FRAME_WIDTH] == 640
        assert cameras[0].properties[cv2.CAP_PROP_FRAME_HEIGHT] == 480
        assert cameras[1].properties[cv2.CAP_PROP_FRAME_WIDTH] == 1280
        assert cameras[1].properties[cv2.CAP_PROP_FRAME_HEIGHT] == 720
        assert left.actual_resolution == right.actual_resolution == (100, 100)
        assert '100×100' in right.status.get() and 'другое разрешение' in right.status.get()
        assert 'FPS: —' not in left.metrics.get() and 'FPS: —' not in right.metrics.get()
        old_right_camera = cameras[1]
        right.resolution.set('1920×1080')
        right.apply_resolution()
        wait_until(lambda: right.stream is not None and cameras.get(1) is not old_right_camera
                   and len(right.frame_stats.timestamps) >= 2)
        assert old_right_camera.released
        assert cameras[1].properties[cv2.CAP_PROP_FRAME_WIDTH] == 1920
        assert cameras[1].properties[cv2.CAP_PROP_FRAME_HEIGHT] == 1080
        assert left.stream is streams[0]
        streams = app.streams[:]
        right.index.set('0')
        right.open_camera()
        assert right.stream is streams[1] and left.stream is streams[0]

        # Exercise the packaged stereo worker and tab with a synthetic disparity.
        app.notebook.select(app.stereo)
        k = np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]])
        calibration = StereoCalibration(k, np.zeros(5), k, np.zeros(5), np.eye(3),
                                        np.array([[-0.05], [0], [0]]), (100, 100))
        app.stereo.worker.set_calibration(calibration)
        app.stereo.worker.configure(StereoSettings(max_width=160, num_disparities=16,
                                                   sync_tolerance_ms=100,
                                                   min_depth_m=0.1, max_depth_m=5))
        app.stereo.worker.set_enabled(True)
        wait_until(lambda: app.stereo.worker.poll().result is not None)
        wait_until(lambda: app.stereo.photo is not None)
        stereo_result = app.stereo.worker.poll().result
        assert stereo_result.depth_m is not None
        median_depth = float(np.nanmedian(stereo_result.depth_m))
        assert 0.4 < median_depth < 0.9, median_depth
        assert stereo_result.baseline_m == 0.05
        assert 'Стерео FPS' in app.stereo.metrics.get()
        app.stereo.worker.set_enabled(False)

        # Exercise the robot visual-aid mode without changing stereo snapshots.
        app.mode_label.set('Дорога / препятствия')
        app.on_mode_changed()
        count = len(left.frame_stats.timestamps)
        wait_until(lambda: len(left.frame_stats.timestamps) > count + 2)
        assert left.photo is not None and right.photo is not None
        snapshot = left.stream.snapshot()
        assert snapshot is not None
        assert not np.any(np.all(snapshot[0] == (0, 180, 0), axis=2))
        app.mode_label.set('Фиолетовые контуры')
        app.on_mode_changed()

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
    root.geometry('1280x860')
    root.minsize(1100, 720)
    CameraApp(root)
    root.mainloop()
