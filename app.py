"""Live camera preview with purple corners."""
import sys
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk


class CameraApp:
    def __init__(self, root):
        self.root = root
        root.title('Камера — фиолетовые углы')
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.camera = None
        self.timer = None
        self.photo = None

        controls = ttk.Frame(root, padding=10)
        controls.pack(fill='x')
        ttk.Label(controls, text='Номер камеры:').pack(side='left')
        self.index = tk.StringVar(value='0')
        ttk.Spinbox(controls, from_=0, to=20, width=4, textvariable=self.index).pack(side='left', padx=5)
        ttk.Button(controls, text='Открыть камеру', command=self.open_camera).pack(side='left', padx=5)
        ttk.Label(controls, text='Размер углов:').pack(side='left', padx=(15, 0))
        self.size = tk.IntVar(value=15)
        ttk.Scale(controls, from_=5, to=35, variable=self.size, orient='horizontal').pack(side='left', fill='x', expand=True, padx=5)
        self.preview = ttk.Label(root, anchor='center', text='Нажмите «Открыть камеру»')
        self.preview.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        self.status = tk.StringVar(value='Подключите Logitech C930e и выберите её номер.')
        ttk.Label(root, textvariable=self.status, padding=(10, 0, 10, 10)).pack(fill='x')

    def open_camera(self):
        try:
            index = int(self.index.get())
            if index < 0 or index > 20:
                raise ValueError
        except ValueError:
            messagebox.showerror('Ошибка', 'Введите номер камеры от 0 до 20.')
            return
        self.release()
        # DirectShow avoids slow Media Foundation device initialization on Windows.
        self.camera = cv2.VideoCapture(index, cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_ANY)
        if not self.camera.isOpened():
            self.release()
            self.status.set('Не удалось открыть камеру. Попробуйте другой номер.')
            return
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.status.set('Камера открыта. Для другой камеры измените номер.')
        self.update_frame()

    def update_frame(self):
        self.timer = None
        if self.camera is None:
            return
        ok, frame = self.camera.read()
        if not ok:
            self.release()
            self.status.set('Не удалось получить кадр. Проверьте подключение камеры.')
            return
        h, w = frame.shape[:2]
        dx, dy = int(w * self.size.get() / 100), int(h * self.size.get() / 100)
        purple = (180, 0, 140)  # BGR
        import numpy as np
        for points in ([(0, 0), (dx, 0), (0, dy)],
                       [(w-1, 0), (w-1-dx, 0), (w-1, dy)],
                       [(0, h-1), (dx, h-1), (0, h-1-dy)],
                       [(w-1, h-1), (w-1-dx, h-1), (w-1, h-1-dy)]):
            cv2.fillPoly(frame, [np.array(points, dtype=np.int32)], purple)
        # Keep the preview within the window while retaining the aspect ratio.
        available_w = max(self.preview.winfo_width(), 640)
        available_h = max(self.preview.winfo_height(), 360)
        scale = min(available_w / w, available_h / h, 1)
        frame = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.preview.configure(image=self.photo, text='')
        self.timer = self.root.after(30, self.update_frame)

    def release(self):
        if self.timer is not None:
            self.root.after_cancel(self.timer)
            self.timer = None
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.preview.configure(image='', text='Камера не открыта')
        self.photo = None

    def close(self):
        self.release()
        self.root.destroy()


def self_test():
    """Exercise the packaged UI and image pipeline without a physical camera."""
    import numpy as np

    class FakeCamera:
        def __init__(self):
            self.frame = np.zeros((100, 100, 3), dtype=np.uint8)
            self.released = False

        def read(self):
            return True, self.frame

        def release(self):
            self.released = True

    root = tk.Tk()
    root.withdraw()
    app = CameraApp(root)
    camera = FakeCamera()
    app.camera = camera
    try:
        app.update_frame()
        assert tuple(camera.frame[0, 0]) == (180, 0, 140)
        assert tuple(camera.frame[50, 50]) == (0, 0, 0)
        assert app.photo is not None and app.timer is not None
    finally:
        app.close()
    assert camera.released


if __name__ == '__main__':
    if '--self-test' in sys.argv:
        self_test()
        sys.exit(0)
    root = tk.Tk()
    root.geometry('1000x650')
    CameraApp(root)
    root.mainloop()
