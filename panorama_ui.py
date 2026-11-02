"""Russian UI for stitching both cameras into one calibrated live panorama."""
from collections import deque
import tkinter as tk
from tkinter import filedialog, ttk

import cv2
from PIL import Image, ImageTk

from panorama import PanoramaError, PanoramaSettings, load_panorama, save_panorama
from panorama_worker import PanoramaWorker


class PanoramaPanel(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, padding=10)
        self.app = app
        self.worker = PanoramaWorker()
        self.timer = None
        self.photo = None
        self.full_photo = None
        self.full_window = None
        self.full_label = None
        self.last_result = None
        self.frame_times = deque(maxlen=30)
        self._notice_id = -1
        self._calibration = None
        self._busy = None
        self._active = False

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        controls = ttk.Frame(self)
        controls.grid(row=0, column=0, sticky='ew', pady=(0, 8))
        ttk.Label(controls, text='Общий вид:').pack(side='left')
        self.view = tk.StringVar(value='Панорама')
        self.view_box = ttk.Combobox(controls, textvariable=self.view, width=12, state='readonly',
                                     values=('Рядом', 'Панорама'))
        self.view_box.pack(side='left', padx=(5, 10))
        self.view_box.bind('<<ComboboxSelected>>', lambda _event: self._apply_settings())
        ttk.Label(controls, text='Источник кадров:').pack(side='left')
        self.source = tk.StringVar(value='Чистые кадры')
        self.source_box = ttk.Combobox(controls, textvariable=self.source, width=20, state='readonly',
                                       values=('Чистые кадры', 'Обработанные кадры'))
        self.source_box.pack(side='left', padx=(5, 10))
        self.source_box.bind('<<ComboboxSelected>>', lambda _event: self._apply_settings())
        self.calibrate_button = ttk.Button(controls, text='Настроить панораму', command=self._calibrate)
        self.calibrate_button.pack(side='left', padx=2)
        self.load_button = ttk.Button(controls, text='Загрузить…', command=self._load)
        self.load_button.pack(side='left', padx=2)
        self.save_button = ttk.Button(controls, text='Сохранить…', command=self._save, state='disabled')
        self.save_button.pack(side='left', padx=2)
        self.full_button = ttk.Button(controls, text='На весь экран', command=self._open_fullscreen)
        self.full_button.pack(side='right', padx=2)

        options = ttk.Frame(self)
        options.grid(row=2, column=0, sticky='ew', pady=(8, 0))
        self.swap = tk.BooleanVar(value=False)
        ttk.Checkbutton(options, text='Поменять камеры местами', variable=self.swap,
                        command=self._apply_settings).pack(side='left', padx=(0, 12))
        ttk.Label(options, text='Ширина кадра:').pack(side='left')
        self.max_width = tk.StringVar(value='1600')
        ttk.Combobox(options, textvariable=self.max_width, width=7, state='readonly',
                     values=('960', '1280', '1600', '1920', '2560')).pack(side='left', padx=(5, 12))
        ttk.Label(options, text='Допуск синхронизации, мс:').pack(side='left')
        self.sync_tolerance = tk.StringVar(value='66')
        ttk.Spinbox(options, from_=5, to=250, increment=1, textvariable=self.sync_tolerance, width=6).pack(side='left', padx=5)
        ttk.Button(options, text='Применить', command=self._apply_settings).pack(side='left', padx=5)

        preview_area = ttk.Frame(self, width=1000, height=520)
        preview_area.grid(row=1, column=0, sticky='nsew')
        preview_area.pack_propagate(False)
        self.preview = ttk.Label(preview_area, anchor='center', background='black', foreground='white',
                                 text='Откройте обе камеры на вкладке «Камеры и контуры».')
        self.preview.pack(fill='both', expand=True)
        preview_area.bind('<Configure>', self._on_resize)

        info = ttk.Frame(self)
        info.grid(row=3, column=0, sticky='ew', pady=(8, 0))
        self.status = tk.StringVar(value='Откройте обе камеры на вкладке «Камеры и контуры».')
        ttk.Label(info, textvariable=self.status, wraplength=1100).pack(fill='x')
        self.metrics = tk.StringVar(value='Общий FPS: —  |  Обработка: —  |  Δt кадров: —  |  Кадр: —')
        ttk.Label(info, textvariable=self.metrics).pack(fill='x', pady=(4, 0))
        self.calibration_info = tk.StringVar(value='Панорама не настроена: до настройки кадры показываются рядом.')
        ttk.Label(info, textvariable=self.calibration_info, wraplength=1100).pack(fill='x', pady=(4, 0))
        self.notice = tk.StringVar(value='')
        ttk.Label(info, textvariable=self.notice, wraplength=1100).pack(fill='x', pady=(4, 0))
        ttk.Label(info, wraplength=1100, text=(
            'Панорама: закрепите камеры неподвижно, обеспечьте общее поле зрения и нажмите «Настроить панораму» '
            'на текстурной сцене. «Чистые кадры» — как видит камера, «Обработанные кадры» — вместе со скелетом, '
            'контурами или режимом робота; настройку панорамы при переключении повторять не нужно. '
            'Близкие объекты на стыке могут двоиться из-за параллакса. '
            'Это живой просмотр; запись видео не выполняется.'
        )).pack(fill='x', pady=(4, 0))

        self.worker.start()
        self._apply_settings()
        self._poll()

    def update_streams(self, left, right):
        self.worker.set_streams(left, right)

    def set_active(self, active):
        self._active = bool(active)
        self.worker.set_enabled(self._active or self.full_window is not None)

    def close(self):
        self._close_fullscreen()
        if self.timer is not None:
            self.after_cancel(self.timer)
            self.timer = None
        self.worker.stop()
        self.worker.thread.join(timeout=1.0)

    def _settings_from_ui(self):
        return PanoramaSettings(
            view='panorama' if self.view.get() == 'Панорама' else 'side_by_side',
            max_width=int(self.max_width.get()),
            sync_tolerance_ms=float(self.sync_tolerance.get().replace(',', '.')),
            swap_cameras=bool(self.swap.get()),
            source='processed' if self.source.get() == 'Обработанные кадры' else 'raw',
        )

    def _apply_settings(self):
        try:
            self.worker.configure(self._settings_from_ui())
        except (ValueError, PanoramaError) as exc:
            self.notice.set(str(exc) if isinstance(exc, PanoramaError) else 'Проверьте параметры общего вида.')

    def _calibrate(self):
        self.view.set('Панорама')
        self._apply_settings()
        self.worker.request_calibration()

    def _load(self):
        path = filedialog.askopenfilename(
            parent=self, title='Настройка панорамы',
            filetypes=(('Настройка панорамы', '*.yml *.yaml *.json'), ('Все файлы', '*.*')),
        )
        if not path:
            return
        try:
            self.worker.set_calibration(load_panorama(path))
            self.view.set('Панорама')
            self._apply_settings()
            self.notice.set('Настройка панорамы загружена.')
        except (OSError, PanoramaError, cv2.error) as exc:
            self.notice.set(f'Не удалось загрузить настройку: {exc}')

    def _save(self):
        calibration = self.worker.poll().calibration
        if calibration is None:
            return
        path = filedialog.asksaveasfilename(
            parent=self, title='Сохранить настройку панорамы', defaultextension='.yml',
            filetypes=(('YAML', '*.yml'), ('JSON', '*.json')),
        )
        if not path:
            return
        try:
            save_panorama(calibration, path)
            self.notice.set('Настройка панорамы сохранена.')
        except (OSError, PanoramaError, cv2.error) as exc:
            self.notice.set(f'Не удалось сохранить настройку: {exc}')

    def _open_fullscreen(self):
        if self.full_window is not None:
            self.full_window.lift()
            return
        window = tk.Toplevel(self)
        window.title('Склейка камер — панорама')
        window.configure(background='black')
        window.attributes('-fullscreen', True)
        window.bind('<Escape>', lambda _event: self._close_fullscreen())
        window.protocol('WM_DELETE_WINDOW', self._close_fullscreen)
        label = tk.Label(window, background='black', foreground='white', anchor='center',
                         text='Ожидание кадров…')
        label.pack(fill='both', expand=True)
        hint = tk.Label(window, background='black', foreground='#cccccc', text='Esc — выйти из полноэкранного режима')
        hint.pack(fill='x')
        self.full_window = window
        self.full_label = label
        self.worker.set_enabled(True)
        if self.last_result is not None:
            self._render(self.last_result)

    def _close_fullscreen(self):
        if self.full_window is not None:
            self.full_window.destroy()
            self.full_window = None
            self.full_label = None
            self.full_photo = None
        self.worker.set_enabled(self._active)

    def _describe_calibration(self, calibration):
        if calibration is None:
            return 'Панорама не настроена. Режим «Рядом» работает без настройки.'
        lw, lh = calibration.left_size
        rw, rh = calibration.right_size
        text = f'Панорама настроена для кадров {lw}×{lh} и {rw}×{rh}.'
        if calibration.matches:
            text += f' Совпадения: {calibration.inliers}/{calibration.matches}.'
        if calibration.rms_px is not None:
            text += f' Ошибка: {calibration.rms_px:.2f} px.'
        return text

    def _poll(self):
        self.timer = None
        state = self.worker.poll()
        self.status.set(state.status)
        if state.notice_id != self._notice_id:
            self._notice_id = state.notice_id
            self.notice.set(state.notice)
        if state.busy != self._busy:
            self._busy = state.busy
            busy_state = 'disabled' if state.busy else 'normal'
            for widget in (self.calibrate_button, self.load_button):
                widget.configure(state=busy_state)
            if state.busy:
                self.notice.set('Настройка панорамы… не двигайте камеры.')
        if state.calibration is not self._calibration:
            self._calibration = state.calibration
            self.calibration_info.set(self._describe_calibration(state.calibration))
            self.save_button.configure(state='normal' if state.calibration else 'disabled')
            if state.calibration is not None:
                self.view.set('Панорама')
        if state.result is not self.last_result:
            self.last_result = state.result
            if state.result is None:
                self.frame_times.clear()
                self.photo = None
                self.preview.configure(image='', text=state.status)
                if self.full_label is not None:
                    self.full_label.configure(image='', text=state.status)
                self.metrics.set('Общий FPS: —  |  Обработка: —  |  Δt кадров: —  |  Кадр: —')
            else:
                result = state.result
                self.frame_times.append(result.created_at)
                fps = '—'
                if len(self.frame_times) > 1:
                    elapsed = self.frame_times[-1] - self.frame_times[0]
                    if elapsed > 0:
                        fps = f'{(len(self.frame_times) - 1) / elapsed:.1f}'
                self.metrics.set(f'Общий FPS: {fps}  |  Обработка: {result.elapsed_ms:.1f} мс  |  '
                                 f'Δt кадров: {result.sync_skew_ms:.1f} мс  |  Кадр: {result.image_size[0]}×{result.image_size[1]}')
                self._render(result)
        self.timer = self.after(33, self._poll)

    def _on_resize(self, _event=None):
        if self.last_result is not None:
            self._render(self.last_result)

    def _fit(self, image, width, height):
        width, height = max(1, width), max(1, height)
        scale = min(width / image.width, height / image.height)
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        return image.resize(size, Image.Resampling.BILINEAR)

    def _render(self, result):
        image = Image.fromarray(result.color_rgb)
        fitted = self._fit(image, self.preview.winfo_width(), self.preview.winfo_height())
        self.photo = ImageTk.PhotoImage(fitted)
        self.preview.configure(image=self.photo, text='')
        if self.full_window is not None and self.full_label is not None:
            full = self._fit(image, self.full_window.winfo_width(), self.full_window.winfo_height() - 24)
            self.full_photo = ImageTk.PhotoImage(full)
            self.full_label.configure(image=self.full_photo, text='')
