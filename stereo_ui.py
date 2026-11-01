"""Tkinter controls and preview for the stereo worker."""
from collections import deque
import tkinter as tk
from tkinter import filedialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from stereo import StereoError, StereoSettings, load_calibration, save_calibration
from stereo_worker import StereoWorker


class StereoPanel(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, padding=10)
        self.app = app
        self.worker = StereoWorker()
        self.timer = None
        self.photo = None
        self.last_result = None
        self.display_size = (0, 0)
        self.display_offset = (0, 0)
        self.frame_times = deque(maxlen=30)
        self._notice_id = -1
        self._calibration = None
        self._busy = None
        self._status = None

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        preview_area = ttk.Frame(self, width=760, height=480)
        preview_area.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
        preview_area.pack_propagate(False)
        self.preview = ttk.Label(preview_area, anchor='center', text='Стерео выключено', background='black', foreground='white')
        self.preview.pack(fill='both', expand=True)
        preview_area.bind('<Configure>', self._on_resize)
        self.preview.bind('<Button-1>', self._measure)

        info = ttk.Frame(self)
        info.grid(row=1, column=0, sticky='ew', padx=(0, 10), pady=(8, 0))
        self.status = tk.StringVar(value='Откройте обе камеры и загрузите калибровку.')
        ttk.Label(info, textvariable=self.status, wraplength=740).pack(fill='x')
        self.metrics = tk.StringVar(value='Стерео FPS: —  |  Обработка: —  |  Δt кадров: —  |  Совпадения: —')
        ttk.Label(info, textvariable=self.metrics).pack(fill='x', pady=(4, 0))
        self.measurement = tk.StringVar(value='Нажмите на карту, чтобы измерить глубину в точке (медиана 9×9).')
        ttk.Label(info, textvariable=self.measurement).pack(fill='x', pady=(4, 0))
        self.legend = tk.StringVar(value='Карта глубины: красный — ближе, синий — дальше; чёрный — нет данных.')
        ttk.Label(info, textvariable=self.legend, wraplength=740).pack(fill='x', pady=(4, 0))

        controls = ttk.Frame(self)
        controls.grid(row=0, column=1, rowspan=2, sticky='ns')
        self.enable_button = ttk.Button(controls, text='Включить стерео', command=self._toggle)
        self.enable_button.pack(fill='x', pady=(0, 8))

        calibration_box = ttk.LabelFrame(controls, text='Калибровка stereo-пары', padding=8)
        calibration_box.pack(fill='x', pady=(0, 8))
        file_buttons = ttk.Frame(calibration_box)
        file_buttons.pack(fill='x')
        self.load_button = ttk.Button(file_buttons, text='Загрузить…', command=self._load)
        self.load_button.pack(side='left', expand=True, fill='x', padx=(0, 4))
        self.save_button = ttk.Button(file_buttons, text='Сохранить…', command=self._save, state='disabled')
        self.save_button.pack(side='left', expand=True, fill='x')
        unit_row = ttk.Frame(calibration_box)
        unit_row.pack(fill='x', pady=(6, 0))
        ttk.Label(unit_row, text='Единицы T при загрузке:').pack(side='left')
        self.load_unit = tk.StringVar(value='Из файла')
        ttk.Combobox(unit_row, textvariable=self.load_unit, state='readonly', width=14,
                     values=('Из файла', 'Метры', 'Сантиметры', 'Миллиметры', 'Неизвестны')).pack(side='left', padx=5)
        self.calibration_info = tk.StringVar(value='Калибровка не загружена.')
        ttk.Label(calibration_box, textvariable=self.calibration_info, wraplength=360).pack(fill='x', pady=(6, 0))

        board_box = ttk.LabelFrame(controls, text='Съёмка шахматной доски', padding=8)
        board_box.pack(fill='x', pady=(0, 8))
        ttk.Label(board_box, text='Внутренние углы (не клетки):').pack(anchor='w')
        board_row = ttk.Frame(board_box)
        board_row.pack(fill='x', pady=(2, 4))
        self.board_columns = tk.StringVar(value='9')
        self.board_rows = tk.StringVar(value='6')
        ttk.Spinbox(board_row, from_=2, to=30, width=4, textvariable=self.board_columns).pack(side='left')
        ttk.Label(board_row, text='×').pack(side='left', padx=4)
        ttk.Spinbox(board_row, from_=2, to=30, width=4, textvariable=self.board_rows).pack(side='left')
        ttk.Label(board_row, text='Сторона клетки, мм:').pack(side='left', padx=(12, 4))
        self.square_mm = tk.StringVar(value='25')
        ttk.Spinbox(board_row, from_=1, to=1000, increment=0.5, width=6, textvariable=self.square_mm).pack(side='left')
        capture_buttons = ttk.Frame(board_box)
        capture_buttons.pack(fill='x', pady=(4, 0))
        self.capture_button = ttk.Button(capture_buttons, text='Снять пару доски', command=self._capture)
        self.capture_button.pack(side='left', expand=True, fill='x', padx=(0, 4))
        self.calibrate_button = ttk.Button(capture_buttons, text='Калибровать', command=self._calibrate)
        self.calibrate_button.pack(side='left', expand=True, fill='x')
        self.clear_button = ttk.Button(board_box, text='Очистить пары', command=self.worker.clear_samples)
        self.clear_button.pack(fill='x', pady=(4, 0))
        self.sample_count = tk.StringVar(value='Пар доски: 0 (нужно ≥10, лучше 15–30)')
        ttk.Label(board_box, textvariable=self.sample_count).pack(anchor='w', pady=(4, 0))
        ttk.Label(board_box, wraplength=360, text=(
            'Камеры и объективы должны быть неподвижны, разрешение — одинаковым. '
            'Доска должна целиком помещаться в оба кадра. Меняйте наклон, расстояние и положение; '
            'не меняйте фокус и не переставляйте камеры после калибровки.'
        )).pack(fill='x', pady=(4, 0))

        settings_box = ttk.LabelFrame(controls, text='Параметры стерео', padding=8)
        settings_box.pack(fill='x', pady=(0, 8))
        grid = ttk.Frame(settings_box)
        grid.pack(fill='x')
        self.view = tk.StringVar(value='Карта глубины')
        self.max_width = tk.StringVar(value='640')
        self.num_disparities = tk.StringVar(value='128')
        self.sync_tolerance = tk.StringVar(value='33')
        self.min_depth = tk.StringVar(value='0.2')
        self.max_depth = tk.StringVar(value='10')
        rows = (
            ('Вид:', self.view, ('Карта глубины', 'Диспаратность'), True),
            ('Ширина обработки:', self.max_width, ('320', '480', '640', '960', '1280'), True),
            ('Диапазон диспаратности:', self.num_disparities, ('64', '128', '256'), True),
            ('Допуск синхронизации, мс:', self.sync_tolerance, None, False),
            ('Цвет: ближе, м:', self.min_depth, None, False),
            ('Цвет: дальше, м:', self.max_depth, None, False),
        )
        for row, (label, variable, values, readonly) in enumerate(rows):
            ttk.Label(grid, text=label).grid(row=row, column=0, sticky='w', pady=2)
            if readonly:
                widget = ttk.Combobox(grid, textvariable=variable, values=values, state='readonly', width=18)
                widget.bind('<<ComboboxSelected>>', lambda _event: self._apply_settings())
            else:
                widget = ttk.Spinbox(grid, from_=0.1, to=500, increment=1, textvariable=variable, width=10)
            widget.grid(row=row, column=1, sticky='e', pady=2)
        grid.columnconfigure(1, weight=1)
        ttk.Button(settings_box, text='Применить параметры', command=self._apply_settings).pack(fill='x', pady=(6, 0))
        ttk.Label(settings_box, wraplength=360, text=(
            'Синхронизация программная по времени получения кадров, не аппаратная. '
            'Для движущихся сцен нужна аппаратная синхронизация и короткая выдержка.'
        )).pack(fill='x', pady=(6, 0))

        self.notice = tk.StringVar(value='')
        ttk.Label(controls, textvariable=self.notice, wraplength=380).pack(fill='x')
        self.worker.start()
        self._poll()

    def update_streams(self, left, right):
        self.worker.set_streams(left, right)

    def close(self):
        if self.timer is not None:
            self.after_cancel(self.timer)
            self.timer = None
        self.worker.stop()
        self.worker.thread.join(timeout=1.0)

    def _toggle(self):
        state = self.worker.poll()
        self.worker.set_enabled(not state.enabled)
        if not state.enabled:
            self.status.set('Запуск стерео…')

    def _load(self):
        path = filedialog.askopenfilename(
            parent=self, title='Калибровка стереопары',
            filetypes=(('OpenCV calibration', '*.yml *.yaml *.json'), ('Все файлы', '*.*')),
        )
        if not path:
            return
        units = {'Из файла': 'auto', 'Метры': 'm', 'Сантиметры': 'cm', 'Миллиметры': 'mm', 'Неизвестны': 'unknown'}
        try:
            calibration = load_calibration(path, override_unit=units[self.load_unit.get()])
            self.worker.set_calibration(calibration)
            self.notice.set('Калибровка загружена.')
        except (OSError, StereoError, cv2.error) as exc:
            self.notice.set(f'Не удалось загрузить калибровку: {exc}')

    def _save(self):
        calibration = self.worker.poll().calibration
        if calibration is None:
            return
        path = filedialog.asksaveasfilename(
            parent=self, title='Сохранить калибровку стереопары', defaultextension='.yml',
            filetypes=(('OpenCV YAML', '*.yml'), ('OpenCV JSON', '*.json')),
        )
        if not path:
            return
        try:
            save_calibration(calibration, path)
            self.notice.set('Калибровка сохранена.')
        except (OSError, StereoError, cv2.error) as exc:
            self.notice.set(f'Не удалось сохранить калибровку: {exc}')

    def _capture(self):
        try:
            board = (int(self.board_columns.get()), int(self.board_rows.get()))
            square = float(self.square_mm.get().replace(',', '.'))
            self.worker.request_capture(board, square)
        except (ValueError, StereoError) as exc:
            self.notice.set(str(exc) if isinstance(exc, StereoError) else 'Проверьте числа углов и размер клетки.')

    def _calibrate(self):
        self.worker.request_calibration()

    def _apply_settings(self):
        try:
            settings = StereoSettings(
                max_width=int(self.max_width.get()),
                num_disparities=int(self.num_disparities.get()),
                sync_tolerance_ms=float(self.sync_tolerance.get().replace(',', '.')),
                min_depth_m=float(self.min_depth.get().replace(',', '.')),
                max_depth_m=float(self.max_depth.get().replace(',', '.')),
                view='depth' if self.view.get() == 'Карта глубины' else 'disparity',
            )
            self.worker.configure(settings)
            self.notice.set('Параметры стерео применены.')
        except (ValueError, StereoError) as exc:
            self.notice.set(str(exc) if isinstance(exc, StereoError) else 'Проверьте числовые параметры стерео.')

    def _describe_calibration(self, calibration):
        if calibration is None:
            return 'Калибровка не загружена.'
        width, height = calibration.image_size
        text = f'Размер кадров: {width}×{height}.'
        if calibration.baseline_m is None:
            text += ' Единицы T неизвестны: доступна только диспаратность, не метры.'
        else:
            text += f' Базовая линия: {calibration.baseline_m * 1000:.1f} мм.'
        if calibration.rms is not None:
            text += f' RMS: {calibration.rms:.3f} px.'
        return text

    def _poll(self):
        self.timer = None
        state = self.worker.poll()
        self.enable_button.configure(text='Выключить стерео' if state.enabled else 'Включить стерео')
        self.status.set(state.status)
        if state.status != self._status:
            self._status = state.status
            if state.result is None:
                self.preview.configure(image='', text=state.status)
        self.sample_count.set(f'Пар доски: {state.sample_count} (нужно ≥10, лучше 15–30)')
        if state.notice_id != self._notice_id:
            self._notice_id = state.notice_id
            self.notice.set(state.notice)
        if state.calibration is not self._calibration:
            self._calibration = state.calibration
            self.calibration_info.set(self._describe_calibration(state.calibration))
            self.save_button.configure(state='normal' if state.calibration else 'disabled')
        if state.busy != self._busy:
            self._busy = state.busy
            busy_state = 'disabled' if state.busy else 'normal'
            for widget in (self.capture_button, self.calibrate_button, self.clear_button, self.load_button):
                widget.configure(state=busy_state)
        if state.result is not self.last_result:
            self.last_result = state.result
            if state.result is None:
                self.frame_times.clear()
                self.photo = None
                self.preview.configure(image='', text=state.status)
                self.metrics.set('Стерео FPS: —  |  Обработка: —  |  Δt кадров: —  |  Совпадения: —')
                self.measurement.set('Нажмите на карту, чтобы измерить глубину в точке (медиана 9×9).')
            else:
                result = state.result
                self.frame_times.append(result.created_at)
                fps = '—'
                if len(self.frame_times) > 1:
                    elapsed = self.frame_times[-1] - self.frame_times[0]
                    if elapsed > 0:
                        fps = f'{(len(self.frame_times) - 1) / elapsed:.1f}'
                self.metrics.set(f'Стерео FPS: {fps}  |  Обработка: {result.elapsed_ms:.1f} мс  |  '
                                 f'Δt кадров: {result.sync_skew_ms:.1f} мс  |  Совпадения: {result.valid_fraction * 100:.0f}%')
                if result.view == 'depth':
                    self.legend.set('Карта глубины: красный — ближе, синий — дальше; чёрный — нет данных. Цветовой диапазон не ограничивает измерения.')
                else:
                    self.legend.set('Диспаратность: красный — больше смещение (обычно ближе), синий — меньше; чёрный — нет данных.')
                self._render(result)
        self.timer = self.after(40, self._poll)

    def _on_resize(self, _event=None):
        if self.last_result is not None:
            self._render(self.last_result)

    def _render(self, result):
        width = max(1, self.preview.winfo_width())
        height = max(1, self.preview.winfo_height())
        image = Image.fromarray(result.color_rgb)
        scale = min(width / image.width, height / image.height)
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(size, Image.Resampling.NEAREST)
        self.display_size = size
        self.display_offset = ((width - size[0]) // 2, (height - size[1]) // 2)
        self.photo = ImageTk.PhotoImage(image)
        self.preview.configure(image=self.photo, text='')

    def _measure(self, event):
        result = self.last_result
        if result is None:
            return
        width, height = self.display_size
        offset_x, offset_y = self.display_offset
        x = int((event.x - offset_x) * result.image_size[0] / width)
        y = int((event.y - offset_y) * result.image_size[1] / height)
        if not (0 <= x < result.image_size[0] and 0 <= y < result.image_size[1]):
            return
        data = result.depth_m if result.view == 'depth' and result.depth_m is not None else result.disparity
        patch = data[max(0, y - 4):y + 5, max(0, x - 4):x + 5]
        values = patch[np.isfinite(patch)]
        if result.view == 'depth' and result.depth_m is not None:
            values = values[values > 0]
        if values.size < 5:
            self.measurement.set('Нет данных в этой точке. Выберите область с текстурой и лучшим освещением.')
            return
        median = float(np.median(values))
        if result.view == 'depth' and result.depth_m is not None:
            self.measurement.set(f'Глубина в точке: {median:.2f} м (медиана 9×9, оценка по стерео).')
        else:
            self.measurement.set(f'Диспаратность в точке: {median:.1f} px (медиана 9×9).')
