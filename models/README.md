# Модели для режима «Скелет человека»

Обе модели работают локально через `cv2.dnn`. Интернет, GPU, ONNX Runtime, PyTorch
и TensorFlow не нужны — достаточно OpenCV, который уже входит в программу.

| Файл | Назначение | Вход | Выход | Размер |
|---|---|---|---|---|
| `nanodet_plus_m_320.onnx` | поиск людей в кадре | `1×3×320×320` | `1×2125×112` | 4.8 МБ |
| `rtmpose_t_256x192.onnx` | 17 точек тела по вырезанному человеку | `1×3×256×192` | `simcc_x 1×17×384`, `simcc_y 1×17×512` | 13.4 МБ |

## Происхождение и лицензии

- **NanoDet-Plus-m** — RangiLyu, <https://github.com/RangiLyu/nanodet>, лицензия
  Apache-2.0 (`licenses/NanoDet-Apache-2.0.txt`). Готовый ONNX-экспорт взят из
  открытого примера развёртывания <https://github.com/hpc203/nanodet-plus-opencv>
  (`opencv/onnxmodel/nanodet-plus-m_320.onnx`).
- **RTMPose-t** — OpenMMLab, <https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose>,
  лицензия Apache-2.0 (`licenses/RTMPose-Apache-2.0.txt`). ONNX-экспорт
  стандартного `rtmpose-t simcc 256×192`; побайтово совпадающие копии опубликованы в
  <https://github.com/siciyuan404/Good-GYM-Android> и
  <https://github.com/konyshevgmbh/pose_estimation_flutter>.
- Порядок точек — COCO-17. Обе модели используют только класс «человек».

Контрольные суммы SHA-256:

```
4f12723cce3d48e47ca92cb925ba74d97a965c069208edca660bbb9f7ce2c610  nanodet_plus_m_320.onnx
f0d2822b63dc40d10143ae1f0f039aef7be3a43621657cfeaaa4a86d96d33bfc  rtmpose_t_256x192.onnx
```

## Проверочное изображение

`assets/pose_sample.jpg` — уменьшенная копия портрета астронавта из набора
примеров scikit-image (`skimage.data.astronaut`, фото NASA, общественное
достояние; распространяется проектом по BSD-3-Clause,
`licenses/scikit-image-sample.txt`). Оно нужно только самотесту packaged-версии:
программа убеждается, что модели действительно загружаются и находят человека.
В обычной работе это изображение не используется.

## Почему не YOLO

`yolov8n-pose.onnx` тоже проверялся и работает в `cv2.dnn`, но распространяется
под AGPL-3.0, что накладывает обязательства на весь исполняемый файл. Связка
NanoDet-Plus + RTMPose даёт то же качество при лицензии Apache-2.0 и меньшем
размере (18 МБ против 13 МБ одного файла, но без ограничений AGPL).
