# Hamster detector model

`hamster-ssd-mobilenet-v2.onnx` is a modified, single-class conversion of Google **SSD MobileNet V2, Open Images V4**, version 1. Google publishes the original model under the Apache License 2.0: [model card and license](https://www.kaggle.com/models/google/mobilenet-v2/tensorFlow1/openimages-v4-ssd-mobilenet-v2/1) (original TF Hub handle: `https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1`). The license text is in [`public/licenses/Apache-2.0.txt`](../licenses/Apache-2.0.txt).

The conversion retains the MobileNet/SSD backbone, box decoding, and the original class channel **Hamster** (index 256 in the TF1 graph). The other 602 class channels, string lookup tables, TF1 preprocessing loop and multi-class NMS were removed; the browser reproduces the input resizing and selects a hamster box (searching overlapping regions when the full-frame pass has no detection). This is a modified model, not a Google-provided ONNX file.

- Included ONNX SHA-256: `d401301db53697bca98dfefb2425e73c32b1feff86b81574a73ac8ae6670b1f5`.
- Input SavedModel SHA-256: `9f49e034537b2518001794019869df0710f3089b0859ff10600408931a6a1895`.
- The SavedModel used for conversion was retrieved from the GitHub mirror [`namthai-dev/object-detection`, pinned Git blob `27252f127740194f63376182e4bc0fbc3f361c10`](https://github.com/namthai-dev/object-detection); the same blob is also present in `rh-aiservices-bu/dog-detector-service`. This is a mirror of the TF Hub model, not an independently verified upstream checksum.
- The inference engine, `onnxruntime-web`, is MIT licensed and supplied via npm; its [license](../licenses/ONNX-Runtime-MIT.txt) is included.

## Reproduce the conversion (optional)

The included ONNX file is sufficient to build and run the app. To regenerate an equivalent model, obtain the pinned `saved_model.pb`, then use Python 3.11 and the versions below:

```sh
gh api repos/namthai-dev/object-detection/git/blobs/27252f127740194f63376182e4bc0fbc3f361c10 \
  -H 'Accept: application/vnd.github.raw' > saved_model.pb
python -m venv .venv
. .venv/bin/activate
pip install 'tensorflow-cpu==2.15.1' 'tf2onnx==1.16.1'
python scripts/convert_hamster_model.py saved_model.pb hamster-rebuilt.onnx
```

The conversion script validates the source checksum and class name. The converter may reorder internal ONNX nodes, so a rebuilt file's byte checksum can differ while its numerical outputs agree. On test frames, the bundled ONNX hamster scores and boxes agreed with the original TF1 model within `3e-6`.
