#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""predict_tab_digit.py

Utility for **inference only** with the guitar-TAB digit CNN trained via
`train_tab_digit_model.py`.

Given a single image (PNG/JPEG/…), the script will load the specified model
(`.h5` *or* `.tflite`) and print the predicted fret number (0-24).

Example CLI usage:
    python3 predict_tab_digit.py path/to/image.png \
        --model models/tab_digit_cnn.h5

The module also exposes a `predict_digit()` function so it can be imported and
used programmatically:

```python
from predict_tab_digit import predict_digit
label = predict_digit("some_image.png", model_path="models/best.h5")
```
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Union, Tuple

import numpy as np
from PIL import Image

# `tensorflow` is optional if you only intend to use the TFLite model at
# runtime.  We therefore import lazily inside helper functions.

# -----------------------------------------------------------------------------
# Constants (must match `train_tab_digit_model.py`)
# -----------------------------------------------------------------------------
IMG_SIZE = 40
LABELS = list(range(25))  # 0-24 inclusive

# -----------------------------------------------------------------------------
# Image pre-processing
# -----------------------------------------------------------------------------

def _load_and_preprocess(img_path: Union[str, Path]) -> np.ndarray:
    """Return a (40, 40, 1) float32 array in *digit-white-on-black* format."""
    img = Image.open(img_path).convert("L")  # grayscale
    # High-quality down/upsampling → same as in training pipeline, with
    # Pillow ≥10 introducing the `Image.Resampling` enum.
    ResamplingEnum = getattr(Image, "Resampling", None)
    if ResamplingEnum is not None:
        resample = getattr(ResamplingEnum, "LANCZOS")
    else:
        # `Image.BICUBIC` may also be absent in very old Pillow stubs; use
        # a safe chained getattr call.
        resample = getattr(
            Image,
            "LANCZOS",
            getattr(Image, "BICUBIC", getattr(Image, "BILINEAR", 2)),
        )
    img = img.resize((IMG_SIZE, IMG_SIZE), resample=resample)

    arr = np.asarray(img, dtype=np.float32) / 255.0  # range [0, 1]

    # Convert to *digit-white on black* (the model's expected polarity).
    # Heuristic: if the image is mostly bright (>0.5) assume a white background
    # and invert.
    if arr.mean() > 0.5:
        arr = 1.0 - arr

    # (H, W) → (H, W, 1)
    arr = arr[..., np.newaxis]
    return arr.astype(np.float32)

# -----------------------------------------------------------------------------
# Model wrappers
# -----------------------------------------------------------------------------

def _predict_keras(model_path: Union[str, Path], sample: np.ndarray) -> Tuple[int, float]:
    """Return *(label, confidence)* using a standard Keras HDF5 model."""
    import tensorflow as tf  # pylint: disable=import-error
    keras = tf.keras  # type: ignore[attr-defined] pylint: disable=no-member

    model = keras.models.load_model(model_path, compile=False)
    preds = model.predict(sample[np.newaxis, ...], verbose=0)[0]
    label = int(np.argmax(preds))
    conf = float(preds[label])
    return label, conf


def _predict_tflite(model_path: Union[str, Path], sample: np.ndarray) -> Tuple[int, float]:
    """Return *(label, confidence)* using a TensorFlow-Lite model."""
    import tensorflow as tf  # pylint: disable=import-error

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_idx = interpreter.get_input_details()[0]["index"]
    interpreter.set_tensor(input_idx, sample[np.newaxis, ...])
    interpreter.invoke()
    output_idx = interpreter.get_output_details()[0]["index"]
    preds = interpreter.get_tensor(output_idx)[0]
    label = int(np.argmax(preds))
    conf = float(preds[label])
    return label, conf

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def predict_digit(img_path: Union[str, Path], *, model_path: Union[str, Path] = "tab_digit_cnn.h5") -> Tuple[int, float]:  # noqa: D401
    """Return *(label, confidence)* for *img_path* using *model_path*.

    Confidence is the softmax probability (0–1) of the predicted class.
    """
    img_path = Path(img_path)
    if not img_path.exists():
        raise FileNotFoundError(img_path)
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    sample = _load_and_preprocess(img_path)

    if model_path.suffix.lower() in {".tflite"}:
        return _predict_tflite(model_path, sample)
    return _predict_keras(model_path, sample)

# -----------------------------------------------------------------------------
# CLI entry-point
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Predict guitar-TAB digit (0-24) from an image.")
    parser.add_argument("image", type=str, help="Path to input image (PNG/JPEG/…)")
    parser.add_argument("--model", type=str, default="tab_digit_cnn.h5", help="Path to .h5 or .tflite model file")
    args = parser.parse_args()

    label, conf = predict_digit(args.image, model_path=args.model)
    print(f"{label}\t{conf:.3f}")


if __name__ == "__main__":
    main() 