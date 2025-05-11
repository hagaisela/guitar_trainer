#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=import-error,unused-import
"""train_tab_digit_model.py

Train a lightweight CNN (≈35 k parameters) to recognise guitar-TAB fret numbers
(0–24) from small monochrome crops.  Images are generated *on-the-fly* with
random fonts, sizes, translations, noise, optional TAB lines, etc., so no
external dataset is needed.  Training will happily run on a 6-core CPU; expect
~2–3 h for 30 epochs with the default settings (≈300 k synthetic samples).

After training the script writes both a standard Keras model
(`tab_digit_cnn.h5`) *and* an integer-quantised TensorFlow-Lite flatbuffer
(`tab_digit_cnn.tflite`) that gives fast inference on mobile/desktop CPUs.

Usage (all arguments are optional):

    python3 train_tab_digit_model.py \
        --epochs 25 \
        --steps-per-epoch 1000 \
        --batch-size 128 \
        --out models/

Dependencies (CPU-only):
    pip install tensorflow pillow numpy tqdm fonttools

The script tries to load a handful of common monospaced fonts (DejaVu Sans Mono,
Liberation Mono, Courier New…).  You can point to additional TTF files via the
`--font` option.
"""
from __future__ import annotations

import argparse
import io
import math
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

SEED = 1234
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# -----------------------------------------------------------------------------
# Pillow version compatibility helpers (>=10 moved resampling enums)
# -----------------------------------------------------------------------------
try:  # Pillow ≥10
    AFFINE = Image.Transform.AFFINE  # type: ignore[attr-defined]
except AttributeError:  # Pillow <10
    AFFINE = Image.AFFINE  # type: ignore[attr-defined]

try:
    BILINEAR = Image.Resampling.BILINEAR  # type: ignore[attr-defined]
    LANCZOS = Image.Resampling.LANCZOS    # type: ignore[attr-defined]
except AttributeError:
    BILINEAR = Image.BILINEAR  # type: ignore[attr-defined]
    LANCZOS = Image.LANCZOS    # type: ignore[attr-defined]

# -----------------------------------------------------------------------------
# Synthetic image generator
# -----------------------------------------------------------------------------
CHARS = [str(n) for n in range(10)]  # individual digits used to compose 0-24
LABELS = list(range(25))  # fret numbers 0-24 inclusive
IMG_SIZE = 40  # output image is IMG_SIZE×IMG_SIZE grayscale

def _load_fonts(extra_fonts: List[Path] | None = None) -> List[ImageFont.FreeTypeFont]:
    """Return a list of PIL ImageFont objects for random selection."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
        "/System/Library/Fonts/Courier.dfont",  # macOS built-in
    ]
    if extra_fonts:
        candidates.extend(map(str, extra_fonts))

    fonts: List[ImageFont.FreeTypeFont] = []
    for fp in candidates:
        try:
            fonts.append(ImageFont.truetype(fp, size=32))
        except Exception:
            pass  # ignore missing fonts
    if not fonts:
        raise RuntimeError("No usable TTF fonts found — install e.g. 'ttf-dejavu'.")
    return fonts

# Extra thin / proportional fonts (added to default monospace set)
EXTRA_FONTS = [
    "assets/fonts/RobotoCondensed-Regular.ttf",
    "assets/fonts/FreeSerif.ttf",
    "assets/fonts/DejaVuSerif.ttf",
]

# Initialise global font list once the helper is defined
FONTS = _load_fonts(extra_fonts=[Path(p) for p in EXTRA_FONTS])

def _generate_sample(label: int, *, fonts: List[ImageFont.FreeTypeFont] | None = None) -> Tuple[np.ndarray, int]:
    """Return *(image, label)* with digit white-on-black."""

    assert 0 <= label <= 24

    # Canvas: *black* or *white* with 50 % probability
    bg_white  = random.random() < 0.5
    bg_colour = 255 if bg_white else 0
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), color=bg_colour)
    draw = ImageDraw.Draw(img)

    # •  Font: anywhere from 18 px (thin) to 46 px (thick)
    font_pool = fonts if fonts else FONTS
    font_obj = random.choice(font_pool)       # pre-loaded FreeTypeFont
    font_size = random.randint(18, 46)
    # Need the original TTF path so we can reload at the desired size.
    try:
        tt_path = font_obj.path               # Pillow ≥9 FreeTypeFont carries .path
    except AttributeError:
        # Fallback: construct from font family (works on macOS dfont)
        tt_path = font_obj.font.family if hasattr(font_obj, "font") else None

    if tt_path is None:
        # As a last resort just reuse the original 32-px object (size mismatch ok)
        font = font_obj
    else:
        font = ImageFont.truetype(tt_path, font_size)

    # Stroke width 0–2 so glyphs may be very thin
    stroke_w  = random.choice([0, 1, 2])

    text = str(label)
    bbox  = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_w)
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]

    # If glyph exceeds canvas, clamp placement to (0,0)
    max_x = max(0, IMG_SIZE - tw)
    max_y = max(0, IMG_SIZE - th)
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)

    colour = 0 if bg_white else 255           # ensure contrast
    draw.text((x, y), text, fill=colour, font=font,
              stroke_width=stroke_w, stroke_fill=colour)

    # •  50 % of the time: *no* TAB lines at all  (videos sometimes omit them)
    if random.random() < 0.5:
        pass
    else:
        spacing = random.randint(4, 6)
        top = random.randint(0, IMG_SIZE - spacing*5 - 1)
        grey = random.randint(120, 190)
        for i in range(6):
            y_line = top + i*spacing
            draw.line((0, y_line, IMG_SIZE, y_line), fill=grey, width=1)

    # •  Perspective / shear   (mimic handheld-camera skew)
    if random.random() < 0.3:
        shear = random.uniform(-0.3, 0.3)
        img = img.transform(img.size, AFFINE, (1, shear, 0, 0, 1, 0),
                            resample=BILINEAR, fillcolor=bg_colour)

    # •  JPEG / motion blur  (YouTube artefacts)
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.6, 1.4)))
    if random.random() < 0.3:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(30, 60))
        img = Image.open(buf)

    # Optional rotation (kept from original code)
    if random.random() < 0.4:
        angle = random.uniform(-8, 8)
        img = img.rotate(angle, resample=BILINEAR, expand=False,
                         fillcolor=bg_colour)

    # ✗  Remove Max/Min filters — dilation/erosion is now redundant
    # -----------------------------------------------------------------
    arr = np.asarray(img, dtype=np.float32) / 255.0

    # •  Always convert to "digit-white on black" because the model expects that.
    if bg_white:
        arr = 1.0 - arr
    arr  = np.expand_dims(arr, axis=-1)
    return arr, label


class TabDigitSequence(keras.utils.Sequence):
    """Keras data generator that synthesises images on the fly."""

    def __init__(self, batch_size: int, steps: int, fonts: List[ImageFont.FreeTypeFont]):
        self.batch_size = batch_size
        self.steps = steps
        self.fonts = fonts

    def __len__(self) -> int:
        return self.steps

    def __getitem__(self, idx):  # pylint: disable=unused-argument
        x = np.zeros((self.batch_size, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
        y = np.zeros((self.batch_size,), dtype=np.int32)
        for i in range(self.batch_size):
            lbl = random.randint(0, 24)
            img, label = _generate_sample(lbl, fonts=self.fonts)
            x[i] = img
            y[i] = label
        return x, keras.utils.to_categorical(y, num_classes=len(LABELS))

# -----------------------------------------------------------------------------
# Model definition — a very small CNN (~35 k parameters)
# -----------------------------------------------------------------------------

def build_model() -> keras.Model:
    """Return a slightly wider CNN (~140 k params, still instant inference)."""
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = layers.Conv2D(32, 3, activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(len(LABELS), activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():  # noqa: C901
    parser = argparse.ArgumentParser(description="Train CNN for guitar-TAB digit recognition (0-24).")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-steps", type=int, default=100)
    parser.add_argument("--out", type=str, default=".", help="output directory")
    parser.add_argument("--font", action="append", type=str, help="additional .ttf font file to include")
    parser.add_argument("--preview", action="store_true", help="generate 25 sample images then exit")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    fonts = _load_fonts([Path(f) for f in args.font] if args.font else None)
    print(f"Loaded {len(fonts)} fonts for augmentation")

    # -----------------------------------------------------------------
    # Preview mode: save a handful of random samples and exit early
    # -----------------------------------------------------------------
    if args.preview:
        preview_dir = out_dir / "preview"
        preview_dir.mkdir(parents=True, exist_ok=True)
        for i in range(25):
            label = random.randint(0, 24)
            arr, lbl = _generate_sample(label, fonts=fonts)
            # Convert back to PIL.Image for saving (invert again so digits dark)
            img = (1.0 - arr[:, :, 0]) * 255.0
            img_pil = Image.fromarray(img.astype(np.uint8), mode="L")
            img_pil.save(preview_dir / f"sample_{i:02d}_lbl{lbl}.png")
        print(f"Preview images written to {preview_dir} → exiting")
        return

    train_gen = TabDigitSequence(args.batch_size, args.steps_per_epoch, fonts)
    val_gen = TabDigitSequence(args.batch_size, args.val_steps, fonts)

    model = build_model()
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(str(out_dir / "best.h5"), monitor="val_accuracy", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.2, patience=3, min_lr=1e-5),
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
        workers=4,
        use_multiprocessing=True,
    )

    print("Training finished — saving final model…")
    model.save(str(out_dir / "tab_digit_cnn.h5"))

    # ---------------------------------------------------------------------
    # Export TensorFlow-Lite quantised model for fast CPU inference
    # ---------------------------------------------------------------------
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        with open(str(out_dir / "tab_digit_cnn.tflite"), "wb") as f:
            f.write(tflite_model)
        print("TFLite model written →", out_dir / "tab_digit_cnn.tflite")
    except Exception as exc:  # pragma: no cover
        print("[WARN] TFLite conversion failed:", exc)


if __name__ == "__main__":
    main() 