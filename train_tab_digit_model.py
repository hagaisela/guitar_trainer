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
import cv2

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
        "/usr/local/opt/font-dejavu/lib/fonts/DejaVuSansMono.ttf",  # brew install font-dejavu
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
        # Fallback to PIL's built-in bitmap font so script still runs, albeit
        # with limited appearance diversity.
        print("[WARN] No TrueType fonts found – falling back to PIL default font. Accuracy may be lower.")
        fonts.append(ImageFont.load_default())
    return fonts

# Extra thin / proportional fonts (added to default monospace set)
EXTRA_FONTS = [
    "assets/fonts/RobotoCondensed-Regular.ttf",
    "assets/fonts/FreeSerif.ttf",
    "assets/fonts/DejaVuSerif.ttf",
]

# Initialise global font list once the helper is defined
FONTS = _load_fonts(extra_fonts=[Path(p) for p in EXTRA_FONTS])

def _generate_sample(label: int, *, fonts: List[ImageFont.FreeTypeFont] | None = None, line_prob: float = 0.3) -> Tuple[np.ndarray, int]:
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

    # Draw the six TAB staff lines with probability *line_prob* to match the
    # cleaned-crop distribution (typically 20–30 %).
    if random.random() < line_prob:
        # Draw 1–3 horizontal lines.  70 % of the time it's just one line
        # (the most common residual after line-removal), 20 % two, 10 % three.
        n_lines = random.choices([1, 2, 3], weights=[0.7, 0.2, 0.1])[0]

        # y-centre of the digit so we can ensure one line crosses it.
        y_centre = y + th // 2

        # Candidate vertical offsets (in pixels) relative to the centre line.
        # We favour small offsets so extra lines sit close to the glyph.
        cand_offs = [0, -4, 4, -8, 8]
        offs = [0]
        if n_lines > 1:
            # Choose additional unique offsets (positive or negative) aside from 0.
            extra = random.sample(cand_offs[1:], k=n_lines - 1)
            offs.extend(extra)

        grey = random.randint(120, 190)
        thickness = random.choice([1, 1, 2])
        for off in offs:
            y_line = y_centre + off
            if 0 <= y_line < IMG_SIZE:
                draw.line((0, y_line, IMG_SIZE, y_line), fill=grey, width=thickness)

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

    # If background happens to be white (bg_colour = 255), invert so the
    # model always sees *white digit on black*.
    if bg_colour == 255:
        arr = 1.0 - arr
    arr  = np.expand_dims(arr, axis=-1)
    return arr, label

def _generate_sample_realistic(label: int, *, fonts: List[ImageFont.FreeTypeFont] | None = None) -> Tuple[np.ndarray, int]:
    """Return *(image, label)* mimicking the *cropped* frame detector.

    Key differences from the simple generator:
    1. Always draws 6 TAB staff lines and *usually* (80 %) vertical frame bars.
    2. Uses proportional fonts (Roboto/Arial‐like) more often.
    3. Adds an optional trailing timing dot (2–3 px) right of the glyph.
    4. Post-renders JPEG compression + blur matching YouTube artefacts.
    The output is still a 40×40 monochrome crop with *white* digit on black.
    """

    assert 0 <= label <= 24
    IMG_W, IMG_H = IMG_SIZE, IMG_SIZE
    bg_colour = 0  # black background
    img = Image.new("L", (IMG_W, IMG_H), color=bg_colour)
    draw = ImageDraw.Draw(img)

    # ── staff lines (always present in realistic generator)
    spacing = random.randint(4, 6)
    top = random.randint(4, IMG_H - spacing * 5 - 4)
    line_col = random.randint(160, 220)
    for i in range(6):
        y_line = top + i * spacing
        draw.line((0, y_line, IMG_W, y_line), fill=line_col, width=1)

    # ── vertical frame bars (80 % probability to mimic variety in sources)
    if random.random() < 0.8:
        bar_w = random.choice([1, 2])
        bar_col = 255  # pure white so they are always visible
        draw.rectangle((0, 0, bar_w - 1, IMG_H - 1), fill=bar_col)
        draw.rectangle((IMG_W - bar_w, 0, IMG_W - 1, IMG_H - 1), fill=bar_col)
    else:
        bar_w = 0  # used later for digit placement bounds

    # ── font selection
    font_pool = fonts if fonts else FONTS
    font_obj = random.choice(font_pool)
    font_size = random.randint(20, 44)
    try:
        tt_path = font_obj.path
    except AttributeError:
        tt_path = font_obj.font.family if hasattr(font_obj, "font") else None
    font = ImageFont.truetype(tt_path, font_size) if tt_path else font_obj

    stroke_w = random.choice([0, 1])

    text = str(label)
    tw, th = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_w)[2:4]

    # Leave 2 px from potential frame bar + small jitter, guard against
    # negative range when the glyph is almost full width/height.
    min_x = bar_w + 2
    max_x = IMG_W - bar_w - tw - 2
    if max_x <= min_x:
        x = min_x
    else:
        x = random.randint(min_x, max_x)

    max_y = IMG_H - th
    y = 0 if max_y <= 0 else random.randint(0, max_y)

    draw.text((x, y), text, fill=255, font=font,
              stroke_width=stroke_w, stroke_fill=255)

    # ── optional timing dot (5 %)
    if random.random() < 0.05 and x + tw + 4 < IMG_W - bar_w:
        cx = x + tw + random.randint(2, 3)
        cy = y + th // 2
        draw.rectangle((cx, cy, cx + 2, cy + 2), fill=255)

    # ── YouTube-like artefacts
    if random.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.4, 0.9)))
    if random.random() < 0.4:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(35, 60))
        img = Image.open(buf)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    # Maintain white digit on black. Our background is 0 (black) so normally
    # no inversion is needed; keep a conditional guard for completeness.
    if bg_colour == 255:
        arr = 1.0 - arr
    arr = np.expand_dims(arr, axis=-1)
    return arr, label

class TabDigitSequence(keras.utils.Sequence):
    """Keras data generator that synthesises images on the fly."""

    def __init__(self, batch_size: int, steps: int, fonts: List[ImageFont.FreeTypeFont], line_prob: float = 0.3):
        self.batch_size = batch_size
        self.steps = steps
        self.fonts = fonts
        self.line_prob = line_prob

    def __len__(self) -> int:
        return self.steps

    def __getitem__(self, idx):  # type: ignore
        x = np.zeros((self.batch_size, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
        y = np.zeros((self.batch_size,), dtype=np.int32)
        for i in range(self.batch_size):
            lbl = random.randint(0, 24)
            img, lab = _generate_sample(lbl, fonts=self.fonts, line_prob=self.line_prob)
            x[i] = img; y[i] = lab
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
    parser.add_argument("--line-prob", type=float, default=0.3, help="probability that a synthetic sample includes the six TAB staff lines (0-1)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    fonts = _load_fonts([Path(f) for f in args.font] if args.font else None)
    print(f"Loaded {len(fonts)} fonts for augmentation")

    # Single generator with configurable probabilities
    gen_fn = _generate_sample

    # -----------------------------------------------------------------
    # Preview mode: save a handful of random samples and exit early
    # -----------------------------------------------------------------
    if args.preview:
        preview_dir = out_dir / "preview"
        preview_dir.mkdir(parents=True, exist_ok=True)
        for i in range(25):
            label = random.randint(0, 24)
            arr, lbl = gen_fn(label, fonts=fonts, line_prob=args.line_prob)
            # Save in the same polarity used at inference time: white digit on black
            img = arr[:, :, 0] * 255.0
            img_pil = Image.fromarray(img.astype(np.uint8), mode="L")
            img_pil.save(preview_dir / f"sample_{i:02d}_lbl{lbl}.png")
        print(f"Preview images written to {preview_dir} → exiting")
        return

    class _Seq(TabDigitSequence):
        """TabDigitSequence that uses the chosen generator function."""

        def __init__(self, batch_size: int, steps: int, fonts, line_prob: float):
            super().__init__(batch_size, steps, fonts, line_prob=line_prob)
            self.gen_fn = gen_fn

        def __getitem__(self, idx):  # type: ignore[override]
            x = np.zeros((self.batch_size, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
            y = np.zeros((self.batch_size,), dtype=np.int32)
            for i in range(self.batch_size):
                lbl = random.randint(0, 24)
                img, lab = self.gen_fn(lbl, fonts=self.fonts, line_prob=self.line_prob)
                x[i] = img; y[i] = lab
            return x, keras.utils.to_categorical(y, num_classes=len(LABELS))

    train_gen = _Seq(args.batch_size, args.steps_per_epoch, fonts, args.line_prob)
    val_gen   = _Seq(args.batch_size, args.val_steps, fonts, args.line_prob)

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