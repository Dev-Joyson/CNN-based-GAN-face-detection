#!/usr/bin/env python3
"""Classify one image with a trained run -- the panel demo.

    python predict.py --config configs/test16_full.yaml photo.jpg [more.png ...]

The point of this file is that it CANNOT drift from training: it calls the same
load_and_resize (original -> 512 -> 256, bicubic, clip) and the same apply_mask
that built the train/val/test batches. A demo with its own resize is a demo of a
different model. That is why this is a script and not a notebook cell.
"""

import argparse
import os

import tensorflow as tf

from model import apply_mask, feathered_ellipse, load_and_resize, load_config


def preprocess(path, cfg, face_mask):
    image, _ = load_and_resize(tf.constant(path), tf.constant(0), cfg)   # uint8, as cached
    image = tf.cast(image, tf.float32) / 255.0
    image = apply_mask(image, face_mask, cfg.mask_mode)                    # identity for `none`
    return tf.clip_by_value(image, 0.0, 1.0)[tf.newaxis]                  # (1, H, W, 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="the config the model was trained with")
    ap.add_argument("images", nargs="+", help="image file(s), JPEG or PNG")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ckpt = os.path.join(cfg.run_dir, "model.keras")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"no trained model at {ckpt}; run train.py first")
    model = tf.keras.models.load_model(ckpt, safe_mode=False)          # Lambda layer
    face_mask = feathered_ellipse(cfg.img_size, cfg.mask_rx, cfg.mask_ry, cfg.mask_feather)

    print(f"model: {ckpt}  (mask_mode={cfg.mask_mode})")
    for path in args.images:
        p = float(model(preprocess(path, cfg, face_mask), training=False)[0, 0])
        label = "FAKE" if p > 0.5 else "REAL"
        confidence = p if p > 0.5 else 1.0 - p
        print(f"{os.path.basename(path):40s} {label:4s}  {confidence * 100:5.1f}%   (p_fake={p:.4f})")


if __name__ == "__main__":
    main()
