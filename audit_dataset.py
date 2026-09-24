#!/usr/bin/env python3
"""Audit the real/fake folders for shortcuts.

Asks one question: can a single trivial property of the FILES separate real from
fake, without looking at the face at all? Every property is scored as an AUC --
the same metric the model reports -- so the numbers are directly comparable.

    AUC 0.50  = that property tells you nothing (good)
    AUC 0.90  = the model could score 0.90 on this property alone (shortcut)

Usage:
    python audit_dataset.py --config configs/test13_face.yaml
    python audit_dataset.py --config configs/test13_face.yaml -n 800
"""

import argparse
import os
import random
from collections import Counter
from glob import glob

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import roc_auc_score

from model import eval_view, list_images, load_and_resize, load_config


def file_features(path):
    """Cheap properties readable without decoding pixels properly."""
    f = {"file_size": float(os.path.getsize(path))}
    with Image.open(path) as im:
        f["width"], f["height"] = float(im.width), float(im.height)
        f["format"], f["mode"] = im.format, im.mode
        f["bytes_per_pixel"] = f["file_size"] / (im.width * im.height)
        # JPEG quantization tables -> a proxy for the encoder's quality setting
        q = getattr(im, "quantization", None)
        f["jpeg_q_table"] = float(np.mean(q[0])) if q else np.nan
        small = np.asarray(im.convert("RGB").resize((64, 64)), dtype=np.float32)
    f["brightness"] = float(small.mean())
    f["contrast"] = float(small.std())
    return f


def highfreq_energy(path, cfg):
    """High-frequency energy AFTER cfg's preprocessing.

    Resize mode: says whether the 512->256 normalisation erased the classes'
    different resampling histories. Crop mode: measured on the native 256^2
    centre patch -- where JPEG history and sensor noise in the reals, blurred
    away by the resize, are visible again. A high AUC here is a shortcut
    the resize-mode audit could not see.
    """
    img, _ = load_and_resize(tf.constant(path), tf.constant(0), cfg)
    img = eval_view(img, cfg)          # crop mode: the centre native patch the model sees
    g = tf.image.rgb_to_grayscale(tf.cast(img, tf.float32) / 255.0)[..., 0]
    mag = np.fft.fftshift(np.log1p(np.abs(tf.signal.fft2d(tf.cast(g, tf.complex64)).numpy())))
    n = cfg.img_size
    c = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
    return float(mag[r >= 0.75 * c].mean())


def collect(paths, cfg, label):
    rows = []
    name = "real" if label == 0 else "fake"
    for i, p in enumerate(paths, 1):
        try:
            f = file_features(p)
            f["highfreq"] = highfreq_energy(p, cfg)
            f["label"] = label
            rows.append(f)
        except Exception as e:                     # corrupt file, odd format
            print(f"  skipped {os.path.basename(p)}: {e}")
        if i % 100 == 0 or i == len(paths):
            # each image is two Drive reads of a ~1-2 MB PNG; this is slow and
            # silent without a heartbeat
            print(f"  {name}: {i}/{len(paths)}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("-n", type=int, default=400, help="images sampled per class")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rng = random.Random(cfg.seed)

    real = list_images(cfg.real_dir)
    fake = [q for d in cfg.fake_dirs for q in list_images(d)]
    if not real or not fake:
        raise FileNotFoundError(f"real={len(real)} fake={len(fake)} -- check paths in {args.config}")

    print(f"real: {len(real)} files   fake: {len(fake)} files")
    print(f"sampling {min(args.n, len(real), len(fake))} of each...\n")
    n = min(args.n, len(real), len(fake))
    rows = collect(rng.sample(real, n), cfg, 0) + collect(rng.sample(fake, n), cfg, 1)

    y = np.array([r["label"] for r in rows])

    # --- categorical: formats and modes -------------------------------------
    print("=" * 62)
    print("FILE FORMATS")
    for name, lbl in (("real", 0), ("fake", 1)):
        sub = [r for r in rows if r["label"] == lbl]
        print(f"  {name}: {dict(Counter(r['format'] for r in sub))}  "
              f"modes={dict(Counter(r['mode'] for r in sub))}")
    formats = {r["format"] for r in rows}
    if len(formats) > 1:
        print("  !! classes differ in file format -- a giveaway on its own.")
    if len(formats) > 1:
        print("     (JPEG leaves block artifacts in the pixels; PNG does not. If one class is")
        print("      JPEG and the other PNG, the model can read that after any resize.)")

    # --- numeric: one AUC per property --------------------------------------
    # Two groups, because they mean different things. FILE properties are erased
    # by decode+resize -- the model never sees a width. PIXEL properties survive
    # the pipeline, so the model CAN use them; that group decides the verdict.
    # A file-level gap is still a warning: if real and fake come at different
    # native sizes they travel different resampling paths to 256, and that can
    # leave a pixel-level trace -- which is exactly what `highfreq` measures.
    def score(keys):
        out = []
        for key in keys:
            v = np.array([r[key] for r in rows], dtype=float)
            if np.isnan(v).any() or len(np.unique(v)) < 2:
                continue
            auc = roc_auc_score(y, v)
            auc = max(auc, 1 - auc)                # direction doesn't matter
            out.append((auc, key, v[y == 0].mean(), v[y == 1].mean()))
        return sorted(out, reverse=True)

    def table(title, results):
        print("\n" + "=" * 62)
        print(title + "  (0.50 = harmless, 1.00 = perfect shortcut)\n")
        print(f"  {'property':<18}{'real mean':>12}{'fake mean':>12}{'AUC':>8}   verdict")
        print("  " + "-" * 58)
        for auc, key, rm, fm in results:
            verdict = "SHORTCUT" if auc >= 0.70 else ("suspicious" if auc >= 0.60 else "ok")
            print(f"  {key:<18}{rm:>12.2f}{fm:>12.2f}{auc:>8.3f}   {verdict}")

    file_res = score(("width", "height", "file_size", "bytes_per_pixel", "jpeg_q_table"))
    pixel_res = score(("highfreq", "brightness", "contrast"))
    table("FILE PROPERTIES -- erased by decode+resize; a warning, not a verdict", file_res)
    if any(k in ("width", "height") and a >= 0.7 for a, k, _, _ in file_res):
        print("\n  !! native sizes differ between classes -> different resampling paths to")
        print(f"     {cfg.img_size}px. Whether that leaves a trace the model can read is the")
        print("     `highfreq` row below. That row is the one that matters.")
    table("PIXEL PROPERTIES -- survive the pipeline; THIS decides the verdict", pixel_res)

    results = pixel_res
    worst = max(results)[0] if results else 0.5
    print("\n" + "=" * 62)
    if worst >= 0.70:
        print(f"VERDICT: a single file property reaches AUC {worst:.3f} without")
        print("looking at any face. Your model can use this instead of learning")
        print("to detect GAN artifacts. Fix the dataset before trusting results.")
    elif worst >= 0.60:
        print(f"VERDICT: mild leakage (best AUC {worst:.3f}). Worth a footnote;")
        print("probably not enough on its own to explain a 0.99 result.")
    else:
        print(f"VERDICT: clean. No single file property beats AUC {worst:.3f},")
        print("so the classes are not trivially separable by metadata.")
    print("This checks metadata only. A held-out generator is the real test.")


if __name__ == "__main__":
    main()
