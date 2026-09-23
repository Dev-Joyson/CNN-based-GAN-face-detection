#!/usr/bin/env python3
"""Held-out generalisation test: score trained models on images from a
generator, or a real-image source, that they never trained on.

    python heldout.py --config configs/test19_sg2_crop_no_fft.yaml \\
        --fake-dir "/content/drive/MyDrive/Fake(SG3-T-psi1)" --tag sg3t
    python heldout.py --config configs/test19_sg2_crop_no_fft.yaml \\
        --real-dir "/content/drive/MyDrive/CelebA-HQ" --tag celebahq
    python heldout.py --config configs/test20_crop_baselines.yaml --model efficientnet_b0 \\
        --fake-dir "/content/drive/MyDrive/Fake(SG3-T-psi1)" --tag sg3t

--config names the trained model (its run dir) and the input pipeline it
expects (crop / resize, cache size). --fake-dir and/or --real-dir name the
held-out folders; whichever is not given is filled from the config's own
folder, using only images the headline never sampled -- disjoint from its
train/val/test by construction, and checked. Balanced: n per class.

Everything goes through the same cached_dataset -> eval_view pipeline the
model was evaluated with, so the only thing that changes is where the images
came from. Writes experiments/<name>/eval_heldout_<tag>.json.

This is the test the in-domain numbers cannot substitute for: a detector at
0.9996 on StyleGAN2 that scores 0.6 on StyleGAN3 has learned StyleGAN2, not
"generated". It is also the FFT branch's last exam -- the literature's claim
for frequency features is cross-generator transfer, not in-domain accuracy.
"""

import argparse
import json
import os
import random

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from model import (augment_and_mask, cached_dataset, feathered_ellipse, list_images,
                   load_config, load_paths)


def pick(folder, n, exclude, seed):
    files = [p for p in list_images(folder) if p not in exclude]
    if len(files) < n:
        print(f"  {folder}: only {len(files)} usable images (asked {n})")
        n = len(files)
    return sorted(random.Random(seed).sample(files, n))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="the trained model's config")
    ap.add_argument("--fake-dir", help="held-out generator's images (default: config's fake_dir, unsampled part)")
    ap.add_argument("--real-dir", help="held-out real source (default: config's real_dir, unsampled part)")
    ap.add_argument("--model", help="score this baseline (experiments/<name>/baselines/<model>/) instead of the headline")
    ap.add_argument("--n", type=int, default=1500, help="images per class")
    ap.add_argument("--tag", required=True, help="eval_heldout_<tag>.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    # a baseline lives under the headline's run dir; its ImageNet preprocessing is a
    # layer inside model.keras, so it takes the same pipeline output as the headline
    run_dir = os.path.join(cfg.run_dir, "baselines", args.model) if args.model else cfg.run_dir
    model_name = f"{cfg.name}/baselines/{args.model}" if args.model else cfg.name
    model = tf.keras.models.load_model(os.path.join(run_dir, "model.keras"), safe_mode=False)

    # everything the headline touched -- the held-out set must not overlap it
    used = set()
    for paths, _ in load_paths(cfg).values():
        used.update(paths)

    real = pick(args.real_dir or cfg.real_dir, args.n, used, cfg.seed)
    fake = pick(args.fake_dir or cfg.fake_dir, args.n, used, cfg.seed)
    assert not (set(real) | set(fake)) & used, "held-out set overlaps the headline's splits"
    n = min(len(real), len(fake))
    real, fake = real[:n], fake[:n]
    print(f"held-out '{args.tag}': {n} real from {args.real_dir or cfg.real_dir + ' (unsampled)'}\n"
          f"                     {n} fake from {args.fake_dir or cfg.fake_dir + ' (unsampled)'}")

    face_mask = feathered_ellipse(cfg.img_size, cfg.mask_rx, cfg.mask_ry, cfg.mask_feather)
    paths, labels = real + fake, [0] * n + [1] * n
    ds = cached_dataset(paths, labels, cfg, f"heldout_{args.tag}")      # its own cache file
    ds = ds.map(lambda x, y: augment_and_mask(x, y, cfg, face_mask, training=False),
                num_parallel_calls=tf.data.AUTOTUNE).batch(cfg.batch_size)

    y_true, y_score = [], []
    for xb, yb in ds:
        y_score.extend(model(xb, training=False).numpy().ravel())
        y_true.extend(yb.numpy())
    y_true, y_score = np.array(y_true), np.array(y_score)
    auc = float(roc_auc_score(y_true, y_score))
    acc = float(((y_score > 0.5) == (y_true == 1)).mean())
    p_real, p_fake = y_score[y_true == 0], y_score[y_true == 1]
    out = {
        "model": model_name, "tag": args.tag, "n_per_class": n,
        "real_dir": args.real_dir or cfg.real_dir, "fake_dir": args.fake_dir or cfg.fake_dir,
        "auc": round(auc, 4), "accuracy_at_0.5": round(acc, 4),
        "mean_p_fake": {"real": round(float(p_real.mean()), 4), "fake": round(float(p_fake.mean()), 4)},
        "fake_recall_at_0.5": round(float((p_fake > 0.5).mean()), 4),
        "real_recall_at_0.5": round(float((p_real < 0.5).mean()), 4),
    }
    with open(os.path.join(run_dir, f"eval_heldout_{args.tag}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"{model_name} on {args.tag}:  AUC {auc:.4f}   acc@0.5 {acc:.3f}   "
          f"mean p_fake real={p_real.mean():.3f} fake={p_fake.mean():.3f}   "
          f"fake recall {out['fake_recall_at_0.5']:.3f}")


if __name__ == "__main__":
    main()
