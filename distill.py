#!/usr/bin/env python3
"""Knowledge distillation: train this architecture to match a fine-tuned
baseline's probabilities instead of the hard labels.

    python distill.py --config configs/test17_sg2.yaml --teacher efficientnet_b0
    python distill.py --config configs/test17_sg2.yaml --teacher efficientnet_b0 --teacher mobilenet_v3_small
    python distill.py --config configs/test17_sg2.yaml --teacher efficientnet_b0 --no-extra --init

Why: the same 1M-param network reaches 0.964 from scratch on 35k images;
a fine-tuned EfficientNet-B0 reaches 0.9997 on the same data because it
arrived with ImageNet features. Its probabilities say how fake each image
is, not just whether -- far more signal per image than a 0/1 label. The
student keeps its architecture, so inference latency is unchanged; the
teacher is a training-time cost only.

Data: the headline's train split, plus (default) the EXTRA pool -- every
image in real_dir / fake_dir that the headline did not sample. Those still
carry their folder label; the teacher just adds the soft one. Val and test
are the headline's, untouched, scored on TRUE labels. Teacher logits are
computed once, offline, on the cached un-augmented images (one per image);
the student sees augmented views with that fixed target.

Loss (Hinton et al. 2015, binary form):
    alpha * T^2 * BCE(sigmoid(z_s/T), sigmoid(z_t/T))  +  (1-alpha) * BCE(p_s, y)

Output: experiments/<name>/ (default test18_distill -- the experiment
numbering continues from the test17 headline) -- the student's own
model.keras (a plain build_model, loadable by evaluate.py and predict.py),
history.csv, metrics.json with the distillation settings, then the full
evaluate() block.
"""

import argparse
import json
import os
from dataclasses import replace

import numpy as np
import tensorflow as tf

from model import (augment_and_mask, build_model, cached_dataset, compile_model,
                   feathered_ellipse, list_images, load_config, load_paths)

EPS = 1e-6


# --------------------------------------------------------------------------- #
# Loss and metrics on a packed target: y_true = [hard label, teacher logit]
# --------------------------------------------------------------------------- #

def _logit(p):
    p = tf.clip_by_value(p, EPS, 1.0 - EPS)
    return tf.math.log(p / (1.0 - p))


def make_kd_loss(temperature, alpha):
    bce = tf.keras.losses.BinaryCrossentropy()

    def loss(y_true, p_student):
        y = y_true[:, :1]
        z_t = y_true[:, 1:2]
        z_s = _logit(p_student)
        soft = bce(tf.sigmoid(z_t / temperature), tf.sigmoid(z_s / temperature))
        hard = bce(y, p_student)
        return alpha * (temperature ** 2) * soft + (1.0 - alpha) * hard
    loss.__name__ = "kd_loss"
    return loss


class HardAUC(tf.keras.metrics.AUC):
    """AUC against the hard label only (column 0 of the packed target)."""
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true[:, :1], y_pred, sample_weight)


class HardAccuracy(tf.keras.metrics.BinaryAccuracy):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true[:, :1], y_pred, sample_weight)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def extra_pool(cfg, splits):
    """Images in the folders that the headline did not sample. Labels are the
    folder's; these are not unlabeled, they are unused."""
    used = set()
    for paths, _ in splits.values():
        used.update(paths)
    real = [p for p in list_images(cfg.real_dir) if p not in used]
    fake = [p for p in list_images(cfg.fake_dir) if p not in used]
    return real + fake, [0] * len(real) + [1] * len(fake)


def teacher_logits(teachers, base_ds, batch=144):
    """Mean teacher logit per image, in dataset order. Offline, un-augmented."""
    ds = base_ds.map(lambda x, y: tf.cast(x, tf.float32) / 255.0).batch(batch)
    per = []
    for t in teachers:
        p = np.concatenate([t(xb, training=False).numpy().ravel() for xb in ds])
        per.append(np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS))))
    return np.mean(per, axis=0).astype(np.float32)


def kd_dataset(base_ds, z_t, cfg, face_mask, training, shuffle):
    """(augmented image, [y, z_t]) batches, aligned by zipping in cache order."""
    zt = tf.data.Dataset.from_tensor_slices(z_t)
    ds = tf.data.Dataset.zip((base_ds, zt))
    n = len(z_t)
    if shuffle:
        ds = ds.shuffle(min(cfg.shuffle_buffer, n), seed=cfg.seed,
                        reshuffle_each_iteration=True)

    def pack(xy, z):
        x, y = xy
        x, _ = augment_and_mask(x, y, cfg, face_mask, training)
        return x, tf.stack([tf.cast(y, tf.float32), z])
    ds = ds.map(pack, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="the HEADLINE config (data, split, student arch)")
    ap.add_argument("--teacher", action="append", required=True,
                    help="baseline name under experiments/<headline>/baselines/; repeat to average")
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.7, help="weight on the soft (teacher) term")
    ap.add_argument("--no-extra", action="store_true", help="train on the headline's 35k only")
    ap.add_argument("--init", action="store_true",
                    help="warm-start the student from the headline's model.keras instead of scratch")
    ap.add_argument("--name", default="test18_distill",
                    help="run name -> experiments/<name>/ (default test18_distill)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tag = "+".join(args.teacher) + ("" if args.no_extra else "+extra") + ("+init" if args.init else "")
    scfg = replace(cfg, name=args.name)
    os.makedirs(scfg.run_dir, exist_ok=True)
    tf.keras.utils.set_random_seed(cfg.seed)
    print(f"=== distill -> {scfg.run_dir}\n    teachers={args.teacher} T={args.temperature} "
          f"alpha={args.alpha} extra={not args.no_extra} init={args.init}")

    # --- teachers ----------------------------------------------------------------
    teachers = []
    for name in args.teacher:
        path = os.path.join(cfg.out_dir, cfg.name, "baselines", name, "model.keras")
        if not os.path.exists(path):
            raise FileNotFoundError(f"teacher not trained: {path}  (run baselines.py --train)")
        teachers.append(tf.keras.models.load_model(path, safe_mode=False))
        print(f"teacher {name}: {teachers[-1].count_params():,} params")

    # --- data: headline splits (+ extra), same caches ---------------------------------
    splits = load_paths(cfg)
    face_mask = feathered_ellipse(cfg.img_size, cfg.mask_rx, cfg.mask_ry, cfg.mask_feather)
    train_paths, train_labels = splits["train"]
    train_ds = cached_dataset(train_paths, train_labels, cfg, "train")   # the headline's own cache
    if not args.no_extra:
        xp, xl = extra_pool(cfg, splits)
        print(f"extra pool: {xl.count(0)} real + {xl.count(1)} fake unused by the headline")
        # a separate cache for the extras, concatenated after: the 35k are not
        # re-read, and the train cache stays shared with train.py
        train_ds = train_ds.concatenate(cached_dataset(xp, xl, cfg, "extra"))
        train_paths, train_labels = train_paths + xp, train_labels + xl
    base = {"train": train_ds, "val": cached_dataset(*splits["val"], cfg, "val")}
    print(f"train: {len(train_paths)}   val: {len(splits['val'][0])}")

    # --- teacher logits, once, in cache order -----------------------------------------
    z = {}
    for split, ds in base.items():
        print(f"teacher labels: {split} ...", flush=True)
        z[split] = teacher_logits(teachers, ds)
        p = 1 / (1 + np.exp(-z[split]))
        y = np.array(train_labels if split == "train" else splits["val"][1])
        print(f"  teacher on {split}: mean p_fake real={p[y == 0].mean():.3f} fake={p[y == 1].mean():.3f}")
    del teachers

    ds_train = kd_dataset(base["train"], z["train"], cfg, face_mask, training=True, shuffle=True)
    ds_val = kd_dataset(base["val"], z["val"], cfg, face_mask, training=False, shuffle=False)

    # --- student -------------------------------------------------------------------------
    student = build_model(cfg)
    if args.init:
        src = os.path.join(cfg.run_dir, "model.keras")
        student.set_weights(tf.keras.models.load_model(src, safe_mode=False).get_weights())
        print(f"student initialised from {src}")
    student.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.lr),
                    loss=make_kd_loss(args.temperature, args.alpha),
                    metrics=[HardAccuracy(name="accuracy"), HardAUC(name="auc")])
    student.summary()

    weights = os.path.join(scfg.run_dir, "best.weights.h5")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(weights, monitor="val_auc", mode="max",
                                           save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                        patience=cfg.patience, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(os.path.join(scfg.run_dir, "history.csv")),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(scfg.run_dir, "tb"), write_graph=False),
    ]
    student.fit(ds_train, validation_data=ds_val, epochs=cfg.epochs, callbacks=callbacks)

    # --- save a PLAIN model (no custom loss in its config) so evaluate/predict load it ---
    clean = compile_model(build_model(cfg), cfg)
    clean.set_weights(student.get_weights())
    clean.save(os.path.join(scfg.run_dir, "model.keras"))
    with open(os.path.join(scfg.run_dir, "metrics.json"), "w") as f:
        json.dump({"distill": {"tag": tag, "headline": cfg.name,
                               "teachers": args.teacher, "temperature": args.temperature,
                               "alpha": args.alpha, "extra": not args.no_extra, "init": args.init,
                               "n_train": len(train_paths)}}, f, indent=2)

    from evaluate import evaluate
    evaluate(scfg)


if __name__ == "__main__":
    main()
