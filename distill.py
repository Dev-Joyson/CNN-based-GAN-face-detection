#!/usr/bin/env python3
"""Knowledge distillation: train this architecture to match a fine-tuned
baseline's probabilities instead of the hard labels.

    python train.py --config configs/test18_distill.yaml

The config is the headline's data fields (so the split is identical) plus a
`distill:` block naming the headline, the teachers, and the settings.
train.py sees the block and comes here.

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

Output: experiments/<name>/ -- the student's own model.keras (a plain
build_model, loadable by evaluate.py and predict.py), history.csv,
metrics.json with the distillation settings, then the full evaluate() block.
"""

import argparse
import json
import os
from dataclasses import replace

import numpy as np
import tensorflow as tf

from model import (HistoryCSV, augment_and_mask, build_model, cached_dataset, compile_model,
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
    """Images in the folders that the headline did not sample, BALANCED: the
    same number from each class, seeded. Labels are the folder's; these are
    not unlabeled, they are unused. The folders are 70k / 40k, so the
    leftovers after a 25k draw are 45k / 15k -- taking all of them would make
    training 62.5k real vs 32.5k fake."""
    import random
    used = set()
    for paths, _ in splits.values():
        used.update(paths)
    real = [p for p in list_images(cfg.real_dir) if p not in used]
    fake = [p for p in list_images(cfg.fake_dir) if p not in used]
    n = min(len(real), len(fake))
    rng = random.Random(cfg.seed)
    real = sorted(rng.sample(real, n)) if len(real) > n else real
    fake = sorted(rng.sample(fake, n)) if len(fake) > n else fake
    return real + fake, [0] * n + [1] * n


def teacher_logits(teachers, base_ds, batch=144):
    """Mean teacher logit per image, in dataset order. Offline, un-augmented."""
    ds = base_ds.map(lambda x, y: tf.cast(x, tf.float32) / 255.0).batch(batch)
    per = []
    for t in teachers:
        p = np.concatenate([t(xb, training=False).numpy().ravel() for xb in ds])
        per.append(np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS))))
    return np.mean(per, axis=0).astype(np.float32)


def fit_calibration(z_val, y_val):
    """Temperature scaling (Guo et al. 2017): one scalar T_cal such that
    sigmoid(z / T_cal) minimises BCE against the TRUE val labels. A teacher at
    mean p_fake 0.003 / 0.999 is over-confident; dividing its logits by T_cal
    turns them into probabilities that mean what they say, which is what
    distillation needs. Fitted on val only -- never test. Grid search; the
    objective is 1-D and smooth."""
    y = y_val.astype(np.float32)
    best_t, best_nll = 1.0, np.inf
    for t in np.concatenate([np.linspace(0.5, 5, 46), np.linspace(5.5, 30, 50)]):
        p = np.clip(1 / (1 + np.exp(-z_val / t)), EPS, 1 - EPS)
        nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if nll < best_nll:
            best_t, best_nll = float(t), float(nll)
    return best_t, best_nll


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

def distill(cfg):
    d = cfg.distill
    hcfg = replace(cfg, name=d["headline"], distill=None)      # the headline: same data, its run dir
    os.makedirs(cfg.run_dir, exist_ok=True)
    tf.keras.utils.set_random_seed(cfg.seed)
    print(f"=== {cfg.name}: distill from {d['teachers']} of {hcfg.name} -> {cfg.run_dir}\n"
          f"    T={d['temperature']} alpha={d['alpha']} extra={d['extra']} init={d['init']}")

    # --- teachers ----------------------------------------------------------------
    teachers = []
    for name in d["teachers"]:
        path = os.path.join(hcfg.run_dir, "baselines", name, "model.keras")
        if not os.path.exists(path):
            raise FileNotFoundError(f"teacher not trained: {path}  (run baselines.py --train)")
        teachers.append(tf.keras.models.load_model(path, safe_mode=False))
        print(f"teacher {name}: {teachers[-1].count_params():,} params")

    # --- data: the headline's splits (+ extra), the headline's caches ---------------
    splits = load_paths(hcfg)
    face_mask = feathered_ellipse(cfg.img_size, cfg.mask_rx, cfg.mask_ry, cfg.mask_feather)
    train_paths, train_labels = splits["train"]
    train_ds = cached_dataset(train_paths, train_labels, hcfg, "train")   # the headline's own cache
    if d["extra"]:
        xp, xl = extra_pool(hcfg, splits)
        print(f"extra pool: {xl.count(0)} real + {xl.count(1)} fake unused by the headline (balanced)")
        # a separate cache for the extras, concatenated after: the 35k are not
        # re-read, and the train cache stays shared with train.py
        train_ds = train_ds.concatenate(cached_dataset(xp, xl, hcfg, "extra"))
        train_paths, train_labels = train_paths + xp, train_labels + xl
    base = {"train": train_ds, "val": cached_dataset(*splits["val"], hcfg, "val")}
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

    if d["calibrate"]:
        y_val = np.array(splits["val"][1])
        t_cal, nll = fit_calibration(z["val"], y_val)
        z = {k: v / t_cal for k, v in z.items()}
        p = 1 / (1 + np.exp(-z["train"]))
        y = np.array(train_labels)
        print(f"calibration: T_cal={t_cal:.2f} (val NLL {nll:.4f}); teacher on train after: "
              f"mean p_fake real={p[y == 0].mean():.3f} fake={p[y == 1].mean():.3f}")

    ds_train = kd_dataset(base["train"], z["train"], cfg, face_mask, training=True, shuffle=True)
    ds_val = kd_dataset(base["val"], z["val"], cfg, face_mask, training=False, shuffle=False)

    # --- student -------------------------------------------------------------------------
    student = build_model(cfg)
    if d["init"]:
        src = os.path.join(hcfg.run_dir, "model.keras")
        student.set_weights(tf.keras.models.load_model(src, safe_mode=False).get_weights())
        print(f"student initialised from {src}")
    student.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.lr),
                    loss=make_kd_loss(d["temperature"], d["alpha"]),
                    metrics=[HardAccuracy(name="accuracy"), HardAUC(name="auc")])
    student.summary()

    weights = os.path.join(cfg.run_dir, "best.weights.h5")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(weights, monitor="val_auc", mode="max",
                                           save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                        patience=cfg.patience, restore_best_weights=True),
        HistoryCSV(os.path.join(cfg.run_dir, "history.csv")),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(cfg.run_dir, "tb"), write_graph=False),
    ]
    student.fit(ds_train, validation_data=ds_val, epochs=cfg.epochs, callbacks=callbacks)

    with open(os.path.join(cfg.run_dir, "metrics.json"), "w") as f:
        json.dump({"distill": d, "n_train": len(train_paths),
                   "t_cal": t_cal if d["calibrate"] else None}, f, indent=2)
    finalize(cfg, student.get_weights())


def finalize(cfg, weights=None):
    """Write a PLAIN model.keras (no custom loss in its config, so evaluate.py
    and predict.py load it) and evaluate. Called at the end of training, or via
    --finalize on a run that was killed or lost its VM: the ModelCheckpoint
    wrote best.weights.h5 after every improving epoch, so nothing is lost."""
    clean = compile_model(build_model(cfg), cfg)
    if weights is None:
        clean.load_weights(os.path.join(cfg.run_dir, "best.weights.h5"))
        print(f"finalize: loaded best.weights.h5 from {cfg.run_dir}")
    else:
        clean.set_weights(weights)
    clean.save(os.path.join(cfg.run_dir, "model.keras"))
    from evaluate import evaluate
    evaluate(cfg)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="a config with a distill: block")
    ap.add_argument("--finalize", action="store_true",
                    help="skip training: build model.keras from best.weights.h5 and evaluate")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if not cfg.distill:
        raise SystemExit(f"{args.config} has no distill: block")
    finalize(cfg) if args.finalize else distill(cfg)


if __name__ == "__main__":
    main()
