#!/usr/bin/env python3
"""Evaluate a trained run: confusion matrix, ROC curve, Grad-CAM.

    python evaluate.py --config configs/test13_face.yaml

Reads experiments/<name>/model.keras and writes the figures next to it, so the
run folder ends up self-describing:

    experiments/<name>/
        model.keras          (from train.py)
        history.csv          (from train.py)
        metrics.json         (from train.py)
        confusion_matrix.png
        roc.png
        gradcam.png
        eval.json            per-class precision/recall, AUC, efficiency numbers

The datasets come from model.build_datasets, so the mask and the pipeline are
exactly what training saw -- an eval that quietly skipped the mask would report
a number the model never earned.
"""

import argparse
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")                  # no display on a Colab VM
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (auc, classification_report, confusion_matrix,
                             roc_curve)
from tensorflow.keras import layers

from model import build_datasets, load_config

CLASSES = ["Real", "Fake"]


def load_run_model(cfg):
    path = os.path.join(cfg.run_dir, "model.keras")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No trained model at {path}. Run:  python train.py --config <cfg>")
    # safe_mode=False: the FFT branch is a Lambda. Importing model.py has already
    # registered fft_layer as serializable, but Keras 3 still gates Lambda loads.
    return tf.keras.models.load_model(path, safe_mode=False)


def predict_split(model, ds):
    """One pass over a split -> (y_true, y_score)."""
    y_true, y_score = [], []
    for images, labels in ds:
        y_score.extend(model.predict(images, verbose=0).flatten())
        y_true.extend(labels.numpy())
    return np.array(y_true), np.array(y_score)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def plot_confusion(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(2), CLASSES)
    ax.set_yticks(range(2), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (test)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:d}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return cm


def plot_roc(y_true, y_score, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC (test)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return float(roc_auc)


# --------------------------------------------------------------------------- #
# Grad-CAM
# --------------------------------------------------------------------------- #

def last_spatial_conv(model):
    """The deepest conv of the SPATIAL branch.

    build_model names it "spatial_conv_5", so ask for that first. The fallback
    exists for checkpoints from before the layers were named: NOT simply the
    last Conv2D -- the FFT branch is built after the spatial one, so its
    32-filter conv comes last in layer order. Take the last conv at max width.
    """
    if any(l.name == "spatial_conv_5" for l in model.layers):
        return "spatial_conv_5"
    convs = [l for l in model.layers if isinstance(l, layers.Conv2D)]
    if not convs:
        raise ValueError("no Conv2D layers found")
    widest = max(l.filters for l in convs)
    return [l for l in convs if l.filters == widest][-1].name


def compute_gradcam(model, img_batch, layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_batch)
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_out[0] * pooled, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8
    return heatmap.numpy()


def overlay(image, heatmap, alpha=0.4):
    """Blend a [0,1] HxW heatmap over a [0,1] HxWx3 image (matplotlib, no cv2)."""
    hm = tf.image.resize(heatmap[..., None], image.shape[:2]).numpy()[..., 0]
    hm = plt.get_cmap("jet")(hm)[..., :3]
    return np.clip((1 - alpha) * image + alpha * hm, 0, 1)


def plot_gradcam(model, ds, out_path, n_per_class=4):
    """A grid of Grad-CAMs: n real and n fake, drawn from the first test batch.

    With mask_mode=face_only the heat should sit on the face; with
    background_only it cannot, which is the point of that control run.
    """
    layer_name = last_spatial_conv(model)
    images, labels = next(iter(ds))
    labels = labels.numpy()
    scores = model.predict(images, verbose=0).flatten()

    picks = []
    for cls in (0, 1):
        picks.extend(np.where(labels == cls)[0][:n_per_class])
    if not picks:
        print("gradcam: no samples in the first batch, skipping")
        return None

    cols = n_per_class
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.2 * rows),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    for ax, idx in zip(axes.flat, picks):
        img = images[idx].numpy()
        heat = compute_gradcam(model, images[idx][None], layer_name)
        ax.imshow(overlay(img, heat))
        truth = CLASSES[labels[idx]]
        pred = CLASSES[int(scores[idx] > 0.5)]
        mark = "ok" if truth == pred else "MISS"
        ax.set_title(f"{truth} -> {pred} {scores[idx]:.3f} [{mark}]", fontsize=9)
        ax.axis("off")

    fig.suptitle(f"Grad-CAM on '{layer_name}'")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return layer_name


# --------------------------------------------------------------------------- #
# Efficiency -- the numbers the resource claim actually rests on
# --------------------------------------------------------------------------- #

def _device_name():
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return "CPU"
    try:
        return tf.config.experimental.get_device_details(gpus[0]).get("device_name", "GPU")
    except Exception:
        return "GPU"


def _peak_memory_mb():
    if not tf.config.list_physical_devices("GPU"):
        return None
    try:
        return round(tf.config.experimental.get_memory_info("GPU:0")["peak"] / 1e6, 1)
    except Exception:
        return None


def count_macs(model):
    """Multiply-accumulates per image, counted analytically from layer shapes.

    Analytic rather than tf.profiler: the profiler does not reliably register
    FLOPs for tf.signal.fft2d, so an automated count can silently undercount the
    FFT branch. Conv2D and Dense dominate and their cost is exact arithmetic.

    MACs, NOT FLOPs: 1 MAC = 1 multiply + 1 add, so FLOPs ~= 2 x MACs. Papers
    use both and rarely say which -- always state the convention next to the
    number, or the comparison is meaningless.

    The fft2d transform itself is excluded: it is O(N log N) ~= 1.0e6 ops at
    256x256, under 0.1% of the conv cost, and is not a multiply-accumulate.
    """
    macs = 0
    for layer in model.layers:
        if isinstance(layer, layers.Conv2D):
            _, out_h, out_w, _ = layer.output.shape
            k_h, k_w = layer.kernel_size
            c_in = layer.input.shape[-1]
            macs += int(out_h) * int(out_w) * layer.filters * k_h * k_w * int(c_in)
        elif isinstance(layer, layers.Dense):
            macs += int(layer.input.shape[-1]) * layer.units
    return int(macs)


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL,
                                       cwd=os.path.dirname(os.path.abspath(__file__))
                                       ).decode().strip()
    except Exception:
        return None


def system_under_test(cfg, warmup, runs):
    """Everything a reader needs to judge whether a latency number is comparable.

    NeurIPS-style compute disclosure, and not optional here: Colab states its
    GPU types vary over time, so a latency figure without the GPU recorded next
    to it cannot be compared against anything -- including your own earlier run.
    """
    try:
        policy = tf.keras.mixed_precision.global_policy().name
    except Exception:
        policy = None
    return {
        "device": _device_name(),
        "tensorflow": tf.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "build_cuda": bool(tf.test.is_built_with_cuda()),
        "mixed_precision_policy": policy,
        "git_sha": _git_sha(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "latency_warmup": warmup,
        "latency_runs": runs,
        "img_size": cfg.img_size,
    }


def measure_efficiency(model, cfg, ds, warmup=50, runs=1000):
    """Params, file size, peak GPU memory, single-image latency, batch throughput.

    Three things this does differently from a naive timing loop, because the
    claim is a comparison and a sloppy number would not survive a question:

    1. Not model.predict(). predict() carries Keras dispatch, batching and
       callback machinery that dwarfs a 1M-param forward pass -- it measures the
       API, not the model. A tf.function over model(x, training=False) is the
       forward pass itself.
    2. .numpy() inside the timed region. GPU ops are queued asynchronously; stop
       the clock without forcing a sync and you have timed kernel launches.
    3. Median and p95, not just mean. GPU timings have a long right tail, so a
       mean over a short loop moves between sessions while the median does not.
       runs defaults to 1000: a p95 from 100 samples rests on 5 observations and
       is noise. Too few queries also bias latency OPTIMISTICALLY, which reads
       as cheating in your own favour -- the opposite of what you want here.

    Latency is the model only -- decode, resize and mask are excluded, since a
    comparison against a bigger detector is about the network. Comparable only
    against baselines measured on this same GPU, in this same session.
    """
    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
    except Exception:
        pass

    images, _ = next(iter(ds))
    infer = tf.function(lambda x: model(x, training=False))

    # single image: the latency number
    x1 = images[:1]
    for _ in range(warmup):
        infer(x1).numpy()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        infer(x1).numpy()
        times.append((time.perf_counter() - t0) * 1000.0)
    times = np.array(times)

    # full batch: the throughput number
    batch_size = int(images.shape[0])
    for _ in range(3):
        infer(images).numpy()
    t0 = time.perf_counter()
    for _ in range(10):
        infer(images).numpy()
    batch_seconds = (time.perf_counter() - t0) / 10.0

    macs = count_macs(model)
    eff = {
        "params": int(model.count_params()),
        "macs": macs,
        "macs_note": "conv+dense multiply-accumulates per image; FLOPs ~= 2x this",
        "system_under_test": system_under_test(cfg, warmup, runs),
        "model_file_mb": round(os.path.getsize(os.path.join(cfg.run_dir, "model.keras")) / 1e6, 2),
        "peak_gpu_memory_mb": _peak_memory_mb(),
        "latency_bs1_ms": {
            "median": round(float(np.median(times)), 3),
            "mean": round(float(times.mean()), 3),
            "p95": round(float(np.percentile(times, 95)), 3),
        },
        "throughput_img_per_s": round(batch_size / batch_seconds, 1),
        "throughput_batch_size": batch_size,
    }

    lat = eff["latency_bs1_ms"]
    sut = eff["system_under_test"]
    print(f"device      : {sut['device']}  (TF {sut['tensorflow']}, git {sut['git_sha']})")
    print(f"params      : {eff['params']:,}  ({eff['model_file_mb']} MB on disk)")
    print(f"MACs        : {macs / 1e9:.3f} G per image at {cfg.img_size}^2  "
          f"(FLOPs ~= {2 * macs / 1e9:.2f} G)")
    print(f"peak memory : {eff['peak_gpu_memory_mb']} MB")
    print(f"latency bs=1: {lat['median']:.2f} ms median | {lat['mean']:.2f} mean "
          f"| {lat['p95']:.2f} p95   (n={runs})")
    print(f"throughput  : {eff['throughput_img_per_s']:,.1f} img/s at batch {batch_size}")
    return eff


# --------------------------------------------------------------------------- #

def evaluate(cfg):
    model = load_run_model(cfg)
    ds = build_datasets(cfg)

    print(f"=== eval {cfg.name} | mask_mode={cfg.mask_mode} ===")

    # first, on a clean device: peak memory should reflect inference, not the
    # GradientTape that Grad-CAM allocates later
    eff = measure_efficiency(model, cfg, ds["test"])

    y_true, y_score = predict_split(model, ds["test"])
    y_pred = (y_score > 0.5).astype(int)

    cm = plot_confusion(y_true, y_pred, os.path.join(cfg.run_dir, "confusion_matrix.png"))
    roc_auc = plot_roc(y_true, y_score, os.path.join(cfg.run_dir, "roc.png"))
    layer_name = plot_gradcam(model, ds["test"], os.path.join(cfg.run_dir, "gradcam.png"))

    report = classification_report(y_true, y_pred, target_names=CLASSES,
                                   output_dict=True, zero_division=0)
    print(classification_report(y_true, y_pred, target_names=CLASSES,
                                zero_division=0))
    print(f"test AUC: {roc_auc:.4f}")

    out = {
        "name": cfg.name,
        "mask_mode": cfg.mask_mode,
        "test_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "classes": CLASSES,
        "report": report,
        "gradcam_layer": layer_name,
        "efficiency": eff,
    }
    with open(os.path.join(cfg.run_dir, "eval.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"wrote figures + eval.json to {cfg.run_dir}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="path to a configs/*.yaml")
    args = ap.parse_args()
    evaluate(load_config(args.config))


if __name__ == "__main__":
    main()
