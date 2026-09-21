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
                             roc_auc_score, roc_curve)
from tensorflow.keras import layers

from model import build_datasets, feathered_ellipse, load_config

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
    convs = [l for l in _iter_layers(model)
             if isinstance(l, (layers.Conv2D, layers.SeparableConv2D))]
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
# Shortcut checks -- what does THIS trained model actually use?
# --------------------------------------------------------------------------- #
# Retraining with face_only / background_only says where signal exists in the
# DATA. The panel asked what the trained model USES. These two run on the
# trained model at test time, no retraining, and answer that directly.

def _collect(ds, n_per_class):
    """Up to n test images per class, uint8 (N,H,W,3). Only meaningful when the
    split is unmasked (mask_mode none) -- a masked image has no background."""
    reals, fakes = [], []
    for images, labels in ds:
        imgs = tf.cast(tf.clip_by_value(images, 0, 1) * 255, tf.uint8).numpy()
        for img, y in zip(imgs, labels.numpy()):
            (fakes if y == 1 else reals).append(img)
        if len(reals) >= n_per_class and len(fakes) >= n_per_class:
            break
    n = min(len(reals), len(fakes), n_per_class)
    return np.stack(reals[:n]), np.stack(fakes[:n])


def _predict_uint8(model, x, batch=64):
    out = []
    for i in range(0, len(x), batch):
        xb = tf.cast(x[i:i + batch], tf.float32) / 255.0
        out.append(model(xb, training=False).numpy().ravel())
    return np.concatenate(out)


def swap_test(model, ds, face_mask, n_per_class=500):
    """Does the prediction follow the face or the background?

    Composites via the same feathered ellipse the mask experiments use:
    face pixels from one image, everything outside the ellipse from another.

      swap     real face + fake bg, fake face + real bg   -> the question
      control  real face + other REAL bg, fake + other FAKE bg -> the seam check

    The control has identical seams and no conflict. If control_auc stays near
    the plain test AUC, compositing is benign and swap_auc means what it says.
    If control_auc collapses, the seam dominates and swap_auc is uninformative
    -- the test reports that about itself rather than hiding it.

    On the swap set, face label and background label are exact complements, so
    one AUC (scored by FACE label) says it all: well above 0.5 = follows the
    face; well below 0.5 = follows the background; ~0.5 = torn, uses both.

    "Background" = outside the ellipse: hair, ears, shoulders, clothes, wall.
    """
    reals, fakes = _collect(ds, n_per_class)
    n = len(reals)
    if n < 20:
        return {"skipped": f"only {n} per class collected"}
    m = face_mask.numpy()                                     # (H, W, 1) in [0, 1]
    r, f = reals.astype(np.float32), fakes.astype(np.float32)
    other = lambda a: np.roll(a, 1, axis=0)                   # a different image, same class
    comp = lambda face, bg: np.clip(face * m + bg * (1 - m), 0, 255).astype(np.uint8)

    p = {
        "real_face_real_bg": _predict_uint8(model, comp(r, other(r))),   # control
        "fake_face_fake_bg": _predict_uint8(model, comp(f, other(f))),   # control
        "real_face_fake_bg": _predict_uint8(model, comp(r, f)),          # swap
        "fake_face_real_bg": _predict_uint8(model, comp(f, r)),          # swap
    }
    y = [0] * n + [1] * n
    control_auc = roc_auc_score(y, np.concatenate([p["real_face_real_bg"], p["fake_face_fake_bg"]]))
    swap_auc_by_face = roc_auc_score(y, np.concatenate([p["real_face_fake_bg"], p["fake_face_real_bg"]]))
    return {
        "n_per_class": n,
        "control_auc_same_class_composites": round(float(control_auc), 4),
        "swap_auc_scored_by_face_label": round(float(swap_auc_by_face), 4),
        "mean_p_fake": {k: round(float(v.mean()), 4) for k, v in p.items()},
        "read": ("control near test AUC => seams benign; then swap >0.5 follows face, "
                 "<0.5 follows background, ~0.5 uses both. background = outside the ellipse."),
    }


def attention_in_face(model, ds, face_mask, layer_name, max_images=6000):
    """Fraction of Grad-CAM heat inside the face ellipse, over the test set.

    gradcam.png is eight pictures; this is the number. Heat is taken toward
    the PREDICTED class (p for predicted-fake, 1-p for predicted-real), so it
    is "evidence the model used", not "evidence for fake".

    Read against uniform_baseline: the ellipse's share of the image (~0.55). A
    model looking everywhere equally scores that; well above it means
    attention on the face. Spatial branch only -- the FFT branch has no
    image-space heatmap, so its share of the decision is not measured here.
    """
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output])
    m = face_mask[..., 0]                                     # (H, W)
    hw = [int(m.shape[0]), int(m.shape[1])]
    frac = {0: [], 1: []}
    seen = 0
    # The tape keeps every activation of the backbone for the whole chunk.
    # 144 x 256^2 through EfficientNet-B0 does not fit on 24 GB; 16 does.
    chunk = 16
    for images, labels in ds:
        for i in range(0, int(images.shape[0]), chunk):
            xb, yb = images[i:i + chunk], labels[i:i + chunk]
            with tf.GradientTape() as tape:
                conv, preds = grad_model(xb, training=False)
                pf = preds[:, 0]
                score = tf.where(pf > 0.5, pf, 1.0 - pf)      # toward the predicted class
            grads = tape.gradient(score, conv)                # (b, h, w, C)
            pooled = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
            heat = tf.nn.relu(tf.reduce_sum(conv * pooled, axis=-1))   # (b, h, w)
            heat = tf.image.resize(heat[..., None], hw)[..., 0]        # (b, H, W)
            total = tf.reduce_sum(heat, axis=(1, 2))
            inside = tf.reduce_sum(heat * m, axis=(1, 2))
            ok = total > 1e-6                                 # skip all-zero maps
            f = (inside / tf.where(ok, total, 1.0)).numpy()
            for fi, oki, y in zip(f, ok.numpy(), yb.numpy()):
                if oki:
                    frac[int(y)].append(float(fi))
            seen += len(f)
        if seen >= max_images:
            break
    return {
        "layer": layer_name,
        "n": seen,
        "uniform_baseline": round(float(tf.reduce_mean(m)), 4),
        "real": round(float(np.mean(frac[0])), 4) if frac[0] else None,
        "fake": round(float(np.mean(frac[1])), 4) if frac[1] else None,
        "read": "share of Grad-CAM heat inside the face ellipse; compare to uniform_baseline. spatial branch only.",
    }


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


def _cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


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
    for layer in _iter_layers(model):
        if isinstance(layer, layers.Dense):
            macs += int(layer.input.shape[-1]) * layer.units
            continue
        if not isinstance(layer, (layers.Conv2D, layers.DepthwiseConv2D,
                                  layers.SeparableConv2D)):
            continue
        _, out_h, out_w, _ = layer.output.shape
        hw = int(out_h) * int(out_w)
        k_h, k_w = layer.kernel_size
        c_in = int(layer.input.shape[-1])
        # order matters: SeparableConv2D and DepthwiseConv2D are NOT Conv2D
        # subclasses in Keras 3, but check the specific ones first anyway
        if isinstance(layer, layers.SeparableConv2D):
            dm = layer.depth_multiplier
            macs += hw * c_in * dm * k_h * k_w            # depthwise
            macs += hw * c_in * dm * layer.filters         # pointwise 1x1
        elif isinstance(layer, layers.DepthwiseConv2D):
            macs += hw * c_in * layer.depth_multiplier * k_h * k_w
        else:                                              # plain Conv2D
            macs += hw * layer.filters * k_h * k_w * c_in // int(layer.groups or 1)
    return int(macs)


def _iter_layers(model):
    """Layers, descending into nested models -- Keras applications wrapped in
    a head are a Model inside a Model."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            yield from _iter_layers(layer)
        else:
            yield layer


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
        "cpu": _cpu_name(),
        "cpu_threads": os.cpu_count(),
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
    def reset_peak():
        try:
            tf.config.experimental.reset_memory_stats("GPU:0")
        except Exception:
            pass

    images, _ = next(iter(ds))
    infer = tf.function(lambda x: model(x, training=False))

    # single image: the latency number, and the memory number that matters for
    # a "how much does it need to run" claim
    reset_peak()
    x1 = images[:1]
    for _ in range(warmup):
        infer(x1).numpy()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        infer(x1).numpy()
        times.append((time.perf_counter() - t0) * 1000.0)
    times = np.array(times)
    peak_bs1 = _peak_memory_mb()

    # full batch: the throughput number; its peak is activations x batch, not
    # the model's footprint
    reset_peak()
    batch_size = int(images.shape[0])
    for _ in range(3):
        infer(images).numpy()
    t0 = time.perf_counter()
    for _ in range(10):
        infer(images).numpy()
    batch_seconds = (time.perf_counter() - t0) / 10.0
    peak_batch = _peak_memory_mb()

    # CPU, bs=1: the second hardware leg. The GPU ranking favours few large
    # kernels; a CPU is where MobileNet's 17x fewer MACs should finally count.
    # If the order flips here, that is the finding. Plain TF on CPU, no XLA
    # (state it). Fewer runs than the GPU loop -- Xception at 256^2 is
    # hundreds of ms per image on a Colab CPU -- so p95 rests on ~15 samples;
    # the median is the number to quote.
    cpu_runs = 300
    with tf.device("/CPU:0"):
        x1_cpu = tf.identity(x1)
        infer_cpu = tf.function(lambda x: model(x, training=False))
        for _ in range(10):
            infer_cpu(x1_cpu).numpy()
        cpu_times = []
        for _ in range(cpu_runs):
            t0 = time.perf_counter()
            infer_cpu(x1_cpu).numpy()
            cpu_times.append((time.perf_counter() - t0) * 1000.0)
    cpu_times = np.array(cpu_times)

    macs = count_macs(model)
    eff = {
        "params": int(model.count_params()),
        "macs": macs,
        "macs_note": "conv+dense multiply-accumulates per image; FLOPs ~= 2x this",
        "system_under_test": system_under_test(cfg, warmup, runs),
        # the inference artifact: params x 4 bytes (fp32). model.keras on disk is
        # ~3x that -- it carries Adam's two moment tensors for resuming training.
        "weights_mb_fp32": round(int(model.count_params()) * 4 / 1e6, 2),
        "checkpoint_file_mb": round(os.path.getsize(os.path.join(cfg.run_dir, "model.keras")) / 1e6, 2),
        "peak_gpu_memory_mb": {"bs1": peak_bs1, f"bs{batch_size}": peak_batch},
        "latency_bs1_ms": {
            "median": round(float(np.median(times)), 3),
            "mean": round(float(times.mean()), 3),
            "p95": round(float(np.percentile(times, 95)), 3),
        },
        "throughput_img_per_s": round(batch_size / batch_seconds, 1),
        "throughput_batch_size": batch_size,
        "latency_bs1_ms_cpu": {
            "median": round(float(np.median(cpu_times)), 2),
            "mean": round(float(cpu_times.mean()), 2),
            "p95": round(float(np.percentile(cpu_times, 95)), 2),
            "runs": cpu_runs,
            "note": "plain TF on CPU, no XLA; all threads",
        },
    }

    lat = eff["latency_bs1_ms"]
    sut = eff["system_under_test"]
    print(f"device      : {sut['device']}  (TF {sut['tensorflow']}, git {sut['git_sha']})")
    print(f"params      : {eff['params']:,}  ({eff['weights_mb_fp32']} MB fp32 weights; "
          f"checkpoint {eff['checkpoint_file_mb']} MB incl. optimizer)")
    print(f"MACs        : {macs / 1e9:.3f} G per image at {cfg.img_size}^2  "
          f"(FLOPs ~= {2 * macs / 1e9:.2f} G)")
    print(f"peak memory : {peak_bs1} MB at bs=1 | {peak_batch} MB at bs={batch_size}")
    print(f"latency bs=1: {lat['median']:.2f} ms median | {lat['mean']:.2f} mean "
          f"| {lat['p95']:.2f} p95   (n={runs})")
    print(f"throughput  : {eff['throughput_img_per_s']:,.1f} img/s at batch {batch_size}")
    c = eff["latency_bs1_ms_cpu"]
    print(f"latency CPU : {c['median']:.1f} ms median | {c['p95']:.1f} p95   "
          f"(n={cpu_runs}, {sut['cpu_threads']} threads, {sut['cpu']})")
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

    face_mask = feathered_ellipse(cfg.img_size, cfg.mask_rx, cfg.mask_ry, cfg.mask_feather)
    if cfg.input_mode == "crop":
        note = "input_mode=crop: a 256^2 native patch is not a whole face; the ellipse does not apply"
        attention, swap = None, {"skipped": note}
    else:
        attention = attention_in_face(model, ds["test"], face_mask, layer_name) if layer_name else None
        if cfg.mask_mode == "none":
            swap = swap_test(model, ds["test"], face_mask)
        else:
            swap = {"skipped": f"mask_mode={cfg.mask_mode}: masked images have no background to swap"}
    print("shortcut checks:")
    print(f"  attention in face : real {attention['real']}  fake {attention['fake']}  "
          f"(uniform {attention['uniform_baseline']})" if attention else "  attention: n/a")
    if "skipped" not in swap:
        print(f"  swap   control AUC {swap['control_auc_same_class_composites']}  "
              f"swap-by-face AUC {swap['swap_auc_scored_by_face_label']}")
    else:
        print(f"  swap: {swap['skipped']}")

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
        "shortcut_checks": {"attention_in_face": attention, "background_swap": swap},
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
