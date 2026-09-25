#!/usr/bin/env python3
"""Thesis figures, regenerated from the record (history.csv, eval.json,
eval_heldout_*.json on Drive) so no number is hand-copied from a table.

    python figures.py --experiments /content/drive/MyDrive/Research/experiments --out figures

Writes, into --out:
  curves_<run>.png         train/val accuracy + AUC per epoch, best epoch marked,
                           one per run that has a history.csv (--runs to restrict)
  curves_overlay.png       val AUC of the named runs on one axis (--overlay)
  efficiency_scatter.png   GPU latency vs test AUC, one point per model
  efficiency_bars.png      MACs / peak memory / CPU latency, one bar per model
  heldout_bars.png         held-out AUC per model per tag (eval_heldout_*.json)
  seeds.png                test AUC per seed for each stem, mean line

No GPU, no TensorFlow: a minute on CPU. Runs that are missing a file are
skipped with a note, never faked.
"""

import argparse
import csv
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_OVERLAY = ["test19_sg2_crop_no_fft", "test21_stride2_stem", "test19_sg2_crop", "test17_sg2"]
EFFICIENCY_ROWS = [  # (run folder relative to experiments/, label)
    ("test19_sg2_crop_no_fft", "this model (headline)"),
    ("test21_stride2_stem", "this model, stride-2 stem"),
    ("test19_sg2_crop", "this model, +FFT"),
    ("test20_crop_baselines/baselines/mobilenet_v3_small", "MobileNetV3-Small"),
    ("test20_crop_baselines/baselines/efficientnet_b0", "EfficientNet-B0"),
    ("test20_crop_baselines/baselines/xception", "Xception"),
]
SEED_STEMS = {"stride 1 (headline)": "test19_sg2_crop_no_fft", "stride 2": "test21_stride2_stem"}


def read_history(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    h = {k: [float(r[k]) for r in rows] for k in rows[0]}
    h["epoch"] = [int(e) + 1 for e in h["epoch"]]            # Keras counts from 0
    return h


def read_json(path):
    with open(path) as f:
        return json.load(f)


def curves(run_dir, out):
    hp = os.path.join(run_dir, "history.csv")
    if not os.path.exists(hp) or os.path.getsize(hp) == 0:
        print(f"  skip curves: no history in {run_dir}")
        return
    h = read_history(hp)
    parts = run_dir.rstrip("/").split(os.sep)
    # a baseline lives at <headline>/baselines/<model>: name it by both, or the
    # resize-pipeline and crop-pipeline baselines overwrite each other
    name = f"{parts[-3]}__{parts[-1]}" if len(parts) >= 3 and parts[-2] == "baselines" else parts[-1]
    best = max(range(len(h["val_auc"])), key=lambda i: h["val_auc"][i])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, key, title in ((ax[0], "accuracy", "accuracy"), (ax[1], "auc", "AUC")):
        a.plot(h["epoch"], h[key], label="train")
        a.plot(h["epoch"], h["val_" + key], label="val")
        a.axvline(h["epoch"][best], color="grey", ls="--", lw=0.8,
                  label=f"best val AUC {h['val_auc'][best]:.4f} @ epoch {h['epoch'][best]}")
        a.set_xlabel("epoch"); a.set_title(f"{name}: {title}"); a.grid(alpha=0.3); a.legend()
    ax[1].set_ylim(max(0.45, min(h["val_auc"]) - 0.02), 1.001)
    fig.tight_layout()
    p = os.path.join(out, f"curves_{name}.png"); fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  wrote {p}")


def overlay(experiments, names, out):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    n = 0
    for name in names:
        hp = os.path.join(experiments, name, "history.csv")
        if not os.path.exists(hp) or os.path.getsize(hp) == 0:
            print(f"  overlay: no history for {name}, skipped"); continue
        h = read_history(hp); ax.plot(h["epoch"], h["val_auc"], label=name); n += 1
    if not n:
        plt.close(fig); return
    ax.set_xlabel("epoch"); ax.set_ylabel("val AUC"); ax.set_ylim(0.45, 1.001); ax.grid(alpha=0.3)
    ax.set_title("validation AUC per epoch"); ax.legend()
    fig.tight_layout(); p = os.path.join(out, "curves_overlay.png"); fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  wrote {p}")


def efficiency(experiments, out):
    rows = []
    for rel, label in EFFICIENCY_ROWS:
        ep = os.path.join(experiments, rel, "eval.json")
        if not os.path.exists(ep):
            print(f"  efficiency: no eval.json for {rel}, skipped"); continue
        e = read_json(ep); eff = e["efficiency"]
        rows.append({"label": label, "auc": e["test_auc"], "params": eff["params"], "macs": eff["macs"],
                     "gpu_ms": eff["latency_bs1_ms"]["median"],
                     "cpu_ms": eff.get("latency_bs1_ms_cpu", {}).get("median"),
                     "mem": eff["peak_gpu_memory_mb"]["bs1"]})
    if not rows:
        return
    # scatter: latency vs AUC, marker area ~ params
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in rows:
        ax.scatter(r["gpu_ms"], r["auc"], s=30 + r["params"] / 4e4, alpha=0.75)
        ax.annotate(f"{r['label']}\n{r['params']/1e6:.2f}M", (r["gpu_ms"], r["auc"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("GPU latency, ms per image at batch 1 (L4)"); ax.set_ylabel("test AUC (StyleGAN2, crop pipeline)")
    ax.set_ylim(min(r["auc"] for r in rows) - 0.002, 1.0005); ax.grid(alpha=0.3)
    ax.set_title("accuracy vs latency; marker area ~ parameters")
    fig.tight_layout(); p = os.path.join(out, "efficiency_scatter.png"); fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  wrote {p}")
    # bars
    keys = [("macs", "MACs (G)", 1e-9), ("mem", "peak GPU MB @ bs=1", 1), ("cpu_ms", "CPU ms @ bs=1", 1), ("gpu_ms", "GPU ms @ bs=1", 1)]
    fig, ax = plt.subplots(1, len(keys), figsize=(4 * len(keys), 4))
    labels = [r["label"] for r in rows]
    for a, (k, title, scale) in zip(ax, keys):
        vals = [(r[k] or 0) * scale for r in rows]
        a.barh(labels, vals); a.set_title(title); a.invert_yaxis(); a.grid(alpha=0.3, axis="x")
        for i, v in enumerate(vals):
            a.text(v, i, f" {v:.2f}" if v < 10 else f" {v:.0f}", va="center", fontsize=8)
    for a in ax[1:]:
        a.set_yticklabels([])
    fig.tight_layout(); p = os.path.join(out, "efficiency_bars.png"); fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  wrote {p}")


def heldout(experiments, out):
    found = {}   # tag -> {label: auc}
    for rel, label in EFFICIENCY_ROWS:
        for hp in glob.glob(os.path.join(experiments, rel, "eval_heldout_*.json")):
            h = read_json(hp); found.setdefault(h["tag"], {})[label] = h["auc"]
    if not found:
        print("  heldout: no eval_heldout_*.json found, skipped"); return
    tags = sorted(found); labels = [l for _, l in EFFICIENCY_ROWS if any(l in found[t] for t in tags)]
    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 2, 4.5))
    w = 0.8 / len(tags)
    for j, t in enumerate(tags):
        vals = [found[t].get(l, 0) for l in labels]
        xs = [i + (j - (len(tags) - 1) / 2) * w for i in range(len(labels))]
        ax.bar(xs, vals, width=w, label=t)
        for x, v in zip(xs, vals):
            if v: ax.text(x, v + 0.005, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.axhline(0.5, color="grey", ls=":", lw=0.8); ax.set_ylim(0.4, 1.02); ax.set_ylabel("held-out AUC")
    ax.set_title("held-out generator / source"); ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); p = os.path.join(out, "heldout_bars.png"); fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  wrote {p}")


def seeds(experiments, out):
    fig, ax = plt.subplots(figsize=(6, 4))
    any_ = False
    for i, (label, base) in enumerate(SEED_STEMS.items()):
        pts = []
        for d in [base] + sorted(glob.glob(os.path.join(experiments, base + "_s*"))):
            d = d if os.path.isabs(d) else os.path.join(experiments, d)
            ep = os.path.join(d, "eval.json")
            if os.path.exists(ep):
                m = re.search(r"_s(\d+)$", d.rstrip("/")); seed = int(m.group(1)) if m else 42
                pts.append((seed, read_json(ep)["test_auc"]))
        if not pts:
            continue
        any_ = True
        ys = [a for _, a in pts]; mean = sum(ys) / len(ys)
        ax.scatter([i] * len(pts), ys, zorder=3)
        for s, a in pts:
            ax.annotate(f"s{s}", (i, a), textcoords="offset points", xytext=(8, -3), fontsize=8)
        ax.hlines(mean, i - 0.25, i + 0.25, color="k", lw=1.2)
        ax.text(i, mean, f"  mean {mean:.4f} (n={len(pts)})", va="bottom", ha="left", fontsize=8)
    if not any_:
        plt.close(fig); print("  seeds: nothing evaluated yet, skipped"); return
    ax.set_xticks(range(len(SEED_STEMS))); ax.set_xticklabels(list(SEED_STEMS)); ax.set_ylabel("test AUC")
    ax.set_title("test AUC per seed"); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); p = os.path.join(out, "seeds.png"); fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  wrote {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments", default="/content/drive/MyDrive/Research/experiments")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--runs", nargs="*", help="only these runs' curves (default: every run with a history.csv)")
    ap.add_argument("--overlay", nargs="*", default=DEFAULT_OVERLAY)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("training curves:")
    runs = args.runs or sorted(os.path.basename(os.path.dirname(p))
                               for p in glob.glob(os.path.join(args.experiments, "*", "history.csv")))
    for r in runs:
        curves(os.path.join(args.experiments, r), args.out)
    for p in sorted(glob.glob(os.path.join(args.experiments, "*", "baselines", "*", "history.csv"))):
        curves(os.path.dirname(p), args.out)
    overlay(args.experiments, args.overlay, args.out)
    print("comparison figures:")
    efficiency(args.experiments, args.out)
    heldout(args.experiments, args.out)
    seeds(args.experiments, args.out)


if __name__ == "__main__":
    main()
