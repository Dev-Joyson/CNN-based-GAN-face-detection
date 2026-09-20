#!/usr/bin/env python3
"""Baselines on the SAME task as the headline: same data, split, mask, input
size, GPU, timing code. Without that, a comparison is two numbers from two
different experiments.

    python baselines.py --config configs/test16_full.yaml --model xception
    python baselines.py --config configs/test16_full.yaml --model xception --train

Without --train: build the architecture (random weights -- latency, params,
MACs and memory do not depend on weights), measure efficiency, write
eval.json with that block only. Minutes. This half of the comparison needs no
training at all.

With --train: ImageNet weights, full fine-tune at a low LR through the same
train() loop the headline used (checkpoint, early stop, CSV, TensorBoard,
--resume), then the full evaluate() -- figures, shortcut checks, efficiency.

Outputs land under the headline's run folder, so one directory holds the model
and everything it is compared against:

    experiments/test16_full/baselines/xception/

Each model's ImageNet preprocessing is a LAYER INSIDE the model, not a change
to the data pipeline. That keeps build_datasets byte-identical for every model
(the fairness guarantee) and it means the preprocessing is timed as part of
the model, which is honest -- it is part of running it. The pipeline hands
out [0, 1]; the first layers convert to what each backbone was trained on.
"""

import argparse
import os
from dataclasses import replace

import tensorflow as tf
from tensorflow.keras import applications, layers, models

from model import build_datasets, compile_model, load_config, train

# name -> (constructor, layer(s) that map [0,1] to what the backbone expects)
# Xception: [-1, 1].  EfficientNet / MobileNetV3 in Keras 3: raw [0, 255], the
# model rescales internally.  ResNet50 wants caffe BGR mean-subtraction, which
# needs a Lambda; omitted -- it is the least informative of the four anyway.
MODELS = {
    "xception":          (applications.Xception,       lambda: [layers.Rescaling(2.0, offset=-1.0)]),
    "efficientnet_b0":   (applications.EfficientNetB0, lambda: [layers.Rescaling(255.0)]),
    "mobilenet_v3_small": (applications.MobileNetV3Small, lambda: [layers.Rescaling(255.0)]),
}


def build_baseline(name, cfg, pretrained):
    ctor, preprocess = MODELS[name]
    inp = layers.Input(shape=(cfg.img_size, cfg.img_size, 3))
    x = inp
    for layer in preprocess():
        x = layer(x)
    # input_tensor, not a nested Model: the backbone's layers land FLAT in this
    # graph, so evaluate.py's Grad-CAM can get_layer() the last conv and
    # count_macs sees every layer without descending. A nested Model breaks
    # both in Keras 3.
    backbone = ctor(include_top=False,
                    weights="imagenet" if pretrained else None,
                    input_tensor=x)
    backbone.trainable = True                              # full fine-tune
    x = backbone.output
    x = layers.GlobalAveragePooling2D()(x)
    # the same head as build_model, so the comparison is backbone vs backbone
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inp, out, name=name)


def baseline_cfgs(cfg, name, lr, batch, epochs):
    """(train config, eval config) for one baseline.

    Both put the run under the headline's folder. Training: lower lr (the
    backbone is pretrained) and a smaller batch (Xception at 256^2 will not
    fit 144 with gradients on 24 GB). Evaluation: the HEADLINE's batch, so the
    throughput number is at the same batch as the model it is compared to.
    The cache key holds neither batch nor lr, so the headline's cache is
    reused as-is.
    """
    tcfg = replace(cfg, name=f"{cfg.name}/baselines/{name}",
                   lr=lr, batch_size=batch, epochs=epochs)
    ecfg = replace(tcfg, batch_size=cfg.batch_size)
    return tcfg, ecfg


def efficiency_only(bcfg, name):
    """No training: build, save, measure. Latency does not depend on weights."""
    import json
    from evaluate import count_macs, measure_efficiency
    tf.keras.utils.set_random_seed(bcfg.seed)
    os.makedirs(bcfg.run_dir, exist_ok=True)
    model = compile_model(build_baseline(name, bcfg, pretrained=False), bcfg)
    model.summary(line_length=100)
    model.save(os.path.join(bcfg.run_dir, "model.keras"))  # measure_efficiency reads its size
    ds = build_datasets(bcfg)                              # headline's cache, test split only
    print(f"=== efficiency-only | {name} | untrained ===")
    eff = measure_efficiency(model, bcfg, ds["test"])
    out = {"name": bcfg.name, "baseline_of": bcfg.name.split("/")[0], "trained": False,
           "efficiency": eff}
    with open(os.path.join(bcfg.run_dir, "eval.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {bcfg.run_dir}/eval.json  (efficiency only; run with --train for accuracy)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="the HEADLINE config to compare against")
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--train", action="store_true", help="fine-tune from ImageNet, then evaluate")
    ap.add_argument("--eval-only", action="store_true",
                    help="re-run evaluate() on an already fine-tuned baseline (e.g. to add a measurement)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-4, help="fine-tune LR (headline uses 3e-4 from scratch)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tcfg, ecfg = baseline_cfgs(cfg, args.model, args.lr, args.batch, args.epochs)
    print(f"baseline {args.model} of {cfg.name} -> {tcfg.run_dir}")

    from evaluate import evaluate
    if args.eval_only:
        evaluate(ecfg)
        return
    if not args.train:
        efficiency_only(ecfg, args.model)
        return

    train(tcfg, resume=args.resume,
          build_fn=lambda c: build_baseline(args.model, c, pretrained=True))
    evaluate(ecfg)


if __name__ == "__main__":
    main()
