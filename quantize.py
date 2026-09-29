#!/usr/bin/env python3
"""Post-training int8 quantization of a trained model, and the measurements that
make it comparable: TFLite file size, CPU latency (fp32 vs int8, same
interpreter, same threads), test AUC/accuracy for both, optional held-out AUC.

    python quantize.py --config configs/test19_sg2_crop_no_fft.yaml
    (three precisions per model: fp32, int8_dynamic = weights only, int8 = full integer)
    python quantize.py --config configs/test20_crop_baselines.yaml --model efficientnet_b0
    python quantize.py --config ... --fake-dir "/content/drive/MyDrive/Fake(SG3-T-psi1)" --tag sg3t
    python quantize.py --config ... --precisions fp32     # TFLite CPU latency only, no quantization

Why this file: "why not compress EfficientNet, which generalises?" is a fair
panel question. The answer has to be measured at equal precision for all four
models, not argued. No retraining: TFLite full-integer quantization with a
representative sample of the test split (Jacob et al., CVPR 2018). Weights and
activations go to int8; input/output stay float so the pipeline is unchanged.
Writes experiments/<run>/eval_quant.json and the two .tflite files next to it.

CPU only, deliberately: int8 is a CPU/NPU story; the GPU numbers in the README
are fp32 and stay the comparison there.
"""

import argparse
import json
import os
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from model import (augment_and_mask, build_datasets, cached_dataset, feathered_ellipse,
                   load_config, load_paths)

N_REP = 200          # representative images for calibration
N_LAT = 300          # latency runs per precision


PRECISIONS = ("fp32", "int8_dynamic", "int8")
# fp32:         the float model through TFLite -- the CPU runtime a device would use
# int8_dynamic: weights int8, activations float (no calibration). The 4x size cut
#               and a speedup on matmul-heavy layers; accuracy essentially intact.
# int8:         full integer, calibrated on 200 test images; fastest on int8
#               hardware, but post-training it can break models with
#               squeeze-excite / hard-swish (EfficientNet collapsed to 0.5).


def cpu_info():
    """CPU model and the vector extensions XNNPACK cares about. MobileNet's TFLite
    latency read 1.30 ms on one Colab VM and 6.62 on another (our models moved 1%):
    depthwise kernels depend on the ISA, plain convs do not. Recorded from now on."""
    info = {"model": "?", "flags": []}
    try:
        txt = open("/proc/cpuinfo").read()
        for line in txt.splitlines():
            if line.startswith("model name"):
                info["model"] = line.split(":", 1)[1].strip(); break
        flags = next((l.split(":", 1)[1].split() for l in txt.splitlines() if l.startswith("flags")), [])
        info["flags"] = sorted(f for f in flags if f.startswith(("avx", "fma", "sse4", "f16c")))
    except OSError:
        import platform
        info["model"] = platform.processor() or platform.machine()
    return info


def convert(model, rep_images, precision):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if precision != "fp32":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
    if precision == "int8":
        conv.representative_dataset = lambda: ([x[None].astype(np.float32)] for x in rep_images)
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.float32
        conv.inference_output_type = tf.float32
    return conv.convert()


class Runner:
    def __init__(self, tflite_bytes, threads):
        self.delegate = "xnnpack"
        try:
            self.interp = tf.lite.Interpreter(model_content=tflite_bytes, num_threads=threads)
            self.interp.allocate_tensors()
        except RuntimeError as e:
            # XNNPACK refuses some quantized ops (MobileNetV3's hard-swish); the
            # reference kernels run everything, slower. Recorded, not hidden.
            print(f"  XNNPACK delegate failed ({str(e).strip().splitlines()[-1]}); using builtin kernels")
            self.delegate = "builtin"
            self.interp = tf.lite.Interpreter(
                model_content=tflite_bytes, num_threads=threads,
                experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
            self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]["index"]
        self.out = self.interp.get_output_details()[0]["index"]

    def __call__(self, x):                      # x: (1, H, W, 3) float32
        self.interp.set_tensor(self.inp, x)
        self.interp.invoke()
        return float(self.interp.get_tensor(self.out).ravel()[0])


def latency_ms(runner, x, warmup=20, runs=N_LAT):
    for _ in range(warmup):
        runner(x)
    t = []
    for _ in range(runs):
        t0 = time.perf_counter(); runner(x); t.append((time.perf_counter() - t0) * 1e3)
    t = np.array(t)
    return {"median": round(float(np.median(t)), 2), "p95": round(float(np.percentile(t, 95)), 2), "runs": runs}


def score(runner, ds, limit=None):
    y_true, y_score, n = [], [], 0
    for xb, yb in ds:
        for x, y in zip(xb.numpy(), yb.numpy()):
            y_score.append(runner(x[None].astype(np.float32))); y_true.append(int(y)); n += 1
            if limit and n >= limit:
                break
        if limit and n >= limit:
            break
    y_true, y_score = np.array(y_true), np.array(y_score)
    return {"auc": round(float(roc_auc_score(y_true, y_score)), 4),
            "accuracy": round(float(((y_score > 0.5) == (y_true == 1)).mean()), 4), "n": int(n)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", help="score this baseline under experiments/<name>/baselines/<model>/")
    ap.add_argument("--fake-dir", help="also score a held-out generator (heldout.py's sampling)")
    ap.add_argument("--tag", default="heldout")
    ap.add_argument("--n-eval", type=int, default=0, help="limit test images (0 = all 7,500)")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--precisions", nargs="+", default=list(PRECISIONS), choices=list(PRECISIONS),
                    help="which to measure; e.g. --precisions fp32 for the TFLite CPU latency alone")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_dir = os.path.join(cfg.run_dir, "baselines", args.model) if args.model else cfg.run_dir
    name = f"{cfg.name}/baselines/{args.model}" if args.model else cfg.name
    model = tf.keras.models.load_model(os.path.join(run_dir, "model.keras"), safe_mode=False)
    ds = build_datasets(cfg)["test"]
    rep = np.concatenate([xb.numpy() for xb, _ in ds.take(max(1, N_REP // cfg.batch_size + 1))])[:N_REP]
    ci = cpu_info()
    print(f"=== quantize {name} | {model.count_params():,} params | {len(rep)} calibration images | {args.threads} threads | {ci['model']} | {' '.join(ci['flags'])} ===")

    out = {"model": name, "threads": args.threads, "params": int(model.count_params()),
           "cpu": cpu_info(), "precision": {}}
    x1 = rep[:1].astype(np.float32)
    for prec in args.precisions:
        blob = convert(model, rep, prec)
        path = os.path.join(run_dir, f"model_{prec}.tflite")
        with open(path, "wb") as f:
            f.write(blob)
        r = Runner(blob, args.threads)
        rec = {"file_mb": round(len(blob) / 1e6, 2), "delegate": r.delegate,
               "cpu_latency_ms": latency_ms(r, x1), "test": score(r, ds, args.n_eval or None)}
        if args.fake_dir:
            from heldout import pick
            used = {p for paths, _ in load_paths(cfg).values() for p in paths}
            real = pick(cfg.real_dir, 1500, used, cfg.seed); fake = pick(args.fake_dir, 1500, used, cfg.seed)
            n = min(len(real), len(fake)); real, fake = real[:n], fake[:n]
            mask = feathered_ellipse(cfg.img_size, cfg.mask_rx, cfg.mask_ry, cfg.mask_feather)
            hds = cached_dataset(real + fake, [0] * n + [1] * n, cfg, f"heldout_{args.tag}")
            hds = hds.map(lambda x, y: augment_and_mask(x, y, cfg, mask, training=False)).batch(32)
            rec[f"heldout_{args.tag}"] = score(r, hds)
        out["precision"][prec] = rec
        h = rec.get(f"heldout_{args.tag}", {})
        print(f"{prec}: {rec['file_mb']} MB | CPU {rec['cpu_latency_ms']['median']} ms (p95 {rec['cpu_latency_ms']['p95']}, {r.delegate}) | "
              f"test AUC {rec['test']['auc']} acc {rec['test']['accuracy']} (n={rec['test']['n']})"
              + (f" | {args.tag} AUC {h['auc']}" if h else ""))

    with open(os.path.join(run_dir, "eval_quant.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {run_dir}/eval_quant.json")


if __name__ == "__main__":
    main()
