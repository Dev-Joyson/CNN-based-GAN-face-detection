# GAN-face detection: lightweight dual-branch CNN

Detects StyleGAN1/StyleGAN2 fake faces vs real FFHQ faces with a **1.02M parameter**
CNN. Both classes are native 1024² and travel the identical resize path. Both
generators are the same family; a held-out generator is still the real test.
Research angle: **efficiency** — high accuracy at low compute/latency.

## How it works

```
python train.py --config configs/test16_full.yaml
       │                 │
       │                 └── configs/*.yaml   ← the only thing you change per experiment
       ▼
    train.py  (20 lines: read config, call train)
       ▼
    model.py  ← all the code lives here
       ▼
  experiments/<name>/   model.keras · history.csv · metrics.json

python evaluate.py --config configs/test16_full.yaml
       ▼
  experiments/<name>/   confusion_matrix.png · roc.png · gradcam.png · eval.json
```

Only touch `model.py` to change the *method*.

| file | what it is |
|---|---|
| `model.py` | Config, data pipeline, FFT layer, model, training loop |
| `train.py` | Entry point: train (`--config`) |
| `evaluate.py` | Entry point: confusion matrix, ROC, Grad-CAM, efficiency numbers |
| `predict.py` | Entry point: classify one image — the panel demo. Uses the training preprocessing, so it cannot drift |
| `heldout.py` | Score a trained model on a generator or real source it never saw (`--fake-dir` / `--real-dir`); the generalisation test and the FFT branch's last exam |
| `baselines.py` | Entry point: Xception / EfficientNet-B0 / MobileNetV3-Small on the identical task (`--model`, `--train`) |
| `cache_sync.py` | Copy a config's tf.data caches to/from Drive, so a reclaimed VM costs minutes, not a 2-hour rebuild |
| `distill.py` | Knowledge distillation; `train.py` dispatches here for any config with a `distill:` block (`configs/test18_distill.yaml`) |
| `audit_dataset.py` | Checks whether the two folders are separable by metadata alone |
| `configs/*.yaml` | Per-experiment settings |
| `tests/` | Three guard tests (see below) |
| `conftest.py` | Puts the repo root on `sys.path` so bare `pytest tests -q` can import `model` |
| `requirements-dev.txt` | Laptop install (CPU-only) for tests + audit; `requirements.txt` is the training env |
| `docs/` | `learnings.md` — one line per thing we found out; `hyperparameters.md` — where every config value came from and whether it was tuned; `research-repo-practice.md` — sourced notes on how this repo should be run |
| `notebooks/` | One file, `colab_runner.ipynb`. It exists because the VS Code Colab extension only activates `onNotebook` — no notebook open, no kernel picker, no `Colab: Open Terminal`, no Drive mount. It holds the connection and the launch commands, never method code. |

## Architecture

```
input (256, 256, 3)
├── spatial branch : Conv+Pool 32→64→128→256→256 → GAP      (256 feats)
└── FFT branch     : grayscale → fft2d(H,W) → |·| → fftshift
                     → log1p → norm → Conv 16 → Conv 32 → GAP  (32 feats)
                     ↓ concat → Dense 128 → Dropout → sigmoid
```

Two branches because GAN artifacts appear in two places: visual tells, and the
checkerboard pattern upsampling leaves in the spectrum. Output: `0` real, `1` fake.

## Two things to understand

**The mask exists to answer one question from the panel.** The model is trained
and reported on full images (`mask_mode: none`). At the progress review the panel
head raised that real and fake *backgrounds* might be separable enough for a CNN
to classify on them alone. The two masked runs are the controlled answer:

| `mask_mode` | model sees | role |
|---|---|---|
| `none` | everything | **the headline** — `test16_full` |
| `face_only` | face only, background zeroed | control: does the face alone carry the signal? |
| `background_only` | background only, face zeroed | control: does the background alone? |

| face_only | background_only | reading |
|---|---|---|
| high | low | signal is in the face — concern dismissed |
| high | high | face works, but background *also* leaks: the dataset has a shortcut, the model isn't dependent on it |
| low | high | the model was reading background — the panel was right |

One feathered oval, applied to train, val and test identically. This is load-bearing.

**Cache placement makes training fast.**

```
read → resize → uint8  ──► CACHE ──►  shuffle → augment → mask → batch
      (slow, runs once)               (fast, every epoch)
```

Caching *before* augment/mask means augmentation stays random each epoch, and
switching `mask_mode` reuses the cache for free. The cache key holds everything
that changes the cached bytes — dataset folders, `limit_per_class`, both resize
sizes, and the `seed` (which picks the split) — so no config change can read
another config's images by accident.

## Where to run what

| task | where | why |
|---|---|---|
| edit, `pytest tests -q`, `audit_dataset.py` | laptop, CPU | seconds, no GPU wanted; `pip install -r requirements-dev.txt` |
| training, `evaluate.py`, baselines | Colab GPU | the dataset is in Drive and the runs are hours long |

Do **not** train on an Apple Silicon laptop. A MacBook Air is fanless and throttles
on sustained load, `tensorflow-metal`'s op coverage is weakest exactly where this
model is unusual (`tf.signal.fft2d`), and 16 GB shared memory forces the batch size
down, which makes the run non-comparable to the Colab ones.

**The hard rule:** every latency number — this model and every baseline — must come
from one GPU in one session. Mixing hardware voids the comparison, and training
locally is mostly a way to get tempted into it.

## Running (Colab)

**1. Connect.** Open `notebooks/colab_runner.ipynb` → Select Kernel → Colab →
New Colab Server → GPU. Run the notebook's first sections: GPU check, mount Drive
(approve the popup), clone, `pytest`. About a minute.

**2. Keep the Mac awake.** In a local terminal, before anything long:

```bash
caffeinate -i
```

A sleeping laptop is the most common way to lose a Colab session.

**3. Train from a terminal, not a cell.** Cmd+Shift+P → `Colab: Open Terminal`
(this shell runs on the Colab VM, not your Mac):

```bash
cd /content/CNN-based-GAN-face-detection
nohup python -u train.py --config configs/test16_full.yaml > train.log 2>&1 &
tail -f train.log          # watch; Ctrl+C stops the watching, not the training
```

`nohup … &` hands the process to the VM; `-u` keeps the log live (Python
buffers stdout when it goes to a file — without `-u`, `tail` shows nothing for ages). Close the notebook, lose the network,
shut the lid for a while — it keeps running. Come back, open the terminal again,
`tail -f train.log`.

**4. Watch the curves.** In a notebook cell, while training runs in the terminal:

```
%load_ext tensorboard
%tensorboard --logdir "/content/drive/MyDrive/Research/experiments"
```

Refreshes itself every 30 s. (If the panel is blank in VS Code, open the same
notebook at colab.research.google.com — it renders there.) `history.csv` in the
run folder is the same data as a file, updated every epoch.

**4b. Keep the session alive.** Colab reclaims a VM it considers idle, and a run in the
terminal does not count as activity — four VMs have been lost during long cache builds
this way. Run this in a notebook cell and leave it running for the whole session:

```python
import time
while True:
    time.sleep(60)
```

**4c. Save the caches.** After any run has built them, copy them to Drive once:

```bash
python cache_sync.py --config configs/test17_sg2.yaml --to-drive
```

On a fresh VM, `--from-drive` restores them in minutes instead of the 1–2½ hour
rebuild. Same config, same cache key, same folder name on both sides.

**5. If the VM dies anyway.** Colab deletes VMs on idle and on a maximum lifetime,
and on Pro nothing running inside the VM can stop that. What survives is on
Drive: the best `model.keras` and `history.csv`. Start a fresh server, redo step
1, then:

```bash
nohup python -u train.py --config configs/test16_full.yaml --resume > train.log 2>&1 &
```

It reloads the checkpoint, continues the epoch count, and rebuilds the cache
(the slow first epoch, again). The checkpoint is the *best* epoch, not the last,
so a resumed run is not bit-identical to an uninterrupted one — say so if a
headline number came from one.

Dataset stays in Drive, cache goes to `/content` (fast local disk), outputs go to
Drive. Epoch 1 is slow — it builds the cache.

## Evaluating a run

```bash
python evaluate.py --config configs/test16_full.yaml
```

Reloads `experiments/<name>/model.keras` and writes the figures beside it, so a
run folder is self-describing. It rebuilds the data through `model.build_datasets`,
so eval sees exactly the mask training saw — an eval that quietly skipped the mask
would report a number the model never earned.

Grad-CAM is the shortcut check, not decoration: under `face_only` the heat should
sit on the face. If it sits on a corner, the model found something that isn't a face.

### Does the trained model use the background?

The panel's question. Retraining with `face_only` / `background_only` says where
signal exists in the *data*; it does not say what the trained model *uses*. Two
checks in `eval.json` → `shortcut_checks` run on the trained model, no retraining:

**Background swap.** Real face pasted onto a fake background and vice versa, via
the same feathered ellipse. One AUC, scored by the *face* label: well above 0.5 the
prediction follows the face; well below, the background; ~0.5, both. The seam is
the known weakness, so a **control** — same-class composites with identical seams
and no conflict — is reported beside it. If the control AUC stays near the plain
test AUC, the seams are benign and the swap number means what it says; if the
control collapses, the swap is uninformative and the output says so.
"Background" means everything outside the ellipse: hair, ears, shoulders, wall.

**Attention in face.** Share of Grad-CAM heat inside the face ellipse, averaged
over the whole test set, per class. Compare to `uniform_baseline` (the ellipse's
share of the image, ~0.55): well above it means the spatial branch attends to the
face. Spatial branch only — the FFT branch has no image-space heatmap.

**Results (2026-09-20):**

| check | mix, epoch 50 | mix, final | **SG2, final** | reading |
|---|---|---|---|---|
| plain test AUC | 0.947 | 0.973 | **0.964** | reference |
| control (same-class composites) | 0.934 | 0.968 | 0.963 | seams cost ≤0.005 — benign |
| swap, scored by face label | 0.847 | 0.886 | **0.896** | follows the face (background-following would be ~0.04) |
| attention in face, fakes | 0.83 | 0.86 | 0.78 | vs 0.57 uniform |
| attention in face, reals | 0.54 | 0.60 | **0.77** | on SG2, reals are judged by the face too |

And against the fine-tuned baselines on the same SG2 test set:

| | this model | MobileNetV3-S | Xception | EfficientNet-B0 |
|---|---|---|---|---|
| test AUC | 0.964 | 0.992 | 0.999 | **0.9997** |
| control (same-class composites) | 0.963 | 0.977 | 0.998 | 0.999 |
| swap, scored by face label | 0.896 | 0.764 | 0.879 | 0.902 |
| drop under a conflicting background | **0.07** | 0.21 | 0.12 | 0.10 |
| attention in face, real / fake | 0.77 / 0.78 | 0.69 / 0.65 | **0.93 / 0.94** | **0.47** / 0.83 |

The pretrained models are more accurate and every one of them is more
background-dependent than this model: a conflicting background costs them 0.10–0.21
AUC against 0.07 here. Xception is the most face-*attentive* of the five yet still
drops 0.12 — where a model looks and what flips its decision are different
measurements, which is why both are made. EfficientNet, when it calls an image
*real*, puts less than half its attention inside the face. Accuracy alone hides all
of that.

The decision follows the face, and it did so more firmly as the data got cleaner:
the swap number rose with training on the mix, and rose again on StyleGAN2 alone.
The real-class attention is the tell — barely above uniform on the mix (0.60), it
is clearly on the face on SG2 (0.77). With StyleGAN1's background blobs gone the
model stopped reading backgrounds for anything. A conflicting background costs
~0.07 AUC and never flips the prediction.

### Efficiency numbers

The same command also prints, and writes into `eval.json`:

```
device      : NVIDIA L4  (TF 2.20.0, XLA)                         measured 2026-09-20
params      : 1,020,417  (4.08 MB fp32 weights; checkpoint 12.33 MB incl. optimizer)
MACs        : 1.199 G per image at 256^2  (FLOPs ~= 2.40 G)
peak memory : 171 MB at bs=1 | 2811 MB at bs=144   (incl. autotune: 1301 | 5374)   re-measured 2026-09-24
activations : 8.39 MB largest tensor | 20.78 MB all, at bs=1 fp32 (analytic)
latency bs=1: 1.34 ms median | 1.34 mean | 1.41 p95   (n=1000)   [1.41 / 1.57 on a second L4 session]
throughput  : 1,564 img/s at batch 144
```

Two L4 sessions gave 1.34 and 1.41 ms median — ~5% apart on identical code. That is
the session-to-session noise floor, and why baselines must be measured in the *same*
session as the model they are compared with, never across sessions.

**Peak memory: the 1,301 MB the table carried until 2026-09-24 was wrong as a
footprint; the column below is the re-measurement.** The counter was reset *before* the warmup loop, which is
where cuDNN autotunes — it tries every conv algorithm, including FFT/Winograd variants
whose scratch workspace for a 256²×32-channel conv runs to hundreds of MB — so the
"peak" was the largest autotune trial, not the model. `measure_efficiency` now reports
both: `bs1_incl_autotune` (the old number) and `bs1` (peak over the timed loop after
warmup: weights + live activations + the chosen algorithm's workspace, what a running
process holds). It also writes an analytic, hardware-independent `activation_mb_bs1`.
Those analytic numbers, at bs=1 fp32:

| model | weights | all activations | largest tensor |
|---|---|---|---|
| **this model (no FFT)** | 4.0 MB | **20.8 MB** | 8.4 MB |
| MobileNetV3-Small | 4.1 MB | 28.7 MB | 1.2 MB |
| EfficientNet-B0 | 16.9 MB | 131.9 MB | 6.4 MB |
| Xception | 84.5 MB | 181.9 MB | 8.0 MB |

The network's own footprint is the smallest of the four. What the old column ranked was
cuDNN's appetite for scratch on a large-spatial conv, which is real on a GPU but is a
runtime property, and one a stride-2 stem (test21) shrinks 4×.

**Re-measured, all five models in one L4 session (2026-09-24, git 11883d0):**

| model | peak MB @ bs=1 | peak MB @ bs=144 | incl. autotune @ bs=1 (old column) | ms @ bs=1 this session |
|---|---|---|---|---|
| **this model, no FFT (headline)** | **171** | 2,811 | 1,301 | **0.96** |
| this model, with FFT | 171 | 2,962 | 1,301 | 1.45 |
| MobileNetV3-Small | **152** | **665** | 191 | 4.81 |
| EfficientNet-B0 | 205 | 2,017 | 234 | 7.10 |
| Xception | 409 | 2,987 | 717 | 4.84 |

At bs=1 the headline holds 171 MB against MobileNet's 152 — within 12%, and below
EfficientNet and Xception. The 7× gap was the autotune search. What survives, honestly:
at **batch 144** this model peaks at 2.8 GB against MobileNet's 0.67 GB, because its
full-resolution early activations scale with the batch — the throughput regime is where
the full-res stem costs.

**The stride-2 stem (test21, 2026-09-24) removes most of that cost.** Same 1.01M
parameters, first conv at stride 2: 0.28 G MACs (from 1.11), 158 MB at bs=1 (below
MobileNet), 1.0 GB at bs=144 (from 2.8), 5,547 img/s (from 2,092), CPU 4.0 ms against
the headline's 6.4 in the same session; GPU bs=1 latency unchanged at 0.97 ms (that
layer was never the GPU's bottleneck). Price: test AUC 0.9990 vs 0.9996, accuracy 0.984
vs 0.99, and ~130 epochs to converge against ~10 — Gragnaniello et al. 2021's claim
that early downsampling hurts is measurable here, but as a slower climb to nearly the
same place, not a ceiling. Whether 0.0006 AUC is a real gap or seed noise is what the
seed runs decide; if it is noise, the cheaper stem is the better headline. GPU latencies this session are
within 5% of the table's (0.96 / 1.45 / 4.81 / 7.10 / 4.84 vs 1.02 / 1.49 / 4.71 /
6.82 / 4.78): the session-to-session noise floor, ranking unchanged.

`eval.json` also carries a `system_under_test` block — GPU, TF version, platform,
mixed-precision policy, git SHA, timestamp, warmup and run count. Colab states its
GPU types vary over time, so a latency number without that block cannot be compared
to anything, including your own earlier run.

**Params are not compute — say which resource you mean.**

| model | params | MACs | input |
|---|---|---|---|
| **this model** | **1.02 M** | **1.20 G** | 256² |
| EfficientNet-B0 | 5.3 M | 0.39 G | 224² |
| ResNet-50 | 25.6 M | 4.1 G | 224² |
| Xception | 22.9 M | 8.4 G | 299² |

5× fewer parameters than EfficientNet-B0 and roughly 3× more compute, because the
spatial branch has no stride-2 stem and no bottleneck — the first two convs run at
full 256² and 128² and account for ~1.11 G of the 1.20 G. A panel will do this
arithmetic. Measured latency may still favour this model (depthwise convs, B0's
whole trick, often underutilise a GPU), which is exactly why the measured number
matters more than the FLOP count — but the claim must name the resource it saves.

This is the evidence for the resource claim, so the measurement is deliberate:
it times `tf.function(model(x, training=False))` rather than `model.predict()`
(which measures Keras dispatch, not a 1M-param forward pass), calls `.numpy()`
inside the timed region to force the async GPU queue to drain, and reports the
median and p95 because GPU timings have a long right tail.

**These numbers only mean something next to a baseline.** Latency is hardware-
and session-specific, so the baselines are measured on the same GPU, in the same
session, on the same test split — a number copied from another run, or another
paper, proves nothing.

### Baselines

```bash
python baselines.py --config configs/test16_full.yaml --model xception            # efficiency only, minutes
python baselines.py --config configs/test16_full.yaml --model xception --train    # fine-tune from ImageNet, ~1 h
```

Models: `xception` (the field's standard, FaceForensics++), `efficientnet_b0`
(the real efficiency rival — fewer MACs than this model), `mobilenet_v3_small`
(the one a panel will ask about). Each runs the *identical* task — same data,
split, mask, input size, `build_datasets`, timing code — and writes to
`experiments/test16_full/baselines/<model>/`, so one folder holds the model and
everything it is compared against.

Latency, params, MACs and memory do not depend on weights, so the efficiency
half needs no training. The accuracy half is a full fine-tune at lr 1e-4 through
the same `train()` loop. Each backbone's ImageNet preprocessing is a layer inside
the model — the data pipeline is byte-identical for every model, and the
preprocessing is timed as part of running it.

**Efficiency, measured — one L4, one session, 2026-09-20.** The accuracy columns are
two pipelines: *resize* (1024→512→256, the baselines' original fine-tunes) and *crop*
(native 256² windows, the headline's pipeline; `configs/test20_crop_baselines.yaml`,
same data/split/cache as the headline, batch 32 / lr 1e-4 / 30 epochs, run folder
`experiments/test20_crop_baselines/baselines/<model>/`). Only the crop column is
comparable to the headline.

**Crop column, read honestly (2026-09-23).** On native pixels every model saturates:
headline 0.9996, MobileNetV3-Small 0.9999, EfficientNet-B0 and Xception 1.0000. The
resize was the ceiling for all of them, not only the from-scratch model, so accuracy
no longer separates the four. What the claim rests on is (a) a from-scratch 1M-param
network reaching parity with fine-tuned ImageNet backbones on the same task, and
(b) the resource columns: 4.7× / 7× / 4.7× lower bs=1 GPU latency than MobileNet /
EfficientNet / Xception at equal or fewer parameters. Whether the baselines share the
StyleGAN3-T generalisation gap is the open question (`heldout.py --model <baseline>`, written, unrun).

| model | params | MACs @256² | **ms @ bs=1, L4** | ms @ bs=1, CPU | img/s @144 | peak MB @ bs=1 (re-measured 2026-09-24) | test AUC, SG2 (resize) | test AUC, SG2 (**crop**) |
|---|---|---|---|---|---|---|---|---|
| **this model, native crops, no FFT (headline)** | 1.01M | 1.11G | **1.02** | **8.4** | 2,098 | 171 | — | **0.9996** |
| this model, **stride-2 stem** (test21) | 1.01M | **0.28G** | 0.97 | **4.0** (6.4 for the headline, same session) | **5,547** | **158** | — | 0.9990 (acc 0.98) |
| this model, native crops, with FFT (test19) | 1.02M | 1.20G | 1.49 | 8.6 | 1,564 | 171 | — | 0.9994 |
| this model, resize pipeline (test17) | 1.02M | 1.20G | 1.34 | 8.7 | 1,564 | 171 (same network) | 0.964 (mix 0.973) | — |
| this model, no FFT branch, resize pipeline | 1.01M | 1.11G | **0.99** | **6.6** | 2,098 | 171 (same network) | 0.953 (mix 0.975) | — |
| MobileNetV3-Small | 1.01M | **0.07G** | 4.71 | 18.7 | **2,684** | **152** | 0.992 (ImageNet-pretrained, fine-tuned) | **0.9999** (acc 0.998; test20, 2026-09-23) |
| EfficientNet-B0 | 4.21M | 0.50G | 6.82 | 39.8 | 540 | 205 | 0.9997 (ImageNet-pretrained, fine-tuned) | **1.0000** (0.999999; acc 0.999; early-stopped ep 19; test20, 2026-09-23) |
| Xception | 21.1M | 5.95G | 4.78 | 65.0 | 377 | 409 | 0.9987 (ImageNet-pretrained, fine-tuned, stopped early at a 0.999 plateau) | **1.0000** (0.999999; acc 0.9995; early-stopped ep 27; test20, 2026-09-23) |

The honest reading. **Single-image latency: this model wins by 3.5–5×**, including
against EfficientNet-B0, which has *half* the MACs — MACs are an indirect metric
(ShuffleNetV2); depthwise-separable nets are FLOP-cheap and GPU-hostile, this model
is five plain convs. **It loses MACs (second-worst) and batched throughput (MobileNet's 17× fewer MACs
pay off once the GPU is saturated), and at bs=1 it holds 12% more memory than
MobileNet (171 vs 152 MB; the earlier "7×" was the autotune search, see *Peak memory*).
At batch 144 the full-resolution early activations cost it 4× MobileNet's memory —
no stride-2 stem; test21 tests whether one is affordable.** Params: tied with MobileNetV3-Small.

GPU latencies for the four baselines and the resize rows are one L4 session
(2026-09-20 morning); the two crop rows are a third session (2026-09-21: 1.49 with
FFT, 1.02 without — same network as the resize rows, so the same latency by
construction, and measured to be). A second session gave 1.44 / 5.06 / 8.05 ms for
this model / MobileNet / EfficientNet — same order, ~5–15% drift, depthwise nets
drifting most. **The CPU column is less stable than the GPU one:** the FFT branch's
CPU cost measured 24% on one VM (6.6 vs 8.7) and 2% on another (8.4 vs 8.6). Quote
CPU numbers as same-session pairs only; the GPU column carries the claim. **CPU column: 12-thread Xeon @ 2.2 GHz, plain
TensorFlow, no XLA, n=300.** The order did *not* flip: this model is 2.1× faster than
MobileNetV3 and 4.6× faster than EfficientNet-B0 on CPU too. Stock TF depthwise
kernels are poor and the memory-bound structure hurts on both. Caveat, stated: a phone
running TFLite/XNNPACK has depthwise convs heavily optimised and is where MobileNet was
designed to win; that was not measured.

The claim this supports is *the lowest single-image latency at ~1M parameters on both
hardware classes measured, at the cost of more compute and activation memory than
mobile-oriented designs, and — from scratch — 3 points of AUC against fine-tuned
pretrained backbones.* A different point on the curve, not a dominated one.

**Distillation (test18 / 18c) — closed, negative, with the reason.** Two teachers,
EfficientNet-B0 (4.2M, 0.9997) and MobileNetV3-Small (the student's size, 0.992),
both made the from-scratch student *worse*: 0.923 and 0.901 against 0.964, and both
students imported the teacher's background reliance (swap drop 0.21 / 0.18 vs 0.07).
Calibrating the teacher found nothing to fix (T_cal 1.10). What was wrong is the
setup, which the literature says fails: teacher scores computed once on clean images
while the student trains on augmented views of them (inconsistent views — Beyer et al.
2022), scores that are in-sample for the teacher and therefore near-hard, and a
target that depends on background features the student is built not to use. The
proper form — online distillation, teacher on the same augmented batch, out-of-sample
data, long schedule — is implemented as `online: true` and reserved for whichever
input pipeline wins the crop experiment.

**Native-resolution crops (test19) — the ceiling was the resize.** Every
from-scratch result at 256² topped out around 0.96 and every pretrained one at 0.99+;
the audit said the post-resize image had almost no high-frequency separation left
(0.52). `input_mode: crop` feeds the same model 256² windows of the *native* 1024²
pixels instead — no bicubic anywhere. Same input size, same model, same latency
(1.49 ms GPU, 8.6 ms CPU). Result: **test AUC 0.9994, accuracy 99%** — EfficientNet-B0's
number from a 1M-param network trained from scratch. It escaped the cold-start
plateau in 6 epochs where the resize model took 9, and jumped 0.17 AUC in two.

**Is it a shortcut?** At native resolution the reals carry sensor noise and a
Flickr-JPEG past that the fakes lack — a model could read "photograph" instead of
"generated". Three checks:

| check | result | reading |
|---|---|---|
| native-res audit, `highfreq` | AUC 0.553 | raw fine-detail *amount* does not separate the classes |
| JPEG q95 on both classes before scoring | 0.9994 → **0.9976** (no-FFT: 0.9996 → 0.9984; stride-2 stem: 0.9990 → 0.9978) | −0.002 (stride-2: −0.001); the resize model drops −0.027 under the same treatment |
| JPEG q75 (social-media grade) | 0.9994 → **0.9862** (no-FFT: 0.9864) | still above the resize model's *clean* 0.964 |

| **whole face downscaled to 256² (the resize pipeline's input)** | **0.5563** (with FFT: 0.5902) | **chance** — the fingerprint is scale-specific; shrink the image and it is gone. The FFT branch does not buy scale robustness either |

Compression flattens sensor noise and overwrites compression history for both classes
equally; a model reading those collapses at q75. This one lost 0.013, and is *more*
robust to compression than the resize pipeline. But downscaling removes the evidence
entirely: the crop model is at chance on the same faces shrunk to 256². **Operating
condition of the headline: native-resolution input.** Robust to recompression, not to
resizing. For platform-resized images (Instagram ~1080 px, WhatsApp smaller) the
current answer is size routing — the resize model (0.964) below ~512 px, the crop
model above; the research answer is scale augmentation at training time (windows drawn
at 1×, ½×, ¼×), which costs nothing at inference. Gragnaniello et al. 2021 report the
same failure mode for detectors on social-media re-uploads; it is the field's open
problem, not this model's alone. The signal is structure that survives
re-encoding — a generator fingerprint's profile, not a camera's. Held-out generator: run —
StyleGAN3-T gives AUC 0.70 for both models (see *Held-out generator* under Results);
the fingerprint is SG2-specific. Still to run: reals from a second source (CelebA-HQ). The whole-face checks (swap, attention) do not apply to
patches; the resize-mode results stand for those. `cache_size: 768` (the hair) is
the follow-up.

**The ablation — the FFT branch helps only when the task is hard.** Same model with
the branch removed and nothing else changed, on both datasets:

| dataset | with FFT | without | Δ AUC | best epoch (with / without) |
|---|---|---|---|---|
| StyleGAN1+2 mix (easy), resize | 0.973 | 0.975 | −0.002 | 85 / 102 |
| **StyleGAN2 only (hard), resize** | **0.964** | **0.953** | **+0.011** | 89 / 114 |
| StyleGAN2, **native crops** | 0.9994 | **0.9996** | −0.0002 | ~50 / 73 |

On the mix — where StyleGAN1's blob artifacts give plain convs an easy handle — the
branch is redundant. On downsampled StyleGAN2, where the audit finds no easy cue, it
buys 0.011 AUC and 2 points of accuracy. **On native crops it buys nothing again:
same AUC, same JPEG robustness (q95 0.9984 vs 0.9976, q75 0.9864 vs 0.9862).** The
pattern: a frequency branch helps only when the input pipeline has removed most of
the signal; give the spatial branch native pixels and it is redundant at 26% of the
latency. Its one remaining exam is generalisation to an unseen generator. The
no-FFT crop model is the headline.

The cost: 0.9% of the parameters, **26% of latency** (0.37 ms — `fft2d`, the magnitude
conversion and two convs at full 256², none of which a MAC count sees), and 26% of
training step time. The shortcut checks are the same shape with or without it
(swap 0.896 vs 0.883): it adds a signal, it does not change where the model looks.

A first pass on the mix showed a spurious +0.010 for the branch — patience-6 early
stopping cut the ablation off mid-climb. Patience is 15 in every config now.

Still open: whether the branch *generalises* better to a generator the model never
saw (the literature's actual claim), and whether it helps more on native-resolution
input, where the frequencies it was designed for have not been resized away.

## Results

| experiment | role | dataset | **test AUC** | val AUC (selection) | run folder |
|---|---|---|---|---|---|
| **test19_sg2_crop_no_fft** | **headline** — native-resolution crops, spatial branch only | FFHQ 1024 + StyleGAN2 ψ=1.0 (NVIDIA), 25k/class | **0.9996** (acc 0.99) | 0.9996 (best epoch 73, killed at 81 flat) | `experiments/test19_sg2_crop_no_fft/` |
| test19_sg2_crop | native crops, with FFT branch | same | 0.9994 (acc 0.99) | 0.9994 (~50 epochs incl. a resume; first run's history lost to a VM) | `experiments/test19_sg2_crop/` |
| test21_stride2_stem | headline with a stride-2 first conv (4× fewer MACs) | same | 0.9990 (acc 0.984) | 0.9983 (best epoch 129; ran to the 150 cap, plateaued from ~125) | `experiments/test21_stride2_stem/` |
| test17_sg2 | headline, resize pipeline (1024→512→256) | same | 0.9636 (acc 0.90) | 0.9645 (best epoch 89 of 104, early-stopped) | `experiments/test17_sg2/` |
| test17_sg2_no_fft | ablation — FFT branch removed | same | 0.9527 (acc 0.88) | 0.9503 (best epoch 114, killed at 116 while grinding) | `experiments/test17_sg2_no_fft/` |
| test18_distill | distillation, raw EfficientNet-B0 teacher, T=2, same 35k | same | 0.9226 (acc 0.84) — **worse**; swap drop 0.21 | 0.9123 (best 104, killed at 111) | `experiments/test18_distill/` |
| test18b_distill_cal | distillation, teacher calibrated first (Guo 2017) | same | not run: T_cal = 1.10 — the teacher is not miscalibrated, it is near-perfect (val 0.013 / 0.995 and right that often) | | — |
| test18c_distill_mnv3 | distillation, MobileNetV3-Small teacher (student's size, 0.992) | same | 0.9014 (acc 0.80) — **worse**; swap drop 0.18 | 0.904 (best 63, killed at 64) | `experiments/test18c_distill_mnv3/` |
| test16_full | *preliminary* — StyleGAN1+2 mix (15k SG2 + 10k SG1 per its `_manifest.csv`, learned 2026-09-24) | FFHQ 1024 + FakeMix, 20k/class | 0.9728 (acc 0.91) | 0.9724 (best 85 of 91) | `experiments/test16_full/` |
| test16_no_fft | *preliminary* ablation on the mix | same | 0.9751 (acc 0.91) | 0.9748 (patience 15; stopped at 101) | `experiments/test16_no_fft/` |
| test13_face | control — `face_only` | FFHQ 1024 + FakeMix, 25k/class | _pending_ | 0.9995 | `experiments/test13_face/` |
| test14_background | control — `background_only` | FFHQ 1024 + FakeMix, 20k/class | _pending_ | _rerun pending_ | `experiments/test14_background/` |

**StyleGAN2 alone is harder.** Same model, same seed: the mix reached val 0.77 after
one epoch; SG2 sat at chance for nine epochs before finding any signal (loss stuck
at ln 2), then climbed the same way to 0.964. The audit explains it — on SG2 no
single pixel property beats AUC 0.52, where the mix had `highfreq` at 0.62 from
StyleGAN1's artifacts. The 0.973 on the mix was partly StyleGAN1 being easy.

All three run on the same dataset. `test13_face` is at 25k/class where the other
two are at 20k — the split members differ, so it is a near-comparison, not an
exact one; drop it to 20k for parity if that matters at presentation time.

**Cite the test column, not val.** `ModelCheckpoint` and `EarlyStopping` both
select on `val_auc`, so 0.9995 is the maximum over 48 epochs on the very set used
to pick the model — optimistically biased by construction (Cawley & Talbot, JMLR
11, 2010). The split is a real 70/15/15 and `evaluate.py` already writes the
unbiased test number into `eval.json`; the table just has to quote that one.

test13's val figure is a record of the original notebook run — there is no run
folder behind it, so retrain before citing it anywhere. The old test14 run is
excluded: its mask line sat inside `if training:`, so val/test went unmasked and
it degenerated.

### Held-out generator: StyleGAN3-T

```bash
python heldout.py --config configs/test19_sg2_crop_no_fft.yaml --fake-dir "/content/drive/MyDrive/Fake(SG3-T-psi1)" --tag sg3t
python heldout.py --config configs/test19_sg2_crop.yaml        --fake-dir "/content/drive/MyDrive/Fake(SG3-T-psi1)" --tag sg3t
```

1,500 images from the official NVIDIA `stylegan3-t-ffhq-1024x1024.pkl` (ψ=1.0,
seeds 100000–101499, reference PyTorch ops) against 1,500 FFHQ reals the models never
sampled. Same native-crop eval view as the test set. Writes `eval_heldout_sg3t.json`
in each run folder (2026-09-22, L4).

| model | in-distribution test AUC (SG2) | **SG3-T AUC** | acc@0.5 | fake recall | mean p_fake real / fake |
|---|---|---|---|---|---|
| test19_sg2_crop_no_fft (headline) | 0.9996 | **0.703** | 0.529 | 0.064 | 0.012 / 0.081 |
| test19_sg2_crop (with FFT) | 0.9994 | **0.711** | 0.526 | 0.056 | 0.012 / 0.077 |
| test21_stride2_stem | 0.9990 | **0.735** | 0.540 | 0.085 | 0.010 / 0.101 |
| MobileNetV3-Small (test20, crop) | 0.9999 | **0.958** | 0.675 | 0.353 | 0.004 / 0.355 |
| EfficientNet-B0 (test20, crop) | 1.0000 | **0.993** | 0.844 | 0.689 | 0.003 / 0.681 |
| Xception (test20, crop) | 1.0000 | **0.967** | 0.651 | 0.301 | 0.000 / 0.309 |

**The pretrained baselines transfer; this model does not (2026-09-24).** Same held-out
set, same pipeline, same code path: EfficientNet-B0 ranks StyleGAN3 fakes at 0.993 AUC
where this model manages 0.703. The reals side is identical for all five (p_fake ≤
0.012), so the gap is entirely in what each network learned about "fake". The
baselines' features come from ImageNet and were only nudged by fine-tuning; ours were
learned from StyleGAN2 alone and are StyleGAN2's fingerprint. This is the effect Ojha
et al. (CVPR 2023) describe — a feature extractor trained on one generator's fakes
overfits to that generator — and Wang et al. (2020) rely on for their ImageNet-
pretrained ResNet-50. It is the honest limit of the from-scratch 1M model, and it is
the price of the resource numbers above. Thresholds: even the baselines' accuracies
at 0.5 are poor (0.65–0.84); their advantage is ranking, which a recalibrated
threshold would turn into accuracy. Ours has nothing to recalibrate.

**Neither of our models transfers.** Both score 94% of StyleGAN3 faces as real; the 0.70 AUC is
a weak ranking signal with no usable threshold. Unseen FFHQ reals are still scored
correctly (p_fake 0.012), so the failure is entirely on the fake side: the fingerprint
learned from StyleGAN2 is not present in StyleGAN3. That is by design of the
generator — Karras et al. (NeurIPS 2021) built StyleGAN3 to be alias-free, removing
exactly the upsampling artifacts StyleGAN2 leaves. The same collapse is reported for
every single-generator detector in Wang et al. (CVPR 2020) and Gragnaniello et al.
(ICME 2021); it is the known generalisation gap, not a property of this model.

**The FFT branch does not close the gap either** (0.711 vs 0.703, inside the noise of
1,500 samples). Fourth row of the ablation: 0 on the mix, +0.011 on resized SG2, 0 on
native SG2, +0.008 on a held-out generator. The frequency branch is dropped from the
headline on every axis measured.

What the claim is, stated exactly: in-distribution detection of StyleGAN2 at
1.01M parameters and ~1 ms, competitive with detectors 4–20× larger on the same
task. Cross-generator generalisation is where the pretrained baselines win (0.96–0.99
vs 0.70) and is reported as such. Two levers keep the architecture — and so the
latency and memory — unchanged and are the next experiments: Wang et al. 2020's
augmentation (blur σ≤3 and JPEG 30–100 at p=0.5, their generalisation recipe) and
training on two generators with a third held out. StyleGAN3-T is now the *final*
held-out set; a second unseen set (StyleGAN3-R) selects between runs so SG3-T is
scored once. Configs: `test22_sg2_crop_wang_aug.yaml` (`aug: wang`, same cache as the
headline) and `test23_sg2_sg1_crop.yaml` (`fake_dir` list + `fake_mix: [15000, 10000]`,
the FakeMix manifest's ratio; StyleGAN1 folder `Research/Dataset New/FakeSG1`,
provenance to be recorded under *Data*).

## Is the dataset honest?

```bash
python audit_dataset.py --config configs/test13_face.yaml
```

Scores each trivial file property (resolution, size, JPEG tables, post-resize
high-frequency energy) as an AUC. If any reaches ~0.9, the model can hit that
score without looking at a face — a shortcut, not detection. Metadata only; a
held-out generator is the real test.

**Result on the current data** (FFHQ 1024 + FakeMix, 800 sampled per class, 2026-09-19):

| property | real | fake | AUC |
|---|---|---|---|
| width / height | 1024 | 1024 | — (identical, nothing to score) |
| file_size | 1.37 MB | 1.43 MB | 0.56 |
| brightness | 111.4 | 113.0 | 0.52 |
| contrast | 61.3 | 56.8 | 0.63 |
| highfreq (post-pipeline) | 1.73 | 1.65 | 0.62 |

**And at native resolution** (the crop pipeline's actual input, 2026-09-21): `highfreq`
0.553 (real 0.75 / fake 0.69), brightness 0.51, contrast 0.51. No trivial cue there
either — whatever the crop model reads is in the *structure* of the fine detail, not
its amount.

Verdict: no single property beats 0.63. The resolution asymmetry of the earlier
data (reals 512, fakes 1024) is gone. The mild `highfreq`/`contrast` gap — fakes
slightly smoother — survives an identical resize path for both classes, so it is
a property of the images, not the preprocessing: GANs under-produce
high-frequency detail (Durall et al., CVPR 2020). That is the signal the FFT
branch is meant to find, and at 0.62 it is far too weak to explain a 0.99 alone.

## Tests

```bash
pip install -r requirements-dev.txt   # once, CPU-only, fine on a laptop
pytest tests -q                       # 12 tests, ~3s, no dataset or GPU needed
```

Root `conftest.py` exists solely so the bare command works: pytest's default
`prepend` import mode inserts `tests/`, not the repo root, so without it
`from model import ...` fails and the guard tests error instead of running.

- `test_fft_axes.py` — `fft2d` transforms the *last two* axes, so a 4D input
  would transform colour, not space. Checked against `numpy.fft.fft2`.
- `test_mask_applied.py` — val and test batches really are masked. Catches the
  `if training:` indentation bug above.
- `test_crop_mode.py` — `input_mode: crop` never resamples: the cache is an exact
  slice of the file and the eval view an exact slice of the cache.

Both verified to fail when those bugs are reintroduced.

## Group split

- **data/preprocessing** — `load_paths`, `load_and_resize`, `augment_and_mask`, mask
- **model/distillation** — `fft_layer`, `conv_block`, `build_model`
- **compression/efficiency** — not yet; add alongside `build_model` + a config flag
