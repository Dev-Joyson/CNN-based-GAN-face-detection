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
| `baselines.py` | Entry point: Xception / EfficientNet-B0 / MobileNetV3-Small on the identical task (`--model`, `--train`) |
| `audit_dataset.py` | Checks whether the two folders are separable by metadata alone |
| `configs/*.yaml` | Per-experiment settings |
| `tests/` | Two guard tests (see below) |
| `conftest.py` | Puts the repo root on `sys.path` so bare `pytest tests -q` can import `model` |
| `requirements-dev.txt` | Laptop install (CPU-only) for tests + audit; `requirements.txt` is the training env |
| `docs/` | `learnings.md` — one line per thing we found out and what changed; `research-repo-practice.md` — sourced notes on how this repo should be run |
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

**Result, test16_full final (best epoch 85, 2026-09-20):**

| check | epoch 50 | **epoch 85** | reading |
|---|---|---|---|
| plain test AUC | 0.947 | **0.973** | reference |
| control (same-class composites) | 0.934 | 0.968 | seams cost 0.005 — benign |
| swap, scored by face label | 0.847 | **0.886** | follows the face (background-following would be ~0.03) |
| attention in face, fakes | 0.83 | **0.86** | vs 0.57 uniform — looks at the face to call fake |
| attention in face, reals | 0.54 | 0.60 | now above uniform |

The decision follows the face, and **every check moved further toward the face
with more training** — the model became more face-focused, not less, as it
improved. A conflicting background costs ~0.08 AUC and does not flip the
prediction. Roughly: 90% face, 10% background.

### Efficiency numbers

The same command also prints, and writes into `eval.json`:

```
device      : NVIDIA L4  (TF 2.20.0, XLA)                         measured 2026-09-20
params      : 1,020,417  (4.08 MB fp32 weights; checkpoint 12.33 MB incl. optimizer)
MACs        : 1.199 G per image at 256^2  (FLOPs ~= 2.40 G)
peak memory : 1301 MB at bs=1 | 7278 MB at bs=144
latency bs=1: 1.34 ms median | 1.34 mean | 1.41 p95   (n=1000)   [1.41 / 1.57 on a second L4 session]
throughput  : 1,564 img/s at batch 144
```

Two L4 sessions gave 1.34 and 1.41 ms median — ~5% apart on identical code. That is
the session-to-session noise floor, and why baselines must be measured in the *same*
session as the model they are compared with, never across sessions.

Peak memory is what TensorFlow *reserved* (cuDNN/XLA workspace, allocator pool), not
what a 4 MB model strictly needs — an upper bound, comparable across models measured
identically on the same machine, not an absolute footprint.

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

**Efficiency, measured — one L4, one session, 2026-09-20:**

| model | params | MACs @256² | **ms @ bs=1** | img/s @144 | peak MB @ bs=1 | test AUC |
|---|---|---|---|---|---|---|
| **this model** | 1.02M | 1.20G | **1.34** | 1,564 | 1,301 | **0.973** |
| this model, no FFT branch | 1.01M | 1.11G | **0.99** | 2,098 | 1,301 | **0.975** |
| MobileNetV3-Small | 1.01M | **0.07G** | 4.71 | **2,684** | **181** | _pending_ |
| EfficientNet-B0 | 4.21M | 0.50G | 6.82 | 540 | 198 | _pending_ |
| Xception | 21.1M | 5.95G | 4.78 | 377 | 560 | _pending_ |

The honest reading. **Single-image latency: this model wins by 3.5–5×**, including
against EfficientNet-B0, which has *half* the MACs — MACs are an indirect metric
(ShuffleNetV2); depthwise-separable nets are FLOP-cheap and GPU-hostile, this model
is five plain convs. **It loses MACs (second-worst), peak memory (worst, 7× MobileNet
— no stride-2 stem, so full-resolution early activations; the FFT tensor is not the
cause, the no-FFT model peaks identically at bs=1), and batched throughput (MobileNet's 17× fewer MACs pay off once the GPU is
saturated).** Params: tied with MobileNetV3-Small.

The claim this supports is *the lowest single-image GPU latency at ~1M parameters,
at the cost of more compute and activation memory than mobile-oriented designs* —
a different point on the curve, not a dominated one. The ranking is GPU-specific;
on a CPU or phone MobileNet's MAC advantage would likely reverse it.

**The ablation — the FFT branch buys nothing in-domain.** `configs/test16_no_fft.yaml`
is the headline with the branch removed and nothing else changed. Result: **with
0.973 / 1.34 ms, without 0.975 / 0.99 ms.** Same accuracy, 26% less latency. The
branch is 0.9% of the parameters but a quarter of the time — `fft2d`, the magnitude
conversion and two convs at full 256², none of which a MAC count sees. The shortcut
checks are unchanged without it (swap 0.880 vs 0.886, attention 0.86 vs 0.86): it
does not change where the model looks either.

A first pass showed a 0.010 gap in the branch's favour. It was an artefact: the
ablation was cut off by patience-6 early stopping at epoch 61, when both runs sat
at 0.960; given patience 15 it climbed to 0.975. Patience is 15 in every config now.

What this leaves open is the branch's real claim from the literature — that
frequency features *generalise* to generators the model never saw. That is a
held-out-generator test, not an in-domain one, and it is the only remaining reason
to keep the branch.

## Results

| experiment | role | dataset | **test AUC** | val AUC (selection) | run folder |
|---|---|---|---|---|---|
| **test16_full** | **headline** — full image, no mask | FFHQ 1024 + FakeMix, 20k/class | **0.9728** (acc 0.91) | 0.9724 (best epoch 85 of 91, early-stopped) | `experiments/test16_full/` |
| test16_no_fft | ablation — FFT branch removed, else identical | FFHQ 1024 + FakeMix, 20k/class | **0.9751** (acc 0.91) | 0.9748 (resumed with patience 15; stopped at 101) | `experiments/test16_no_fft/` |
| test13_face | control — `face_only` | FFHQ 1024 + FakeMix, 25k/class | _pending_ | 0.9995 | `experiments/test13_face/` |
| test14_background | control — `background_only` | FFHQ 1024 + FakeMix, 20k/class | _pending_ | _rerun pending_ | `experiments/test14_background/` |

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

Both verified to fail when those bugs are reintroduced.

## Group split

- **data/preprocessing** — `load_paths`, `load_and_resize`, `augment_and_mask`, mask
- **model/distillation** — `fft_layer`, `conv_block`, `build_model`
- **compression/efficiency** — not yet; add alongside `build_model` + a config flag
