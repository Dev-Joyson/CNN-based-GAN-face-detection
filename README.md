# GAN-face detection: lightweight dual-branch CNN

Detects StyleGAN2 fake faces vs real FFHQ faces with a **1.02M parameter** CNN.
Research angle: **efficiency** — high accuracy at low compute/latency.

## How it works

```
python train.py --config configs/test13_face.yaml
       │                 │
       │                 └── configs/*.yaml   ← the only thing you change per experiment
       ▼
    train.py  (20 lines: read config, call train)
       ▼
    model.py  ← all the code lives here
       ▼
  experiments/<name>/   model.keras · history.csv · metrics.json

python evaluate.py --config configs/test13_face.yaml
       ▼
  experiments/<name>/   confusion_matrix.png · roc.png · gradcam.png · eval.json
```

Only touch `model.py` to change the *method*.

| file | what it is |
|---|---|
| `model.py` | Config, data pipeline, FFT layer, model, training loop |
| `train.py` | Entry point: train (`--config`) |
| `evaluate.py` | Entry point: confusion matrix, ROC, Grad-CAM, efficiency numbers |
| `audit_dataset.py` | Checks whether the two folders are separable by metadata alone |
| `configs/*.yaml` | Per-experiment settings |
| `tests/` | Two guard tests (see below) |
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

**The mask controls what the model may see.** One feathered oval, three modes:

| `mask_mode` | model sees |
|---|---|
| `face_only` | face only, background blacked out |
| `background_only` | background only — the **control**: if it still works, it wasn't reading the face |
| `none` | everything |

Applied to train, val and test identically. This is load-bearing.

**Cache placement makes training fast.**

```
read → resize → uint8  ──► CACHE ──►  shuffle → augment → mask → batch
      (slow, runs once)               (fast, every epoch)
```

Caching *before* augment/mask means augmentation stays random each epoch, and
switching `mask_mode` reuses the cache for free. Cache is keyed by
`limit_per_class`; changing `img_size` needs a manual clear.

## Running (Colab)

Open `notebooks/colab_runner.ipynb` → Select Kernel → Colab → New Colab Server → GPU.
Or from a `Colab: Open Terminal` shell on the VM:

```bash
git clone https://github.com/Dev-Joyson/CNN-based-GAN-face-detection.git
cd CNN-based-GAN-face-detection && pip install -q pyyaml
python train.py --config configs/test13_face.yaml
```

Dataset stays in Drive, cache goes to `/content` (fast local disk), outputs go to
Drive so they survive a disconnect. Epoch 1 is slow — it builds the cache.

## Evaluating a run

```bash
python evaluate.py --config configs/test13_face.yaml
```

Reloads `experiments/<name>/model.keras` and writes the figures beside it, so a
run folder is self-describing. It rebuilds the data through `model.build_datasets`,
so eval sees exactly the mask training saw — an eval that quietly skipped the mask
would report a number the model never earned.

Grad-CAM is the shortcut check, not decoration: under `face_only` the heat should
sit on the face. If it sits on a corner, the model found something that isn't a face.

### Efficiency numbers

The same command also prints, and writes into `eval.json`:

```
device      : <GPU name>
params      : <count>  (<size> MB on disk)
peak memory : <MB>
latency bs=1: <median> ms median | <mean> mean | <p95> p95   (n=100)
throughput  : <img/s> at batch 144
```

This is the evidence for the resource claim, so the measurement is deliberate:
it times `tf.function(model(x, training=False))` rather than `model.predict()`
(which measures Keras dispatch, not a 1M-param forward pass), calls `.numpy()`
inside the timed region to force the async GPU queue to drain, and reports the
median and p95 because GPU timings have a long right tail.

**These numbers only mean something next to a baseline.** Latency is hardware-
and session-specific, so Xception / EfficientNet / ResNet must be measured on
the same GPU in the same session against the same test split — a number copied
from another run, or another paper, proves nothing. No baseline runner exists
yet; it belongs beside `build_model` when that phase starts.

## Results

| experiment | what changed | val AUC |
|---|---|---|
| test13_face | `face_only`, 25k/class | **0.9995** |
| test14_background | `background_only`, 20k/class | _rerun pending_ |

test13's number is a record of the original notebook run (best of 48 epochs) —
retrain to reproduce it. The old test14 run is excluded: its mask line sat inside
`if training:`, so val/test were unmasked and it went degenerate.

## Is the dataset honest?

```bash
python audit_dataset.py --config configs/test13_face.yaml
```

Scores each trivial file property (resolution, size, JPEG tables, post-resize
high-frequency energy) as an AUC. If any reaches ~0.9, the model can hit that
score without looking at a face — a shortcut, not detection. Metadata only; a
held-out generator is the real test.

## Tests

```bash
pytest tests -q     # 12 tests, ~3s, no dataset or GPU needed
```

- `test_fft_axes.py` — `fft2d` transforms the *last two* axes, so a 4D input
  would transform colour, not space. Checked against `numpy.fft.fft2`.
- `test_mask_applied.py` — val and test batches really are masked. Catches the
  `if training:` indentation bug above.

Both verified to fail when those bugs are reintroduced.

## Group split

- **data/preprocessing** — `load_paths`, `load_and_resize`, `augment_and_mask`, mask
- **model/distillation** — `fft_layer`, `conv_block`, `build_model`
- **compression/efficiency** — not yet; add alongside `build_model` + a config flag
