# Lightweight dual-branch CNN for GAN-face detection

Detecting StyleGAN2-synthesized faces from real FFHQ faces with a **~1.02M
parameter** dual-branch CNN (spatial convolutions + an FFT-spectrum branch).
The research angle is **efficiency**: comparable detection quality at a fraction
of the compute/latency of large detectors.

## Structure

| path | what it is |
|---|---|
| `model.py` | Single source of truth: config, data pipeline, FFT layer, model, training loop |
| `train.py` | CLI entry point (`--config`) |
| `configs/*.yaml` | One file per experiment — **the only thing that changes between runs** |
| `tests/` | Two guard tests for bugs we've already been bitten by |
| `notebooks/` | Scratch/plotting only, never imported |

## Architecture

```
input (256, 256, 3)
├── spatial branch : Conv+Pool 32 → 64 → 128 → 256 → 256 → GAP        (256 feats)
└── FFT branch     : grayscale → fft2d over (H, W) → |·| → fftshift
                     → log1p → per-sample norm → Conv 16 → Pool
                     → Conv 32 → GAP                                   (32 feats)
                            ↓ concat → Dense 128 → Dropout 0.4 → sigmoid
```

## Running on Colab

```python
from google.colab import drive; drive.mount('/content/drive')
!git clone <this-repo> && cd CNN-based-GAN-face-detection && pip install -q pyyaml
```
```bash
python train.py --config configs/test13_face.yaml
```

Dataset stays in Drive, the resized `uint8` cache is written to `/content`
(local disk — reading it from Drive every epoch is what made training slow),
and outputs land in `experiments/<name>/`: `model.keras`, `history.csv`,
`metrics.json`.

The cache path is keyed by `limit_per_class`, **not** by `mask_mode` — because
the cache stores images *before* augmentation and masking, switching from
`face_only` to `background_only` reuses it and costs nothing. Changing
`limit_per_class` or `img_size` does need a fresh cache directory (it gets one
automatically for `limit_per_class`).

## Experiments

| experiment | what changed | val AUC | model folder |
|---|---|---|---|
| test13_face | `mask_mode: face_only`, 25k/class | **0.9995** | `experiments/test13_face/` |
| test14_background | `mask_mode: background_only`, 20k/class | _rerun pending_ | `experiments/test14_background/` |

`test13` val AUC is read off the original notebook run (best of 48 epochs).
The original `test14` notebook run is **not** reported: its mask line was
indented inside `if training:`, so val/test were unmasked, and its log shows
`val_auc: 0.0000e+00` with train loss ~1e-16 — a degenerate run. Rerun it with
`configs/test14_background.yaml` and fill in the number.

## The two guard tests

```bash
pip install -r requirements.txt   # or just: pip install pytest
pytest tests -q                   # 12 tests, no dataset or GPU needed
```

1. **`test_fft_axes.py`** — `tf.signal.fft2d` always transforms the *last two*
   axes. Handing it `(B, H, W, 3)` gives a 3-point transform over **colour**,
   not a 2D spatial spectrum. The test compares `fft_layer` against
   `numpy.fft.fft2` on the grayscale image over axes `(1, 2)`.
2. **`test_mask_applied.py`** — asserts train **and** val **and** test batches
   come out masked. The mask line once sat inside `if training:`, so the model
   was evaluated on inputs it was never trained on. Parametrized over all three
   splits so the same slip can't come back quietly.

Both were verified to fail when the original bugs are reintroduced.

## Group work

Module boundaries are kept clean for the three workstreams:

- **data/preprocessing** — `load_paths`, `load_and_resize`, `augment_and_mask`, mask geometry
- **model/distillation** — `fft_layer`, `conv_block`, `build_model`
- **compression/efficiency** — not added yet; slots in alongside `build_model` + a config flag
