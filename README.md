# Lightweight dual-branch CNN for GAN-face detection

Detecting StyleGAN2-synthesized faces from real FFHQ faces with a **~1.02M
parameter** dual-branch CNN (spatial convolutions + an FFT-spectrum branch).
The research angle is **efficiency**: comparable detection quality at a fraction
of the compute/latency of large detectors.

---

## How this repo works

There is **one real code file**. Everything else is a switch, a check, or notes.

```
  python train.py --config configs/test13_face.yaml
             │              │
             │              └── configs/*.yaml   the settings you change
             │                                   between experiments
             ▼
          train.py      reads the flag, loads the YAML, calls train()
             │          (~20 lines, no logic of its own)
             ▼
          model.py      ALL the actual code: config, data pipeline,
             │          FFT layer, model, training loop
             ▼
   experiments/<name>/  model.keras · history.csv · metrics.json
```

To run a new experiment you write a new YAML. You should not need to touch
`model.py` unless you are changing the *method*.

### Where everything lives

| path | what it is |
|---|---|
| `model.py` | Single source of truth: config, data pipeline, FFT layer, model, training loop |
| `train.py` | The button you press. Parses `--config`, calls `model.train()` |
| `configs/*.yaml` | One file per experiment — **the only thing that changes between runs** |
| `tests/` | Two guard tests for bugs this project has already been bitten by |
| `notebooks/` | Scratch space for plotting/Grad-CAM. Never imported by the code |
| `experiments/` | Run outputs. Git-ignored — models are too big to commit |
| `requirements.txt` | Pinned dependencies |

### Inside `model.py`, top to bottom

Five labelled sections. Grep the function name to jump to any of them.

| section | what it does |
|---|---|
| **Config** | `Config` dataclass — every setting with a default. The YAML just overrides these. Crashes early on a typo'd `mask_mode` rather than running something weird |
| **Mask** | `feathered_ellipse()` builds an oval stencil: `1` on the face, `0` at the edges, soft fade between. `apply_mask()` is the config-driven part |
| **Data pipeline** | `load_paths()` finds + splits the files; `load_and_resize()` is the slow cached part; `augment_and_mask()` is the fast per-epoch part; `build_dataset()` wires them in order |
| **Model** | `fft_layer()` is the frequency branch; `conv_block()` is one conv + one halving; `build_model()` joins both branches |
| **Training** | `train()` sets the seed, builds data + model, trains with 3 callbacks, then evaluates val/test into `metrics.json` |

### The two ideas worth understanding

**1. The mask decides what the model is allowed to see.** One oval stencil, three modes:

| `mask_mode` | maths | the model sees |
|---|---|---|
| `face_only` | `image × mask` | the face; background blacked out |
| `background_only` | `image × (1 − mask)` | hair/shoulders/background; face blacked out |
| `none` | unchanged | everything |

`background_only` is the **control experiment**: if the detector still works
with the face removed, it was never really reading the face.

The mask is applied to **train, val and test identically**. This is load-bearing
— see the guard tests below.

**2. Where the cache sits is why training is fast.** The pipeline order is:

```
read JPEG → resize 512 → resize 256 → cast uint8  ──► CACHE ◄── slow, runs once
                                                        │
                              shuffle → augment → mask → batch  ── fast, every epoch
```

The cache holds *plain resized images*, before any augmentation or masking. Two
consequences:

- Augmentation stays random every epoch (it happens after the cache).
- **Switching `mask_mode` reuses the cache for free** — no need to clear it.

The cache path is keyed by `limit_per_class`, so test13 (25k) and test14 (20k)
get separate caches automatically. Changing `img_size` needs a manual clear.

---

## Architecture

```
input (256, 256, 3)
├── spatial branch : Conv+Pool 32 → 64 → 128 → 256 → 256 → GAP        (256 feats)
└── FFT branch     : grayscale → fft2d over (H, W) → |·| → fftshift
                     → log1p → per-sample norm → Conv 16 → Pool
                     → Conv 32 → GAP                                   (32 feats)
                            ↓ concat → Dense 128 → Dropout 0.4 → sigmoid
```

Two branches because GAN artifacts show up in two different places: the spatial
branch looks for visual tells, the FFT branch looks for the periodic
checkerboard pattern upsampling layers leave in the frequency spectrum.

Output is one number: `0` = real, `1` = fake.

## Running on Colab

```python
from google.colab import drive; drive.mount('/content/drive')
!git clone <this-repo> && cd CNN-based-GAN-face-detection && pip install -q pyyaml
```
```bash
python train.py --config configs/test13_face.yaml
```

Dataset stays in Drive; the cache is written to `/content` (local disk — reading
images off Drive every epoch is what made training slow); outputs land in
`experiments/<name>/`.

On Colab, TensorFlow/numpy/sklearn/matplotlib are already installed, so
`pip install pyyaml` is usually all you need. `requirements.txt` is there for
running locally.

## Experiments

| experiment | what changed | val AUC | model folder |
|---|---|---|---|
| test13_face | `mask_mode: face_only`, 25k/class | **0.9995** | `experiments/test13_face/` |
| test14_background | `mask_mode: background_only`, 20k/class | _rerun pending_ | `experiments/test14_background/` |

`test13`'s val AUC is the best of the 48 epochs logged in the original
`Test13_FFT_More_dataset.ipynb` run. That notebook is no longer in the repo, so
this number is a **record of a past run, not something the current code
reproduces** — retrain to regenerate it.

The original `test14` run is deliberately **not** reported: its mask line was
indented inside `if training:`, so val/test were unmasked, and its log showed
`val_auc: 0.0000e+00` with train loss ~1e-16 — degenerate. Rerun with
`configs/test14_background.yaml` and fill in the number.

## The two guard tests

```bash
pytest tests -q     # 12 tests, ~3s, no dataset or GPU needed
```

They invent their own tiny images, so they run anywhere — laptop included.

1. **`test_fft_axes.py`** — `tf.signal.fft2d` always transforms the *last two*
   axes. Handing it `(B, H, W, 3)` gives a 3-point transform over **colour**,
   not a 2D spatial spectrum. The test recomputes the spectrum with
   `numpy.fft.fft2` over axes `(1, 2)` and checks `fft_layer` matches.
2. **`test_mask_applied.py`** — pushes all-white images through the pipeline and
   checks a corner pixel and a centre pixel, for train **and** val **and** test.
   The mask line once sat inside `if training:`, so the model was evaluated on
   inputs it was never trained on. Parametrized over all three splits so the
   same slip can't come back quietly.

Both were verified to fail when the original bugs are reintroduced — a test that
has never failed hasn't been shown to work.

## Group work

Module boundaries are kept clean for the three workstreams:

- **data/preprocessing** — `load_paths`, `load_and_resize`, `augment_and_mask`, mask geometry
- **model/distillation** — `fft_layer`, `conv_block`, `build_model`
- **compression/efficiency** — not added yet; slots in alongside `build_model` + a config flag

Adding an experiment = a new YAML in `configs/`. Adding a *method* = a function
in `model.py` plus a config flag to switch it on. Nothing goes in `notebooks/`
that another run depends on.
