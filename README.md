# GAN-face detection: lightweight dual-branch CNN

Detects StyleGAN2 fake faces vs real FFHQ faces with a **1.02M parameter** CNN.
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
| `audit_dataset.py` | Checks whether the two folders are separable by metadata alone |
| `configs/*.yaml` | Per-experiment settings |
| `tests/` | Two guard tests (see below) |
| `conftest.py` | Puts the repo root on `sys.path` so bare `pytest tests -q` can import `model` |
| `requirements-dev.txt` | Laptop install (CPU-only) for tests + audit; `requirements.txt` is the training env |
| `docs/` | `research-repo-practice.md` — sourced notes on how this repo should be run |
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
nohup python train.py --config configs/test16_full.yaml > train.log 2>&1 &
tail -f train.log          # watch; Ctrl+C stops the watching, not the training
```

`nohup … &` hands the process to the VM. Close the notebook, lose the network,
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
nohup python train.py --config configs/test16_full.yaml --resume > train.log 2>&1 &
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

### Efficiency numbers

The same command also prints, and writes into `eval.json`:

```
device      : <GPU name>  (TF <version>, git <sha>)
params      : <count>  (<size> MB on disk)
MACs        : 1.199 G per image at 256^2  (FLOPs ~= 2.40 G)
peak memory : <MB>
latency bs=1: <median> ms median | <mean> mean | <p95> p95   (n=1000)
throughput  : <img/s> at batch 144
```

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
and session-specific, so Xception / EfficientNet / ResNet must be measured on
the same GPU in the same session against the same test split — a number copied
from another run, or another paper, proves nothing. No baseline runner exists
yet; it belongs beside `build_model` when that phase starts.

## Results

| experiment | role | dataset | **test AUC** | val AUC (selection) | run folder |
|---|---|---|---|---|---|
| **test16_full** | **headline** — full image, no mask | `Dataset New`, 20k/class | _pending_ | _pending_ | `experiments/test16_full/` |
| test13_face | control — `face_only` | `Dataset New`, 25k/class | _pending_ | 0.9995 | `experiments/test13_face/` |
| test14_background | control — `background_only` | `Dataset New`, 20k/class | _pending_ | _rerun pending_ | `experiments/test14_background/` |

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
