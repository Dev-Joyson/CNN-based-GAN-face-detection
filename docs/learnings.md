# Learnings

One line per thing that surprised us, and what changed because of it. Add to the
right section when it happens; keep each entry to a line. This is the record —
the README says how things *are*, this says how we found out.

## Data

- **2026-09-18 — Reals were 512², fakes 1024².** Both resized to 256, but fakes were
  low-passed twice and reals once. Resize equalises size, not spectra; the FFT branch is
  built to find exactly that residual, and the face/background masks do not control for
  it. → FFHQ reals at 1024 so both classes take the identical path.
- **2026-09-19 — A sorted prefix of a mixed folder can be one generator.** FakeMix holds
  StyleGAN1 + 2; FFHQ is numbered. → `load_paths` takes a seeded random sample.
- **2026-09-19 — 25,001 files for 25,000 images.** One non-image would have reached
  `decode_image` mid-epoch. → folders are filtered by extension.
- **2026-09-19 — Audit on FFHQ-1024 + FakeMix: no single property beats AUC 0.63.**
  The residual `highfreq`/`contrast` gap survives an identical resize path, so it is the
  images, not the pipeline — GANs under-produce high frequencies (Durall 2020). That is
  the signal, not a shortcut.
- **2026-09-20 — With the resampling asymmetry removed, the same architecture scores
  0.973 test AUC (best epoch 85 of 91, early-stopped), not 0.9995.** The gap is
  roughly how much of the old result was preprocessing. The curve was slow and steady
  from 0.77 — the shape of learning, not of a shortcut. Accuracy 91%, balanced.
- **2026-09-20 — 50 epochs was too few.** The notebook's cap; val_auc was 0.945 and
  climbing at 50, 0.972 at 85. Let patience decide, not the cap.
- **Separate folders, one generator family: 0.9995 alone proves nothing.** Baselines
  double as a difficulty check; a held-out generator is the real test.

## Pipeline

- **The mask line was once indented inside `if training:`.** Val/test went unmasked and
  test14 degenerated. → outside the block, with a guard test that fails if it moves.
- **`fft2d` acts on the last two axes.** Given (B,H,W,3) it transformed colour, not
  space. → grayscale, squeeze, then FFT; guard test compares to `numpy.fft.fft2`.
- **Cache goes after resize, before augment/mask.** Augmentation stays random per epoch;
  switching `mask_mode` reuses the cache.
- **2026-09-17 — Cache key must hold everything that changes the bytes.** With two
  datasets and a 3-seed plan, keying on `limit_per_class` alone would have served one
  config's images to another, silently. → folders + sizes + seed in the key.
- **Shuffle must come after the cache.** Before it, epoch 1's order is frozen into the
  file and replayed forever.
- **2026-09-19 — Drive reads ~12 images/s.** Epoch 1 on 40k images ≈ 50 min, GPU idle;
  epoch 2 ran in 66 s off the cache. Paid once per VM.
- **2026-09-19 — `cache_dataset_ops.cc:333 "will be discarded"` fires even when the cache
  completes.** It comes from a side-iterator at the epoch boundary. Verify a cache by
  its file: `train.data` must be N × H × W × 3 (+4 B/record) with no `.lockfile`. A
  warm-pass "fix" was written for this and reverted once the bytes said otherwise.

## Model and the claim

- **2026-09-13 — Params are not compute.** 1.02M params but 1.20 G MACs — 5× fewer
  params than EfficientNet-B0 and ~3× more compute, because there is no stride-2 stem.
  The claim must name which resource it saves. → MACs reported beside params.
- **2026-09-17 — The headline is the full-image model.** face_only / background_only
  are controls built to answer the panel head's "is it reading the background?" — not
  the result. (Had this backwards for two days.)
- **2026-09-20 — The mask retrains answer the wrong question.** face_only / background_only
  say where signal exists in the data; the panel asked what the trained model uses.
  → test-time background swap (with a same-class seam control) and Grad-CAM-in-face
  fraction, both on the trained model, in `evaluate.py`. Also: the "backgrounds look
  different" the panel saw may have been the resampling asymmetry itself.
- **2026-09-20 — The trained model follows the face, measured.** Swap AUC 0.886 vs
  control 0.968 (background-following would be ~0.03); Grad-CAM 86% in-face for fakes.
  Background conflict costs ~0.08 AUC, does not flip it. Every check moved further
  toward the face between epoch 50 and 85 — more training made it more face-focused.
- **2026-09-20 — A reclaimed VM at the finish line costs nothing.** Best checkpoint on
  Drive + history.csv = the run; `sort -t, -k6 -g history.csv | tail -1` finds the
  best epoch and whether patience had already run out. It had. Evaluate, don't resume.
- **2026-09-19 — TF "peak memory" is what it reserved, not what the model needs.**
  1.3 GB at bs=1 for a 4 MB model is cuDNN/XLA workspace. Comparable across models
  measured the same way; never quote it as a footprint. The checkpoint (12.3 MB) is
  3x the weights (4.1 MB) because it carries Adam's moments.
- **Baselines compare on the headline task, never on a masked one.** Xception is the
  field standard (FaceForensics++); EfficientNet-B0 is the real rival; MobileNetV3 is the
  one a panel will ask about. Fine-tune fully, `preprocess_input` inside each model.
- **Latency needs no training; accuracy does.** The efficiency half of the comparison
  can be measured before any baseline is fine-tuned.
- **Published detectors zero-shot measure generalisation, not architecture.** CNNDetection
  was trained on ProGAN; scoring it on StyleGAN2 data says nothing about our model.

## Measurement

- **2026-09-13 — Report test AUC, not best-epoch val AUC.** Checkpoint and early stop
  both select on `val_auc`; the val number is biased by construction (Cawley & Talbot 2010).
- **Time `tf.function(model(x))`, not `model.predict()`.** `predict()` measures Keras
  dispatch, which dwarfs a 1M-param forward pass. `.numpy()` inside the timed region;
  median and p95; n ≥ 1000, or a p95 rests on five samples.
- **Record the system under test.** Colab's GPU type varies by session; a latency without
  the GPU next to it is comparable to nothing. → `eval.json` carries it.
- **One seed is attackable.** Split variance dominates init variance, and `cfg.seed`
  drives both, so 3 seeds is one number change. Planned.
- **Do not enable `enable_op_determinism()`.** TF documents `Dataset.map` with random
  ops becoming orders of magnitude slower.

## Workflow

- **2026-09-13 — The notebook cannot be deleted.** The VS Code Colab extension activates
  `onNotebook`; no notebook, no kernel picker, no terminal, no Drive mount.
- **Flat layout, plain YAML, no Hydra/MLflow/W&B/DVC.** No study links directory layout
  to any outcome; the tooling solves problems this repo does not have.
- **2026-09-19 — Train from the VM terminal under `nohup`, with `python -u`.** Cell-bound
  training dies with the frontend; buffered stdout hides the log. `caffeinate -i` on the
  Mac. Colab Pro has no background execution — a reclaimed VM is survivable only via
  `--resume` from the Drive checkpoint.
- **2026-09-19 — XLA's conv-autotune "results mismatch" is a self-check, not a failure.**
  It discards the disagreeing algorithm. Verify by watching `val_auc` climb.
- **Dev requirements are floors, not Colab's pins.** macOS ships Python 3.9; Colab's
  matplotlib needs 3.10.
