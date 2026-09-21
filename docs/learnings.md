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
  0.973 test AUC (best epoch 85 of 91, early-stopped), not 0.9995.** But two things
  changed at once — the resize fix AND the fakes went from StyleGAN2-only to a
  StyleGAN1+2 mix — so the gap is their combined effect, not attributable to either
  alone. Separating them needs a StyleGAN2-only run on the clean pipeline. The curve
  was slow and steady from 0.77 — the shape of learning, not of a shortcut.
- **2026-09-20 — A mixed fake set averages two difficulties.** StyleGAN1 has blob and
  stronger spectral artifacts that StyleGAN2 was designed to remove; 0.973 likely
  hides an easy SG1 number and a harder SG2 one. Report per-generator AUC.
- **2026-09-20 — 50 epochs was too few.** The notebook's cap; val_auc was 0.945 and
  climbing at 50, 0.972 at 85. Let patience decide, not the cap.
- **2026-09-20 — StyleGAN2 alone: test AUC 0.964, and nine epochs at chance first.** Same
  model/seed that hit val 0.77 in one epoch on the mix sat at loss = ln 2 until epoch 9,
  then climbed normally to 0.9645 (best 89, stopped 104). No easy handle: the audit's
  worst pixel property is 0.52. The plateau length is itself a measure of how much
  StyleGAN1 was carrying the mix result. Shortcut checks stronger than on the mix
  (swap 0.896; real-class attention 0.77, up from 0.60).
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
- **2026-09-20 — MACs predicted the latency ranking wrong.** EfficientNet-B0 has half
  this model's MACs and is 5× slower at bs=1; MobileNetV3-S has 17× fewer and is 3.5×
  slower. Depthwise-separable = many tiny launch-bound kernels; five plain convs = few
  fat ones. At batch 144 the order partly reverses (MobileNet 2,684 img/s vs 1,564).
  The claim must name the regime: single-image GPU latency.
- **2026-09-20 — This model is the worst of the four on peak memory at bs=1** (1.3 GB
  vs 181–560 MB): no stride-2 stem, so the first two convs run at 256²/128², plus a
  256² complex FFT tensor. Same root cause as the MACs. Say it, don't hide it.
- **2026-09-20 — The FFT branch helps only when the task is hard.** Mix: 0.973 with vs
  0.975 without — nothing. StyleGAN2 alone: 0.964 vs 0.953, +2 pts accuracy, best epoch
  25 sooner, gap consistent from epoch 40 on. Where convs have an easy cue the branch
  is redundant; where they don't, it contributes. One seed — seed 43 confirms or kills
  it. (A first mix pass showed a spurious +0.010: patience-6 early stopping. Patience
  is 15 now; a single-seed endpoint gap is not a finding, a 60-epoch consistent gap is
  closer to one.)
- **2026-09-20 — The FFT branch is 0.9% of the params and 26% of the latency.** MACs do not
  see fft2d, the complex magnitude, or that its convs run at full 256². "Cheap in
  parameters" and "cheap in time" are different claims. Without it: 0.99 ms, 4.8× MobileNet.
- **2026-09-20 — The 1.3 GB peak at bs=1 is the spatial stem, not the FFT.** The no-FFT model
  peaks identically. Earlier attribution corrected.
- **2026-09-20 — A fine-tuned MobileNetV3-Small reaches 0.992 on SG2 where this model
  reaches 0.964.** Same 1M params; ImageNet pretraining vs from scratch on 35k images.
  The task is learnable to 0.99 — the ceiling was the training regime, not the data.
  It is 3.9× slower in-session and drops 0.21 AUC under a conflicting background where
  this model drops 0.07: more accurate, and much more background-dependent.
- **2026-09-20 — Keras' AUC metric is approximate (200 thresholds).** The training log's
  `test:` said 0.9885; sklearn's exact AUC on the same weights is 0.9916. Quote the exact
  one from evaluate.py, never the Keras metric.
- **2026-09-20 — Depthwise nets vary more across sessions than plain convs.** MobileNetV3
  4.71 → 5.48 ms between two L4 VMs; this model 1.34 → 1.42. Launch-bound kernels are
  sensitive to session state. In-session ratios only.
- **2026-09-20 — The CPU ranking did not flip.** 12-thread Xeon, plain TF: this model
  8.7 ms, MobileNetV3 18.7, EfficientNet-B0 39.8. I predicted MobileNet's MACs would win
  on CPU; stock TF depthwise kernels are poor and it did not. Two hardware legs now.
  Unmeasured and stated: TFLite/XNNPACK on a phone, where depthwise is optimised.
- **2026-09-20 — EfficientNet-B0 fine-tuned: 0.9997 on SG2.** The ceiling is ~1.0. It
  judges "real" by what is outside the face (in-face attention 0.47, below uniform).
  It is the distillation teacher.
- **2026-09-20 — A 144-image GradientTape through EfficientNet-B0 OOMs on 24 GB.** Grad-CAM
  attention is computed in chunks of 16 now. Found after a successful fine-tune, not
  before -- the evaluation is the first thing to test on a new backbone.
- **2026-09-20 — The four-model table is complete.** On StyleGAN2, one L4 + one CPU: this
  model is fastest on both by 3.5–7.5×, 3 points of AUC behind the fine-tuned pretrained
  backbones (0.964 vs 0.992–0.9997), and the least background-dependent (swap drop 0.07
  vs 0.10–0.21). Xception is the most face-attentive (0.93) yet drops 0.12 under a
  swapped background: attention and swap measure different things.
- **2026-09-21 — Distilling from the raw EfficientNet-B0 teacher made the student worse:
  0.923 vs 0.964, and it imported the teacher's background reliance** (swap drop 0.21,
  from 0.07). The teacher averages 0.003 / 0.999: saturated, no dark knowledge, and
  matching logits of ±7–14 is a harder target than 0/1. A more accurate teacher is not
  a better one; an informative one is. → calibrate first (test18b); if that fails, the
  size gap is the problem and MobileNetV3 (1M params, 0.992) is the closer teacher.
- **2026-09-21 — The teacher is not over-confident; it is near-perfect.** Temperature
  scaling on val gave T_cal = 1.10: when EfficientNet says 0.995 it is right 99.5% of
  the time. My diagnosis (saturation) was wrong. There is simply no dark knowledge on
  this task at 256² for a teacher that has solved it. What remains is the size gap
  (test18c, MobileNet teacher) or no gain at all from distillation here.
- **2026-09-21 — Distillation closed: two teachers, both worse.** MobileNetV3 teacher
  (the student's size): 0.901, swap drop 0.18, real-recall 0.68. With EfficientNet:
  0.923, drop 0.21. Not a size-gap problem. The setup was the literature's failure
  case (Beyer et al. 2022): offline clean-image scores vs augmented student views,
  in-sample teacher labels, and a background-dependent target. Online distillation on
  the winning pipeline is the one remaining attempt; crops first.
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
