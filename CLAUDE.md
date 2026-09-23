# CNN-based GAN face detection — project brief for Claude

Read this first, then `README.md` (results tables, how to run), `docs/learnings.md`
(every finding, dated), `docs/hyperparameters.md` (why each config value). Those three
files are the record; do not re-derive what they already say.

## The claim (do not drift from it)

A ~1M-parameter CNN detects StyleGAN2 faces (vs FFHQ) with accuracy competitive with
detectors 4–20× larger (Xception, EfficientNet-B0, MobileNetV3-Small), measured on the
**identical task and same GPU**: params, MACs, latency (bs=1 median/p95), peak memory,
CPU latency. "Edge deployment" was dropped by the panel — the claim is resource
comparison, not deployment. Cross-generator generalisation is **out of scope** and
reported honestly (StyleGAN3-T AUC 0.70).

Headline model: `configs/test19_sg2_crop_no_fft.yaml` — five plain conv blocks, no FFT
branch, native-resolution 256² crops. Test AUC 0.9996, 1.01M params, ~1 ms on L4.

## Standards the panel holds us to

- **Every hyperparameter needs a written why**: tried on this data, or cited — and a
  citation is only a starting point that must be verified on this data. Record in
  `docs/hyperparameters.md` (tried / cited / sensitivity status).
- **Every added file or tool must justify itself** in one sentence, or it is dropped.
  Flat layout, plain YAML, no Hydra/MLflow/W&B/DVC.
- **Cite the test column, not val.** Checkpoint and early stopping select on val_auc.
- **Latency is measured, not inferred from MACs.** Same session, same GPU, record the
  system under test (`eval.json` does). CPU numbers vary across Colab VMs — never
  compare them across sessions.
- **Any surprising accuracy is a shortcut until proven otherwise.** The chain so far:
  reals 512²/fakes 1024² asymmetry (0.9995, shortcut) → clean data 0.964 → native crops
  0.9996, checked by native audit, JPEG probes, held-out generator.
- Record every finding in `docs/learnings.md` as a dated bullet; keep README tables
  current; commit after each result.

## What the pipeline is

- Data: FFHQ 1024² (`Real(FFHQ)`, 70k) + official NVIDIA StyleGAN2 ψ=1.0
  (`Fake(SG2-psi1)`, 40k), 25k/class sampled by seed, 70/15/15 stratified split.
- `input_mode: crop`: centre 512² native pixels cached as uint8, random 256² windows in
  training, centre window at eval, **no resampling**. Resize mode (1024→512→256) is the
  old pipeline; it deletes the fingerprint (0.964).
- Operating condition: native-resolution input only. On downscaled input the crop model
  is at chance (0.556). Stated, not hidden.
- FFT branch: tested four ways (mix, resized SG2, native SG2, StyleGAN3 held-out); buys
  0 / +0.011 / 0 / +0.008. Dropped from the headline; kept as `fft_branch: true` ablation.
- Distillation (test18*) was worse than hard labels; documented, not pursued.

## Where things live

- Colab VM paths (all configs): reals `/content/drive/MyDrive/Real(FFHQ)`, fakes
  `/content/drive/MyDrive/Fake(SG2-psi1)`, StyleGAN3 held-out
  `/content/drive/MyDrive/Fake(SG3-T-psi1)` (1,500 imgs), outputs
  `/content/drive/MyDrive/Research/experiments/<config>/`, local cache `/content/cache`.
  The whole group works from **one shared Colab Pro account and one Drive**, so these
  paths resolve unchanged for everyone. Consequences: compute units are shared (one
  long GPU job at a time, coordinate before starting one), and a second person
  connecting can attach to the same running VM — check `ps aux | grep python` and
  `tail` the `/content/*.log` files before assuming the machine is idle.
- Crop cache (~33 GB) is synced to Drive; on a fresh VM run
  `python cache_sync.py --config configs/test19_sg2_crop.yaml --from-drive` (≈20 min)
  before any crop-pipeline training, or it re-decodes 50k PNGs.
- Colab workflow: `caffeinate -i` on the Mac, train from the VM terminal under
  `nohup python -u … &`, `tail -f` the log, `--resume` after a VM reclaim
  (README "Running (Colab)"). VMs get reclaimed often; expect it.

## Done

- Headline + FFT ablation on crops (test19), resize pipeline (test17), preliminary
  mix (test16), distillation (test18, 18c), on-resize probes, JPEG probes, native and
  post-resize dataset audits, StyleGAN3-T held-out (both models, 2026-09-22).
- Baselines fine-tuned and evaluated on the **resize** pipeline only (README
  "Baselines"): MobileNet 0.992, Xception 0.9987, EffNet 0.9997. Their efficiency
  numbers are final; their accuracy numbers are not comparable to the crop headline.
- Efficiency table for all four models on L4 + Xeon CPU (final).

## Still to do, in order

1. **Baselines on the crop pipeline** (accuracy column only re-runs), under
   `configs/test20_crop_baselines.yaml` (headline config renamed; same cache key):
   `nohup python -u baselines.py --config configs/test20_crop_baselines.yaml --model efficientnet_b0 --train > /content/bl_effb0.log 2>&1 &`
   then `xception`. ~30 epochs each at batch 32, ~70 min. **mobilenet_v3_small done
   2026-09-23: test AUC 0.9999** (README Baselines, learnings). Check
   `experiments/test20_crop_baselines/baselines/<model>/` for `eval.json` before
   re-running; `--resume` if `model.keras` exists without it.
2. **Baselines on StyleGAN3-T**: `heldout.py` needs a `--model <baseline>` option
   (loads the baseline's checkpoint instead of the headline's) — not written yet.
   Answers whether the generalisation gap is shared by the big models (expected yes).
3. **Seeds 43 and 44** of the headline: `configs/test19_sg2_crop_no_fft_s43/_s44.yaml`
   via `train.py`, then `evaluate.py`; report mean ± std of test AUC.
4. **Held-out reals** (CelebA-HQ 1024) with `heldout.py --real-dir … --tag celebahq`.
5. **Sensitivity checks** on the crop pipeline for `docs/hyperparameters.md`:
   lr {1e-4, 1e-3}, augmentation {off, Wang 2020's}. One run each, note the number.
6. Controls test13_face / test14_background are `_pending_` in README; only re-run
   if the panel asks again about background dependence (crops make them moot).
7. Not planned unless time: `cache_size: 768`, online distillation, scale
   augmentation, BN+cosine recipe. Listed as future work.

## Gotchas already paid for

- `history.csv` must be appended-and-closed per epoch (Drive uploads on file close);
  `HistoryCSV` callback does this. Do not reintroduce CSVLogger.
- `cache_dataset_ops "will be discarded"` warning is benign.
- Drive mount can list `Real(FFHQ)` as empty on a fresh VM: wait and retry.
- Grad-CAM/attention on EfficientNet OOMs at batch 144 → chunks of 16 (done).
- StyleGAN3 CUDA plugins do not compile on Colab; the generator ran with `_init()`
  patched to `return False` in bias_act/upfirdn2d/filtered_lrelu (~7 s/img).
- Baseline `--train` uses batch 32 / lr 1e-4 / 30 epochs; eval batch = headline's.

## Commit convention

Message = what changed and the number it produced, e.g.
`results: held-out StyleGAN3-T -- AUC 0.703 no-FFT / 0.711 FFT`. Push to `main`.
