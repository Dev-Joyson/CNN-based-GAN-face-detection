# CNN-based GAN face detection — project brief for Claude

Read this first, then `README.md` (results tables, how to run), `docs/learnings.md`
(every finding, dated), `docs/hyperparameters.md` (why each config value). Those three
files are the record; do not re-derive what they already say.

## The claim (do not drift from it)

A ~1M-parameter CNN detects StyleGAN2 faces (vs FFHQ) with accuracy competitive with
detectors 4–20× larger (Xception, EfficientNet-B0, MobileNetV3-Small), measured on the
**identical task and same GPU**: params, MACs, latency (bs=1 median/p95), peak memory,
CPU latency. "Edge deployment" was dropped by the panel — the claim is resource
comparison, not deployment. Cross-generator generalisation is the **stated limit**:
StyleGAN3-T AUC 0.70 vs the pretrained baselines' 0.96–0.99 (todo 2c tries to close it
without changing the architecture).

Headline model: `configs/test19_sg2_crop_no_fft.yaml` — five plain conv blocks, no FFT
branch, native-resolution 256² crops. Test AUC 0.9998 ± 0.0002 over three seeds
(0.9996 on seed 42), 1.01M params, ~1 ms on L4, 171 MB at bs=1. Stride-2 variant
(test21): 0.9988 ± 0.0002 at 4× fewer MACs — the reported cheap variant, not the headline.

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
- Baselines fine-tuned on both pipelines (README "Baselines"): resize MobileNet
  0.992 / Xception 0.9987 / EffNet 0.9997; **crop (test20) 0.9999 / 1.0000 / 1.0000**.
  Efficiency numbers final. On crops accuracy saturates for all four models.
- Efficiency table for all four models on L4 + Xeon CPU (final; memory column
  re-measured 2026-09-24).

## Still to do, in order

1. ~~Baselines on the crop pipeline~~ **done 2026-09-23** under
   `configs/test20_crop_baselines.yaml` (headline config renamed; same cache key),
   `experiments/test20_crop_baselines/baselines/<model>/`: MobileNetV3-Small 0.9999,
   EfficientNet-B0 1.0000, Xception 1.0000 (test AUC). Accuracy saturates on native
   pixels for every model; see README Baselines "Crop column, read honestly".
2. ~~Re-measure peak memory~~ **done 2026-09-24**, one L4 session: headline 171 MB at
   bs=1, MobileNet 152, EffNet 205, Xception 409 (old 1,301 was autotune scratch).
   README table updated. Remaining honest weakness: batch-144 memory 2.8 GB vs 0.67.
2b. ~~Baselines on StyleGAN3-T~~ **done 2026-09-24**: MobileNet 0.958, EfficientNet
   0.993, Xception 0.967 vs ours 0.703. The gap is NOT shared — pretrained features
   transfer, from-scratch ones do not (README *Held-out generator*). This is the
   thesis's stated limit unless 2c moves it.
2c. **Generalisation at the same architecture — ALL FOUR levers, decided by Joyson
   2026-09-25, to run right after the seed chain.** The research question becomes: a
   1M from-scratch model matches pretrained detectors in-distribution; what does it
   lose on unseen generators (SG3-T 0.70 vs 0.96–0.99) and degraded input (chance on
   downscaled), and can training alone, at zero inference cost, recover it? Levers,
   latency/memory unchanged: (i) Wang et al. 2020 augmentation — blur σ~U[0,3] and
   JPEG q~U[30,100], each p=0.5 — as a config; (ii) two-generator training (SG2 +
   the 10k SG1 that FakeMix's manifest identifies) with SG3-T held out. **Rule: SG3-T
   is the final test, scored once per model. SG3-R generated 2026-09-25: 1,500 images,
   ψ=1.0, seeds 200000–201499, official `stylegan3-r-ffhq-1024x1024.pkl`, reference ops,
   at `/content/drive/MyDrive/Fake(SG3-R-psi1)` — the selection set.**
   Configs written 2026-09-24: `test22_sg2_crop_wang_aug.yaml` (headline's cache) and
   `test23_sg2_sg1_crop.yaml` (new cache; needs `Research/Dataset New/FakeSG1` with
   ≥10,000 images — verify, and record the SG1 set's provenance/ψ under README Data).
   **Reference on SG3-R (2026-09-27): headline 0.742, stride-2 0.686.** FakeSG1 has
   23,000 images; test23 now uses `fake_mix: [12500, 12500]` (equal share).
   Run: `train.py` → `evaluate.py` → `heldout.py --fake-dir "…/Fake(SG3-R-psi1)" --tag sg3r`;
   compare to 0.742; the winner (if any) is scored on sg3t once.
   **test22 from scratch: never learned (0.5, 16 epochs, early-stopped, 2026-09-27).**
   → `init_from` knob added to Config/train(); `test22b_wang_aug_from_headline.yaml`
   warm-starts from the headline's weights. **test22b done 2026-09-27: SG2 0.9997,
   SG3-R 0.772 (+0.03 over 0.742; small, real at ±0.01). Lever (i) = marginal.**
   test22b early-stopped on saturated SG2 val while still adapting → `checkpoint: last`
   knob + `test22c_wang_aug_continue.yaml` (30 fixed epochs from test22b, score SG3-R
   after; chain again if rising). Next: test23 (running; at 0.5000 through epoch 10 — if it dies at 16 unstarted,
   rerun as `test23b_sg2_sg1_crop_es40.yaml`: same run, `es_start_epoch: 40`, same
   cache — test23 DID die at 16, test23b launched 2026-09-27 ~11:45; if test23b is
   still at 0.5000 at epoch 40, run `test23c_sg2_sg1_crop_bn.yaml` (`batchnorm: true`,
   same cache; BN folds at inference, +0.2% params)). **test23c (BN) DID start: best
   val 0.9990 at epoch 45, VM lost at 46, resumed 2026-09-28.** After its SG3-R score,
   Joyson's agreed order ("remember it"): (1) Wang aug warm-started from test23c
   (as test22b did from the headline); (2) test24 scale aug stacked; (3) test25 online
   KD from EfficientNet; (4) a THIRD generator outside the StyleGAN1/2 family if
   diversity moved the number (SG3-R is the selection set — never train on it);
   (5) BN-only headline run to attribute BN vs mix. A big SG3-R gain pulls (4)
   forward; a small one pulls (3) forward. test22c stays in the queue.
   **test23c result 2026-09-27: SG3-R 0.777 (+0.035), SG2+SG1 test 0.9999.** Small
   gain → (3) online distillation: test25 from scratch sat at chance for 5 epochs then
   STARTED, best epoch 58, VM lost at 66, finalized from best weights. **Result
   2026-09-28: SG2 0.9987, SG3-R 0.666 — WORSE than the headline (0.742).** Copies the
   teacher's decisions, not its representation. test25b not needed. NEXT: test24 scale
   aug (running 2026-09-28 06:00, VM 9a451c05aa64; started on its own at ~epoch 12; score
   with `evaluate.py --config configs/probe_test24_on_resize.yaml --tag on_resize` for the
   downscaled probe vs 0.556, plus sg3r), then
   `test26_sg1sg2_wang_from_23c.yaml` (written 2026-09-28: Wang aug warm-started from
   test23c, 30 fixed epochs, checkpoint last; needs test23's cache), then BN-only
   headline, then quantization. Distillation+Wang aug: low priority, only if budget (needs the headline cache synced +
   `test20_crop_baselines/baselines/efficientnet_b0/model.keras`, on Drive). Then
   test24 scale aug, then stacking (Wang aug from test23c), then the BN-only headline
   for attribution, then quantization.
   (iii) **scale augmentation** — `scale_aug: [1, 2]`, windows at native and half
   scale (¼ would need cache_size 1024; stated), `configs/test24_sg2_crop_scale_aug.yaml`
   (written 2026-09-25), targets the downscaled-input failure; (iv) **online
   distillation** from EfficientNet-B0 (the baseline that generalises, 0.993 on SG3-T):
   `distill.online: true`, teacher and student see the identical augmented batch
   in-graph (`OnlineKD` in distill.py; `StudentCheckpoint` saves student-only weights so
   `--finalize` works), `configs/test25_online_kd_effnet.yaml` (written 2026-09-25;
   needs `experiments/test20_crop_baselines/baselines/efficientnet_b0/model.keras`, which
   exists). All four configs are on the stride-1 stem; if stride 2 wins the seeds, add
   `stem_stride: 2` to each before running. Order: (i) test22 → (ii) test23 → (iii) → combine the
   winners → (iv). Select on SG3-R, score SG3-T once per model. Also score each on the
   on-resize probe (`--tag on_resize`) since (iii) targets it.
3. ~~Stride-2 stem, test21~~ **trained + evaluated 2026-09-24**: test AUC 0.9990
   (acc 0.984) vs headline 0.9996; 0.28 G MACs, CPU 4.0 ms, 158 MB bs=1, 1.0 GB bs=144,
   5,547 img/s; GPU bs=1 unchanged. Best epoch 129 of 150 (plateaued). JPEG q95 0.9978
   (−0.001), StyleGAN3-T 0.735 — same as the headline on both. **Headline decision
   pending the seeds** (chain started 2026-09-24 19:07 on VM 224c029a330e:
   test21 s43, s44, then test19 s43, s44; log `/content/seeds.log`): if
   0.0006 AUC is within seed noise, stride 2 is the better headline (lightest on every
   column but MACs vs MobileNet).
4. ~~Seeds 43 and 44~~ **done 2026-09-25 — headline stays stride 1.** Stride 1:
   0.9996 / 0.9998 / 0.99995 = 0.9998 ± 0.0002; stride 2: 0.9990 / 0.9987 / 0.9987 =
   0.9988 ± 0.0002. Gap 0.001 = five spreads. All generalisation configs (test22–25)
   stay on stride 1 as written. Original plan text follows for the record.
   Seeds 43 and 44 — now decide the headline. Run them for BOTH stems if budget
   allows (stride 1: `configs/test19_sg2_crop_no_fft_s43/_s44.yaml`; stride 2: copies
   with `stem_stride: 2`, to be written), else stride 2 first since it trains at 27 s/epoch.
   Report mean ± std of test AUC per stem; the cheaper stem wins a tie.
   **Progress:** stride-2 DONE 2026-09-25: s42 0.9990 / s43 0.9987 / s44 0.9987 →
   **0.9988 ± 0.0002** (acc 0.985; s43 climbed out of a 5-epoch chance start; s44 hit
   the 150 cap after a VM reclaim + `--resume` from epoch 133). Stride-1 s43 started
   2026-09-25 ~07:10 (cache build), s44 queued (`/content/seeds.log`, VM 75bedbb4438c).
   Headline single number 0.9996 is 4 spreads above the stride-2 mean.
4b. **Int8 quantization of all four models, at the END** (decided 2026-09-27): post-
   training TFLite int8, no retraining; CPU latency + SG2 accuracy + SG3-T per model at
   equal precision. Answers "why not compress EfficientNet, which generalises?".
   Applied to the FINAL student, so after 2c; the three baselines can be done earlier
   as a CPU side job. Script not written yet.
5. **Held-out reals** (CelebA-HQ 1024) with `heldout.py --real-dir … --tag celebahq`.
6. **Sensitivity checks** on the crop pipeline for `docs/hyperparameters.md`:
   lr {1e-4, 1e-3}, augmentation {off, Wang 2020's}. One run each, note the number.
7. Controls test13_face / test14_background are `_pending_` in README; only re-run
   if the panel asks again about background dependence (crops make them moot).
8. Not planned unless time: `cache_size: 768`, online distillation, scale
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
