# Hyperparameters — justified as *tried* or *cited*, nothing else

The panel's standard (2026-09-21): a value is justified only by **evidence on this
data**. A paper's value worked on *their* data; citing it is a justified starting
point, not a justified value. So every tunable has one of three answers:

1. **Swept** — {a, b, c} run on this data; the winner is used, and the numbers are here.
2. **Cited + sensitivity-checked** — taken from paper P, then its neighbours run on
   this data and shown to give the same result within seed noise. Two extra runs,
   and it *is* evidence on this data.
3. **Not a choice** — a hardware constraint (batch = what fits), or not a
   hyperparameter at all (seed, shuffle buffer).

"The paper said so", "the default", "the notebook had it" are not answers. The
table records which of the three each value has — with the numbers once run — or
what it still needs.

Status: **✓ passes** (1, 2 or 3 with numbers), **~ cited, not yet checked on this
data**, **✗ nothing yet**.

| parameter | value | status | justification (tried / cited) | what closes the gap |
|---|---|---|---|---|
| `img_size` | 256 | ✓ tried + cited | Compute scales with pixels (1.2 G MACs at 256², 4.8 G at 512²); the efficiency claim needs a fixed budget. Wang et al. 2020 (CNNDetection) train on 224² crops. The native-crop experiment tests the alternative *at equal compute*. | crop experiment (planned) |
| `input_mode` | crop (test19) | ✓ tried: 0.9994 vs 0.964 | The resize is a low-pass that deletes the fingerprint (post-pipeline highfreq 0.52). Crop mode is the controlled test: same input size and latency, native pixels. Patch-based forensics: Chai et al. 2020. | test17 vs test19 is the check |
| `cache_size` | 512 | constraint, then tried | Per-epoch disk read scales with it: 512 → 27 GB, 768 → 62 GB, 1024 → 157 GB. 512 is the cheapest test of the hypothesis; 768 (hair) follows if 512 helps. | 768 run if 512 wins |
| `native_size` | 512 | ✗ | Redundant now that both classes are 1024². No paper, no sweep. | one run with `native_size: 1024` (single resize); expect no change; then keep whichever is simpler |
| `limit_per_class` | 25,000 | ~ | Budget. Bouthillier et al. 2021: split variance dominates; more data mostly buys stability. | lever 4 (40k) gives the second point |
| `batch_size` | 144 | ~ constraint | Largest batch of 256² images that fits the 20 GB L4 with this model (17 GB used). Goyal et al. 2017: batch and LR scale together; at fixed LR, larger batch = better GPU use, not different optimum. | sweep {48, 96, 144} at fixed LR — 3 runs — OR state the memory constraint plainly (panels accept this) |
| `lr` | 3e-4 | ~ | Kingma & Ba 2015 recommend Adam with 1e-3 default; 3e-4 is the widely used conservative value for from-scratch CNNs. Converged without divergence. **Not swept.** | **sweep {1e-4, 3e-4, 1e-3} — 3 runs, ~1.5 h each** (or adopt cosine+warmup, cited: Loshchilov & Hutter 2017, Goyal et al. 2017 — the v2 recipe) |
| baseline `lr` | 1e-4 | ~ cited | Fine-tuning a pretrained backbone: an order of magnitude below from-scratch, to preserve pretrained features (Keras transfer-learning guide uses 1e-5–1e-4; Yosinski et al. 2014). | — |
| `epochs` | 150 | ✓ tried | A cap under early stopping; 50 was hit mid-climb (test16: 0.945 → 0.972 by 85). Never reached since. | — |
| `patience` | 15 | ✓ tried | 6 cut the test16 ablation off at epoch 61 at 0.960; 15 let it reach 0.975. Prechelt 1998 on early-stopping criteria. | — |
| early-stopping metric | val AUC | ✓ (3) | Threshold-free; the metric reported. Prechelt 1998. | — |
| `seed` | 42 | ✓ n/a | Not a tuned value; must be fixed and reported. Bouthillier et al. 2021: report variation across seeds. | seeds 43, 44 (planned) |
| `crop_frac_min` | 0.85 | ✗ | Random-resized-crop is standard (Szegedy et al. 2015 use 8–100% area); 0.85 is the notebook's. | fold into the augmentation sweep below, or cite Szegedy's range and adopt a value from it |
| `aug` | ours (p=0.3 box blur, JPEG q≥60) | ~ cited, **wang run pending** | Wang et al. 2020's recipe (Gaussian σ~U[0,3], JPEG q~U[30,100], p=0.5 each) is their stated generalisation lever; ours is milder. `aug: wang` is now a config value. | **test22_sg2_crop_wang_aug**: same everything, `aug: wang`; scored on SG3-R, then SG3-T once |
| `fake_mix` | — (single generator) / [15000, 10000] (test23) | ✓ stated | Two-generator training at the FakeMix manifest's 60/40 ratio, written in the config rather than inherited from a folder. Ojha et al. 2023 on single-generator overfitting. | test23 vs headline on SG3-R/SG3-T |
| `scale_aug` | — (headline) / [1, 2] (test24) | ~ reasoned, **run pending** | Windows at native and half scale; targets the on-resize failure (0.556). f=4 (the probe's scale) needs cache_size 1024 — unaffordable, stated. Multi-scale training is standard in detection (e.g. SSD/YOLO); the exact factor set is ours. | test24 on the on-resize probe + SG3-R |
| **distill** `online` | false (test18) / true (test25) | ✓ reasoned + cited | Beyer et al. 2022: teacher must see the student's exact augmented view ("consistent teaching"); the offline path violated this and lost to hard labels. Online KD is the corrected experiment, same T/alpha. | test25 vs headline on SG3-R; T/alpha sweep if close |
| blur / JPEG aug (ours) | p=0.3 each; JPEG q 60–100; blur 3×3 avg | ~ cited | **Wang et al. 2020 (CNNDetection)** use exactly these two augmentations for GAN detection: Gaussian blur σ~U[0,3] and JPEG quality~U[30,100], each with p=0.5 (also a p=0.1 variant), and show they are what makes detection generalise. Our values differ (p=0.3, q≥60, box blur). | **either adopt Wang's values (cited) or sweep {off, ours, Wang's} — 3 runs.** Suspected of deleting SG2's fine artifacts; the sweep answers that too |
| dropout | 0.4 | ~ cited | Srivastava et al. 2014: 0.5 for fully-connected hidden units. 0.4 is the notebook's. | adopt 0.5 (cited) or sweep {0.2, 0.4, 0.5} |
| `shuffle_buffer` | 8,192 | ✓ n/a | Engineering, not a model hyperparameter: RAM-bounded (1.6 GB). Any buffer ≥ a few batches gives adequate mixing. | — |
| mask ellipse | rx .38, ry .48, feather .05 | ✓ n/a | Geometry of FFHQ's fixed alignment (Karras et al. 2019); chosen visually to cover the face. Used only by the controls and shortcut checks, not by the headline model. | — |
| `stem_stride` | 1 (headline) / 2 (test21) | ✓ tried + cited | Gragnaniello et al. 2021: early downsampling destroys the GAN fingerprint. **Tried 2026-09-24**: stride 2 gives test AUC 0.9990 vs 0.9996 (acc 0.984 vs 0.99), converging in ~130 epochs instead of ~10, for 4× fewer MACs, CPU 4.0 vs 6.4 ms, 158 vs 171 MB at bs=1, 1.0 vs 2.8 GB at bs=144. The citation holds as "slower and slightly worse", not "destroyed". | seeds decide whether 0.0006 is noise; if so, stride 2 becomes the headline |
| architecture | 32-64-128-256-256, Dense 128, FFT 16-32 | ~ | Doubling widths per stage is the VGG scheme (Simonyan & Zisserman 2014) at half width, sized to a ~1M-param budget. The FFT branch is ablated (tried). Widths and depth are not. | width/depth are design, defended by the parameter budget and the ablation; a width sweep is lever-3 territory (v2) |
| **distill** `temperature` | 2.0 | ~ cited | **Hinton, Vinyals & Dean 2015**: temperatures 1–20 tried; for small students, 2.5–4 worked best; T>1 needed when the teacher is near 0/1 (ours: 0.9997 AUC). 2 is at the low end of their range. | **sweep {1, 2, 4} — 3 runs, ~1 h each with caches on Drive and `init: true`** |
| **distill** `alpha` | 0.7 | ~ cited | Hinton et al. 2015: "considerably lower weight" on the hard-label term than the soft one, i.e. alpha > 0.5. 0.7 is within that, not their number. | **sweep {0.5, 0.7, 0.9} at the best T — 3 runs** |
| **distill** `calibrate` | false | ✓ tried | Guo et al. 2017 temperature scaling was fitted on val: **T_cal = 1.10** — the teacher is not miscalibrated; its 0.013 / 0.995 on unseen val is accurate. Calibration has nothing to fix, so 18b was not run. | measured |
| **distill** `extra` | false | ✓ tried (by design) | Joyson's call: one-variable comparison vs test17 first; `true` (balanced +30k) second. | `true` run (planned) |
| **distill** `init` | false | ✓ by design | From scratch keeps soft-vs-hard a one-variable change. `true` is the cheaper variant for the sweeps. | — |
| **distill** `teachers` | mobilenet_v3_small (18c) | ✓ tried | EfficientNet-B0 (highest AUC, 4.2M params) made the student *worse* (0.923 vs 0.964) and imported its background reliance. Cho & Hariharan 2019: larger teachers do not make better students; Mirzadeh et al. 2020: the teacher-student gap matters. MobileNetV3-Small is the student's size (1.01M) at 0.992. | test18 vs test18c is the check |

## Sweep / sensitivity plan, in priority order (what the panel will ask about first)

Each "~" row needs at least a sensitivity check (the value ± one neighbour) to become
"✓". Record the numbers in the table when they exist.

| sweep | runs | GPU h | why first |
|---|---|---|---|
| distill T {1,2,4} then alpha {0.5,0.9} | 5 | ~5 | the headline's improvement rests on these two numbers |
| `lr` {1e-4, 3e-4, 1e-3} | 3 | ~4.5 | the most-asked hyperparameter; also decides whether v2's schedule is needed |
| augmentation {off, ours, Wang 2020} | 3 | ~4.5 | citable either way; tests the "JPEG deletes the artifacts" suspicion |
| `batch` {48, 96, 144} | 3 | ~4.5 | or accept the memory-constraint justification and skip |
| dropout, crop range | — | — | adopt the cited values (0.5; Szegedy's range) rather than sweep |

Roughly 20 GPU hours for the top three. Each sweep is a config per value, run by
`train.py`, one row each in the results table. With caches in Drive, no run pays a
rebuild.

## References

- Kingma & Ba 2015, *Adam: A Method for Stochastic Optimization*.
- Hinton, Vinyals & Dean 2015, *Distilling the Knowledge in a Neural Network*.
- Wang, Wang, Zhang, Owens & Efros 2020, *CNN-generated images are surprisingly easy to spot… for now* (CVPR) — blur+JPEG augmentation for GAN detection.
- Srivastava et al. 2014, *Dropout: A Simple Way to Prevent Neural Networks from Overfitting* (JMLR).
- Simonyan & Zisserman 2014, *Very Deep Convolutional Networks* (VGG).
- Szegedy et al. 2015, *Going Deeper with Convolutions* — random-resized crop.
- Goyal et al. 2017, *Accurate, Large Minibatch SGD* — batch/LR scaling, warmup.
- Loshchilov & Hutter 2017, *SGDR* — cosine schedule.
- Prechelt 1998, *Early Stopping — But When?*
- Yosinski et al. 2014, *How transferable are features in deep neural networks?*
- Bouthillier et al. 2021, *Accounting for Variance in Machine Learning Benchmarks* (MLSys).
- Karras, Laine & Aila 2019, *A Style-Based Generator Architecture* — FFHQ alignment.
