# Hyperparameters — where each value came from, and why

One row per value that appears in a config. Three honest categories:

- **inherited** — from the original notebooks (Test11 → Test16). The choice
  predates this repo; the justification is the notebook author's, stated here
  as best understood, and the value is kept for comparability with every result
  so far.
- **principled** — chosen for a stated reason, with a source or a measurement.
- **untuned default** — a reasonable value that was *not* swept. Say so; the
  column on the right says what a sweep would cost.

| parameter | value | category | why | tuned? |
|---|---|---|---|---|
| `img_size` | 256 | principled | Compute scales with pixels: this model is 1.2 G MACs at 256², would be 4.8 G at 512². The whole efficiency table lives at 256². Cost: the resize is a low-pass that deletes most of the GAN fingerprint (post-pipeline `highfreq` AUC 0.52) — see the native-crop experiment. | No; crop experiment tests the alternative at equal compute |
| `native_size` | 512 | inherited | Notebook's "normalize native resolution" step, needed when reals were 512 and fakes 1024. Both are 1024 now, so it is a redundant second downsample. Kept for comparability; `native_size: 1024` (identity) is a one-line variant for the crop experiment. | No |
| `limit_per_class` | 25,000 | constraint | Data/units budget. 20k converged smoothly (test16); 25k for the SG2 runs. The pools hold 70k / 40k; distillation uses the rest. | No; more data is lever 4 |
| `batch_size` | 144 | inherited, constraint | The notebook's value; it is the largest batch of 256² images this model fits on a 20 GB GPU with room (17 GB used). Larger batches use the GPU better; 144 was a memory ceiling, not a search. Baselines train at 32 because Xception at 256² does not fit 144 with gradients. | No |
| `lr` | 3e-4 | inherited | Adam with 3e-4 is the widely used from-scratch CNN default (Kingma & Ba's paper default is 1e-3; 3e-4 is the common conservative choice). Empirically fine: converged to 0.964 without divergence. The nine-epoch plateau on SG2 suggests warmup/cosine would help — that is the v2 recipe, lever 3. | No |
| baseline `lr` | 1e-4 | principled | Fine-tuning a pretrained backbone uses a lower LR than training from scratch, to not destroy pretrained features (standard transfer-learning practice; Keras' own guide uses 1e-5–1e-4). | No |
| `epochs` | 150 | principled | A cap that early stopping decides under. 50 (the notebook) was hit while still climbing (test16: 0.945 at 50 → 0.972 at 85). Never reached since. | n/a |
| `patience` | 15 | measured | 6 (the notebook) cut the test16 ablation off at epoch 61 when both runs sat at 0.960; with 15 it climbed to 0.975. Slow noisy curves need a longer window. | Effectively yes (6 → 15) |
| `seed` | 42 | convention | Any fixed value works; what matters is that it is fixed, reported, and drives both the split and the init. Seeds 43/44 are the planned ± on the headline. | n/a |
| `crop_frac_min` | 0.85 | inherited | Test16 changed it from 0.95; random crop keeps 85–100% of the side. Mild translation/scale augmentation. | No |
| `shuffle_buffer` | 8,192 | constraint | ~1.6 GB of cached uint8 in RAM. The notebook shuffled the whole set (fits at 16k images, not at 35k on a standard runtime). | No |
| blur / JPEG aug | p=0.3 each, JPEG q 60–100 | inherited | Robustness to real-world degradation and a defence against a compression shortcut. Suspected of *deleting* StyleGAN2's fine artifacts in a third of training images — untested; a legitimate ablation. | No |
| dropout | 0.4 | inherited | On the 128-unit fusion layer. Notebook value. | No |
| mask ellipse | rx 0.38, ry 0.48, feather 0.05 | inherited | The notebook's cv2 ellipse for aligned FFHQ faces; feather avoids a hard edge. Used only by the face/background controls and the shortcut checks. | No |
| architecture widths | 32-64-128-256-256, Dense 128, FFT 16-32 | inherited | The notebook's design. Its cost profile (no stride-2 stem → 1.2 G MACs, 1.3 GB peak) is measured and discussed; the FFT branch is ablated. | Widths: no |
| **distill** `temperature` | 2.0 | untuned default | Hinton et al. 2015 use T in 1–20. The teacher is at 0.9997 AUC, so its probabilities sit near 0/1; T>1 spreads them so the ranking among confident cases survives. 2 is a conservative common value. | **No — sweep {1, 2, 4}: ~1 h each with caches on Drive and `init: true`** |
| **distill** `alpha` | 0.7 | untuned default | Weight on the soft term; Hinton recommends weighting it more than the hard term. 0.5–0.9 is the usual range. | **No — sweep {0.5, 0.7, 0.9}, same cost** |
| **distill** `extra` | true | principled | The MobileNet result showed the gap is the training regime, not the data ceiling; the unused images are labelled and free. Taken *balanced* — 15k per class, all the spare fakes — so training is 65k, evenly split (the raw leftovers are 45k/15k). `false` gives the clean soft-vs-hard comparison at 35k. | Both to be run |
| **distill** `init` | false | principled | From scratch keeps "same model, soft labels vs hard" a one-variable comparison against 0.964. `true` (warm start) is cheaper and likely as good; run second. | Both to be run |
| **distill** `teachers` | efficientnet_b0 | measured | Highest test AUC of the three (0.9997 vs 0.9987 / 0.992). Averaging all three is an option. | No |

## The honest summary for a panel

Most training values are the notebook's, kept for comparability, and were not
searched. Two were changed on evidence (epoch cap, patience). The distillation
values are literature defaults and are the ones most worth sweeping — cheap once
the caches are in Drive. The architecture is ablated, not tuned.
