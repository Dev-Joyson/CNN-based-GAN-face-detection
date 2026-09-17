# How to run a small ML research repo — what the evidence actually says

Scope: this repo. A ~1M-parameter dual-branch CNN (spatial convs + FFT branch) separating
StyleGAN2 faces from FFHQ, TensorFlow-Keras, trained on Colab, dataset in Drive, experiments
driven by `configs/*.yaml`. Three people, ~830 lines of Python, thesis deadline.

The research claim is a **resource comparison**: competitive accuracy at lower hardware cost
than Xception / EfficientNet / ResNet-class detectors. Everything below is judged against
that claim, not against a generic "best practices" ideal.

**Evidence tiers used throughout.** Most ML-repo advice is convention wearing the costume of
evidence, so each claim is tagged:

- **[E] Evidence-backed** — a study measured an outcome.
- **[C] Convention** — broad consensus across authorities, never measured.
- **[O] Opinion** — a named author's or vendor's stated preference, self-declared as such.
- **[U] Unsourced** — community folklore; I could not find a primary source.

---

## 0. The one finding that matters most

**Verified by arithmetic against this repo's own architecture.**

`build_model` in [`model.py`](../model.py) is 5 plain 3×3 convolutions at 32→64→128→256→256
with pooling between them, over a 256×256 input, plus a 2-conv FFT branch and a 128-unit head.
Counting multiply-accumulates analytically (conv MACs = H·W·C_out·k²·C_in):

| stage | MACs |
|---|---|
| conv 3→32 @ 256² | 56.6 M |
| conv 32→64 @ 128² | 302.0 M |
| conv 64→128 @ 64² | 302.0 M |
| conv 128→256 @ 32² | 302.0 M |
| conv 256→256 @ 16² | 151.0 M |
| FFT branch (2 convs) | 84.9 M |
| dense head | 0.04 M |
| **total** | **≈ 1.20 GMAC (≈ 2.4 GFLOP)** |

The same arithmetic reproduces **1,020,417 parameters**, matching the README's "1.02M", which
confirms the model of the architecture is right.

Now put that next to the baselines, from the EfficientNet paper's own Table 2
(<https://ar5iv.labs.arxiv.org/html/1905.11946>), whose "FLOPs" column is multiply-adds:

| model | params | MACs | input |
|---|---|---|---|
| **this repo** | **1.02 M** | **≈ 1.20 G** | 256² |
| EfficientNet-B0 | 5.3 M | 0.39 G | 224² |
| ResNet-50 | 26 M | 4.1 G | 224² |
| Xception | 23 M | 8.4 G | 299² |

**This model has 5× fewer parameters than EfficientNet-B0 and roughly 3× more compute.**
Normalising B0 up to 256² (×(256/224)² ≈ 1.31) still only gets it to ≈0.51 GMAC. The reason is
structural: there is no stride-2 stem and no bottleneck, so the early convolutions run at full
256² and 128² resolution, where MACs are dominated by spatial size, not channel count. Parameter
count does not see that; MACs do.

This is exactly the failure mode the efficiency literature names. Dehghani et al., *The
Efficiency Misnomer* (ICLR 2022, <https://arxiv.org/abs/2110.12894>) shows model rankings
**invert** depending on which cost indicator you pick, and warns specifically that parameter
count is misleading because "a model can have very few trainable parameters and still be very
slow". Schwartz et al., *Green AI* (CACM 2020, <https://arxiv.org/abs/1907.10597>) rejects
parameter count as an efficiency metric outright: "different algorithms make different use of
their parameters". **[E]** for the arithmetic, **[E]** for the cited critiques.

**Consequence for the thesis, stated plainly:** the "1.02M parameters" headline is a real and
defensible claim about *model size and memory*. It is **not** a claim about compute, and against
EfficientNet-B0 specifically the compute claim runs the other way. The measured latency may
still favour this model — at batch size 1 a small model is often launch-latency-bound rather
than compute-bound, so 3× the MACs need not mean 3× the wall-clock — but that is a hypothesis
the repo has not yet tested, and a panel that computes the MACs will get there before you do.

The fix is not to weaken the claim. It is to **measure MACs and latency together and say which
resource you are saving**: parameters and model file size (yes, clearly), peak memory (measure
it), wall-clock latency at bs=1 (measure it), total compute (probably not, versus B0). A claim
that is precise about *which* resource is far stronger than a vague one that a reviewer can
puncture with one arithmetic check.

---

## 1. Ranking — value to defending the thesis, over effort

Effort in units a student recognises: "minutes" = edit and commit; "runs" = Colab GPU-hours.

| # | Change | Value | Effort | Ratio |
|---|---|---|---|---|
| 1 | Report **test** AUC as the headline, not best-epoch val AUC | Very high | ~0 (number already in `eval.json`) | ★★★★★ |
| 2 | Add **MACs/FLOPs** to `measure_efficiency` alongside params | Very high | ~30 min | ★★★★★ |
| 3 | Record the **system under test** (GPU model, TF/CUDA version, precision, XLA, batch, n, warmup) into `eval.json` | Very high | ~20 min | ★★★★★ |
| 4 | Raise latency `runs` from 100 to ≥1000, or drop p95 | High | 1 line, ~seconds of GPU | ★★★★★ |
| 5 | Build the **baseline runner** — Xception/EfficientNetB0/ResNet50, same session, same GPU, same split, same input size, same `tf.function` wrapper | Very high (the claim does not exist without it) | ~half a day + 1 run | ★★★★☆ |
| 6 | Finish the **`test14_background` control run** | Very high | 1 run | ★★★★☆ |
| 7 | Run **3 seeds** (42/43/44) and report median + range | High | 3 runs | ★★★★☆ |
| 8 | Latency sweep at **bs = 1, 8, 32** | High | ~20 min + minutes of GPU | ★★★★☆ |
| 9 | Write **git SHA + TF version + GPU + wall-clock + timestamp** into `metrics.json` | Medium-high | ~15 min | ★★★★☆ |
| 10 | **Held-out generator** test set (e.g. StyleGAN3 / a diffusion model) | Highest scientific value | 1–2 days incl. data | ★★★☆☆ |
| 11 | Generate the README results table **from the run folders** instead of by hand | Medium-high | ~20 min | ★★★★☆ |
| 12 | Put `img_size`/`native_size` into the **cache key** | Medium (prevents a silent-wrong-results bug) | 1 line | ★★★★☆ |
| 13 | Make `pytest` invocation robust (`pythonpath = ["."]` or `python -m pytest`) | Low-medium | 3 lines | ★★★☆☆ |
| 14 | TensorBoard callback | Low (history.csv already covers it) | 1 line | ★★☆☆☆ |
| 15 | Hydra / MLflow / W&B / DVC / `src/` package / Docker / CI | **Negative at this scale** | days | ✗ |

---

## 2. What is already right — do not touch

This is the most important section, because the default failure mode of advice like this is to
generate work. Each item below is *defended*, not just permitted.

**The flat layout with `model.py` as one source of truth.** This is not a compromise; it is
literally the shape the most-cited primary source for this exact project profile recommends.
Wilson et al., *Good Enough Practices in Scientific Computing* (PLOS Comp Bio 2017,
<https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005510>) defines its
audience as "researchers who are working alone or with a handful of collaborators on projects
lasting a few days to several months" and its Box 3 worked example is *one functions module plus
one driver script*: `src/sightings_analysis.py` + `src/runall.py`. That is `model.py` +
`train.py`. **[C]** Keep it. (See §3 for why `src/` as a Python *package* would be a step
backwards here.)

**Two entry points with `--config`, and nothing else.** `train.py` is 19 lines and does exactly
one thing. Wilson et al.: "For a small project with 1 main output, a single controller script
should be placed in the main `src` directory and distinguished clearly by a name such as
'runall'." **[C]**

**Plain YAML + a frozen dataclass-ish `Config` with `__post_init__` validation.** See §4. No
change warranted.

**Run folders that are self-describing.** `experiments/<name>/` holds `model.keras`,
`history.csv`, `metrics.json` (which already stores `asdict(cfg)` — the full config, not a
summary), plus the eval figures and `eval.json`. This satisfies four of the five items in the
Papers with Code *ML Code Completeness Checklist*
(<https://github.com/paperswithcode/releasing-research-code>): dependency spec, training code,
evaluation code, pre-trained model. **[C]**

**`evaluate.py` rebuilding data through `model.build_datasets`.** The comment in the file —
"an eval that quietly skipped the mask would report a number the model never earned" — is
correct and the design is right. Do not add a separate eval pipeline.

**The two guard tests.** `test_fft_axes.py` and `test_mask_applied.py` are regression tests for
bugs that actually happened, run in ~3s with no GPU or dataset. This is Wilson et al.'s "turn
bugs into test cases" (from *Best Practices for Scientific Computing*, PLOS Biology 2014,
<https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1001745>). Two tests that
catch two real bugs beat forty tests that catch nothing. **[C]** Do not chase coverage.

**`audit_dataset.py`.** Scoring every trivial file property as an AUC on the same scale as the
model's own metric is a genuinely good idea and directly addresses the shortcut risk. Keep.

**The efficiency measurement methodology already in `measure_efficiency`.** Four things it does
are exactly right and each has a source:

1. **Batch size 1 for the latency number.** Matches MLPerf Inference's SingleStream scenario
   (samples/query = 1) and MobileNetV3's protocol — "All latencies are in ms and are measured
   using a single large core with a batch size of one"
   (<https://ar5iv.labs.arxiv.org/html/1905.02244>). **[C]**
2. **Warm-up before timing.** Justified empirically: the PyTorch benchmarking recipe
   (<https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html>) documents a first-call
   penalty of **2775.5 µs vs 22.4 µs** on the second call — ~124× — because cuBLAS loads on
   first use. **[E]**
3. **Forcing device synchronisation inside the timed region** (via `.numpy()`). PyTorch's CUDA
   semantics doc states the general CUDA fact: "time measurements without synchronizations are
   not accurate" (<https://docs.pytorch.org/docs/stable/notes/cuda.html>). Without a sync you
   time kernel launches. **[E]** *Caveat: this is documented by PyTorch, not TensorFlow — I
   could not find equivalent official TF benchmarking guidance. It is a property of CUDA, not of
   PyTorch, so the reasoning transfers, but cite it as such.*
4. **Reporting a percentile, not only a mean.** MLPerf's SingleStream metric is the 90th
   percentile latency, and `torch.utils.benchmark` reports median for the same reason: the tail
   is what binds and a mean over small n is dragged by launch jitter. **[C]**

**Peak memory via `tf.config.experimental.get_memory_info('GPU:0')['peak']` with
`reset_memory_stats` first.** This is the *correct* way and the obvious alternative is wrong.
TF's own docs state: "For GPUs, TensorFlow will allocate all the memory by default... The dict
specifies only the current and peak memory that TensorFlow is actually using, not the memory
that TensorFlow has allocated on the GPU"
(<https://www.tensorflow.org/api_docs/python/tf/config/experimental/get_memory_info>). So
`nvidia-smi` measures the allocator, not the model. The repo got this right. **[E]**

**Timing `tf.function(model(x, training=False))` rather than `model.predict()`.** Correct —
`predict()` measures Keras dispatch. Just make sure every baseline uses the identical wrapper.

**Measuring baselines in the same session on the same GPU** — already stated as a requirement in
the README. This is non-negotiable and Google says why in writing: "The types of GPUs and TPUs
that are available in Colab vary over time"
(<https://research.google.com/colaboratory/faq.html>). A latency number copied from another
paper or another session is inadmissible. **[E]** The README already knows this. Good.

**Grad-CAM as a shortcut check rather than decoration.** Correct framing.

---

## 3. Repo layout — is one `model.py` defensible?

**Yes, and the burden of proof is on the other side.**

### What the primary sources actually say

Wilson et al. 2017 (link above) recommend: project in its own directory; `doc/`, `data/`,
`results/`, `src/`, `bin/`; meaningful filenames. But note two things. First, their `src/` holds
two loose `.py` files — no `__init__.py`, no package, no installation. Second, on where exactly
to put things they say: *"The answer is that it doesn't matter, as long as each team's projects
follow the same rule. As with many of our other recommendations, consistency and predictability
are more important than hairsplitting."* **[C]**

Their "What we left out" section is the direct warning against the over-engineering instinct,
and it is worth quoting at length because it is aimed precisely at a 3-person student project:

> "One important observation about this list is that many experienced programmers actually do
> some or all of these things even for small projects... **The problem comes when those
> experienced developers give advice to people who haven't already mastered the tools and don't
> realize (yet) that they will save time if and when their project grows.** In that situation,
> advocating unit testing with coverage checking and continuous integration is more likely to
> overwhelm newcomers rather than aid them."

They explicitly leave out build tools ("newcomers can achieve the same behavior by writing shell
scripts... given the speed of today's machines, that is unimportant for small projects"),
branches, coverage, CI, and comprehensive documentation.

Noble 2009 (<https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000424>)
proposes a similar layout and is explicit that it is opinion: *"I do not claim that the
strategies I outline here are optimal. These are simply the principles and practices that I have
developed over 12 years of bioinformatics research."* **[O]**

cookiecutter-data-science declares itself opinion in its first line:
*"The default project structure reflects certain opinions about how to do collaborative data
science work. These opinions grew out of our own experiences with what works and what doesn't."*
(<https://cookiecutter-data-science.drivendata.org/opinions/>) It cites no study. It also tells
you to shrink it: *"be liberal in changing the folders around for **your** project, but be
conservative in modifying the default cookiecutter structure for **all** projects."* And on
tracking, its own advice for small projects: *"For smaller projects, it's fine to start with
homegrown tracking using file formats like JSON that are both human- and machine-readable. You
can graduate to experiment tracking tools (e.g., MLflow) if it's warranted."* **[O]** — that
sentence describes what this repo already does.

The Turing Way (<https://book.the-turing-way.org/reproducible-research/compendia/>) separates
input / methods / output and distinguishes a *basic* compendium (data, analysis, README) from an
*executable* one (adds Docker, Makefile, tests, CITATION). It scales guidance by project need
rather than mandating one layout. **[C]**

### Is there evidence that layout matters?

**No.** I searched for empirical software engineering work measuring directory layout against
reproducibility, defect rate, or onboarding time, and found none that treats layout as an
independent variable. Any claim that "`src/` reduces defects" or "a package layout speeds
onboarding" should be treated as **[U]**.

What *is* measured points at paths and dependencies, not folders:

- **Trisovic et al., *Scientific Data* 2022** (<https://www.nature.com/articles/s41597-022-01143-6>):
  2,000+ replication datasets re-executed in clean containers. "We find that 74% of R files
  failed to complete without error in the initial execution, while 56% failed when code cleaning
  was applied." Automatic cleaning "fixed all errors related to the command `setwd`" — i.e.
  hard-coded working directories were a top mechanical cause. Their recommendation: "Use relative
  file paths in your code." Also notable for scale calibration: the median replication dataset
  has 8 files and 2 R files, with a median 160 lines per file. **Real published research code is
  tiny and flat.** **[E]**
- **Pimentel et al., MSR 2019** (<https://www.ic.uff.br/~leomurta/papers/pimentel2019a.pdf>):
  1.16M Jupyter notebooks from GitHub; "only 24.11% executed without errors and only 4.03%
  produced the same results." Top exceptions include `FileNotFoundError`/`IOError` in 12.59% of
  notebooks, "when users use absolute paths to access data files". Their best-practice list
  includes "Abstract code into functions, classes, and modules and test them", "Declare the
  dependencies in requirement files and pin the versions", and "Use relative paths". **[E]**

Both studies endorse what this repo already did — code lives in modules, not notebooks; the
notebook in `notebooks/` is a launcher and explicitly "never method code". That is the
empirically-supported move, and it is done.

### Would a `src/` **package** layout be better here?

No — and PyPA documents the cost. From
<https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/>:

> "The src layout **requires installation of the project to be able to run its code**, and the
> flat layout does not."

Its three stated benefits are all about *distributing an installable package*: preventing
accidental use of the in-development copy, packaging-tooling misconfiguration, and
editable-vs-regular install parity. This repo is never `pip install`ed and never published, so it
has none of those failure modes and would pay only the cost — a `pip install -e .` step in every
Colab session. **[C]**

pytest does say "especially if you use the default import mode `prepend`, it is **strongly**
suggested to use a `src` layout"
(<https://docs.pytest.org/en/stable/explanation/goodpractices.html>) — but its stated reason is
the installed-vs-local confusion, which again only applies to installed packages. The same page
gives the flat-layout escape hatch: "If you do not use an editable install and do not use the src
layout... you can rely on the fact that Python by default puts the current directory in
`sys.path` to import your package and run `python -m pytest`."

### One real, cheap fragility

That quote has a sting. With `prepend` import mode, pytest inserts the *test file's* basedir on
`sys.path` — here `tests/`, not the repo root — and it is `python -m pytest` (not the bare
`pytest` console script) that adds the current directory. The README documents
`pytest tests -q`. It evidently works in the project's environment (`.pytest_cache` shows a
clean 12-test run), but it is environment-dependent. *I could not reproduce either way: this
machine has neither pytest nor TensorFlow installed.*

Cheap, honest fix — add a `pyproject.toml` with three lines:

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
```

That makes `from model import fft_layer` work under any invocation, and costs nothing else. It
does **not** make the project a package.

**Verdict: keep `model.py`. Add the three-line pytest config. Do not create `src/`.**

---

## 4. Config management — YAML + dataclass vs Hydra / OmegaConf / gin

**Verdict: what this repo does is correct. Adopting Hydra would be a net loss.**

Hydra's own documentation states its key features
(<https://hydra.cc/docs/intro/>):

> - "Hierarchical configuration composable from multiple sources"
> - "Configuration can be specified or overridden from the command line"
> - "Dynamic command line tab completion"
> - "Run your application locally or launch it to run remotely"
> - "Run multiple jobs with different arguments with a single command"

Multirun's documented purpose (<https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/>)
is sweeping: `python my_app.py -m db=mysql,postgresql schema=warehouse,support,school` launches
six jobs. Hydra also auto-creates a timestamped output dir per run and writes `config.yaml`,
`hydra.yaml` and `overrides.yaml` into a `.hydra/` subdirectory
(<https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory/>). Note the `chdir`
gotcha is gone: "As of Hydra v1.2, `hydra.job.chdir` defaults to `False`."

Now check each against this repo:

| Hydra solves | Does this repo have that problem? |
|---|---|
| Hierarchical composition from multiple sources | No. One flat 17-field config per experiment. There is no `model/` × `data/` × `optimizer/` cross-product to compose. |
| Command-line override | Marginally. Two configs differ in exactly two fields (`mask_mode`, `limit_per_class`). Copying a 17-line YAML is not a burden. |
| Multirun / sweeps | **No.** There are two experiments, not a grid. No hyperparameter search is being run. Each run is a long Colab training job that needs babysitting anyway. |
| Timestamped output dir + config snapshot | **Already solved.** `cfg.run_dir` per named experiment, and `train()` writes `asdict(cfg)` into `metrics.json`. |
| Remote launching | No. |

**OmegaConf** (<https://omegaconf.readthedocs.io/en/latest/>) is "a YAML based hierarchical
configuration system, with support for merging configurations from multiple sources" plus
variable interpolation and dataclass-backed structured configs. The dataclass type-safety part
is the only genuinely attractive feature — and `Config` in `model.py` already provides it,
including a real validity check in `__post_init__` that raises on a bad `mask_mode`.

**gin-config** (<https://github.com/google/gin-config>) uses `@gin.configurable` decorators so
config values are injected into function defaults, which it says "removes the need to define and
maintain configuration objects... or write boilerplate parameter plumbing". The cost is that
configuration becomes *implicit* — you can no longer read a function signature and know what it
will receive. For a repo whose stated virtue is "`configs/*.yaml` is the only thing you change
per experiment", that is a direct regression in legibility. **[O]**

**Where the payoff line actually is.** No source states a project size threshold — that would be
**[U]**. But Hydra's own feature list is a usable test: adopt it when you are running
*combinatorial sweeps* or *composing configs from independently-varying groups*. This repo does
neither. Its configs differ by 2 fields across 2 files. That is the regime where the tool costs
more than it returns.

**One thing worth adding, and it is not a framework.** `evaluate.py` records the device;
`train.py` does not. Add to the `metrics` dict in `train()`:

```python
metrics["env"] = {
    "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True).stdout.strip(),
    "tf_version": tf.__version__,
    "device": _device_name(),          # reuse evaluate.py's helper
    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    "train_seconds": round(time.time() - t_start, 1),
}
```

That is ~10 lines and it converts every run folder into a self-contained provenance record. The
NeurIPS checklist requires exactly this information under Q8 (see §6). It is the single
highest-ratio config-adjacent change available.

---

## 5. Experiment tracking — what is the realistic minimum?

### The failure mode of a hand-maintained table is already visible in this repo

The README's results table currently says:

| experiment | what changed | val AUC |
|---|---|---|
| test13_face | `face_only`, 25k/class | **0.9995** |
| test14_background | `background_only`, 20k/class | _rerun pending_ |

with the note "test13's number is a record of the original notebook run (best of 48 epochs) —
retrain to reproduce it."

That is the failure mode, in three parts, and the README is honest enough to admit all three:

1. **The number is not traceable to any run folder.** There is no `experiments/test13_face/`
   artefact behind it. It came from a notebook that no longer exists in the pipeline. If a panel
   asks "show me the run that produced 0.9995", the answer today is "we can't".
2. **The table drifts silently.** Half the table says "rerun pending". A hand-maintained table
   has no mechanism to notice that it has diverged from the code.
3. **The reported statistic is the wrong one** — see §6.1 below.

This is not a hypothetical risk. It has already happened. And it is exactly the scenario that
kills a thesis defence: not a wrong number, but an *unreproducible* one.

### What the evidence says about tools

The only controlled evidence I found: **Idowu et al., *Empirical Software Engineering* 29:74
(2024), "Machine learning experiment management tools: a mixed-methods empirical study"**
(<https://research.chalmers.se/en/publication/544636>,
<https://dl.acm.org/doi/10.1007/s10664-024-10444-w>). A survey of 81 practitioners plus a
**controlled experiment with 15 student developers**. Findings: "70% of our survey respondents
perform ML experiments using specialized tools, while out of those who do not use such tools, 52%
are unaware of experiment management tools or of their benefits", and in the controlled
experiment "Using ML experiment management tools reduced error rates and increased completion
rates." **[E]** — this is the strongest evidence in this whole document for adopting any tool,
and it is still one small student experiment, not a definitive result.

### The realistic minimum for this thesis

Not MLflow. Not W&B. Not DVC. The minimum is: **every number in the README must be regenerable
from a run folder by a command written in the README.**

That is Papers with Code item 5, verbatim: *"README file including table of results accompanied
by precise commands to run/produce those results"*
(<https://github.com/paperswithcode/releasing-research-code>). It is also Pineau's checklist item
"README file includes table of results accompanied by precise command to run to produce those
results" (<https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf>). **[C]**

Concretely, ~20 lines:

```python
# tools/make_table.py — reads experiments/*/eval.json, prints the README table
```

Then the README table is generated output, not prose. Drift becomes impossible. Effort: under an
hour. This is item 11 in the ranking and it is the whole of what "experiment tracking" needs to
mean here.

### The tools, honestly

- **TensorBoard** (<https://www.tensorflow.org/tensorboard/get_started>) — one line:
  `tf.keras.callbacks.TensorBoard(log_dir=...)`. Genuinely free, works in Colab via
  `%tensorboard`. But `CSVLogger` already writes `history.csv`, and you can plot that with the
  matplotlib you already import. **Add it if you want live curves during training; it is not
  needed to defend anything.** Nice-to-have.
- **MLflow** (<https://mlflow.org/docs/latest/ml/tracking/>) — "By default, without any particular
  server/database configuration, MLflow Tracking logs data to the local `mlruns` directory." So
  it *can* run with no server. But `mlruns/` lives on the Colab VM, and this repo's whole storage
  strategy is "outputs go to Drive so they survive a disconnect". You would be adding a second,
  weaker provenance store next to the `metrics.json` you already have. **Don't.**
- **Weights & Biases** — adds an account, network dependency, and a hosted copy of your data.
  For two experiments, no.
- **DVC** (<https://doc.dvc.org/use-cases/versioning-data-and-models>) — `dvc add` produces a
  `.dvc` file holding an MD5 hash, committed to git; a remote is optional. The *idea* is sound
  and relevant, because the dataset lives in Drive and is the least version-controlled thing in
  this project. But installing DVC to get a hash is disproportionate. **Cheaper equivalent:**
  have `audit_dataset.py` print the per-class file count and a hash of the sorted filename list
  into the run folder. Same provenance guarantee, zero new dependencies, ~10 lines.

**Verdict: keep the JSON-per-run approach. Generate the README table from it. Skip every tool.**

---

## 6. Reproducibility — what the checklists actually require

### 6.1 The headline number is the wrong statistic (fix this first, it is free)

`train()` uses `ModelCheckpoint(monitor='val_auc', save_best_only=True)` *and*
`EarlyStopping(monitor='val_auc', restore_best_weights=True)`. So the validation set is used to
**select** the model. The README then reports **val AUC** — the maximum over 48 epochs on that
same selection set — as the headline 0.9995.

That estimate is optimistically biased by construction. Cawley & Talbot, *On Over-fitting in
Model Selection and Subsequent Selection Bias in Performance Evaluation*, JMLR 11 (2010)
(<https://www.jmlr.org/papers/v11/cawley10a.html>), state it directly: "common performance
evaluation practices are susceptible to a form of selection bias as a result of this form of
over-fitting and hence are unreliable", and show the resulting effects are "of comparable
magnitude to differences in performance between learning algorithms". **[E]**

The repo already does the right thing in code — `load_paths` makes a genuine 70/15/15 split and
`evaluate.py` reports test AUC into `eval.json`. **The fix is purely editorial: make the README
table report test AUC, with val AUC in a secondary column labelled as the selection metric.**
Zero effort, removes the easiest possible attack on the accuracy claim. Ranked #1.

### 6.2 Seeds: is one run acceptable?

**By the letter of every checklist: yes, provided you say it is one run. By community norm and
by the empirical literature: no, not for a headline number.**

- **Pineau's ML Reproducibility Checklist v2.0**
  (<https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf>) requires "**The exact number
  of training and evaluation runs**" and, separately, "A description of results with central
  tendency (e.g. mean) & variation (e.g. error bars)". Notably, **the word "seed" does not appear
  in the checklist at all.** The requirement is disclosure, not replication.
- **The NeurIPS Paper Checklist** (<https://neurips.cc/public/guides/PaperChecklist>) Q7 asks:
  *"Does the paper report error bars suitably and correctly defined or other appropriate
  information about the statistical significance of the experiments?"* — and its guidelines
  require that you state **what factor the error bar captures** ("train/test split,
  initialization, random drawing of some parameter"), **how it was computed**, the assumptions,
  and **whether it is SD or SEM**. It nowhere mandates N seeds.
- **ICLR** treats the Reproducibility Statement as "strongly encouraged", not required
  (<https://iclr.cc/Conferences/2025/AuthorGuide>).
- Pineau et al., JMLR 22(164) 2021 (<https://jmlr.org/papers/v22/20-303.html>) is refreshingly
  candid about the whole apparatus: *"At this stage, we do not have concluding evidence that
  these processes indeed have an impact on the quality of the work or of the papers that are
  submitted and published."* **[E]** It also notes the tension: 87% of authors saw value in
  defining metrics clearly, yet 36% marked error bars "not applicable".

**The empirical case for more than one run:**

- **Picard, arXiv:2109.08203** (<https://arxiv.org/abs/2109.08203>). CIFAR-10 ResNet9 over 10⁴
  seeds: accuracy 90.02 ± 0.23, min 89.01, max 90.83. ImageNet ResNet50 over 50 seeds:
  75.48 ± 0.10. Conclusion: *"a 0.5% difference in accuracy could be entirely explained by just a
  seed difference, without having chosen a particularly bad or good seed."* **[E]**
- **Bouthillier et al., MLSys 2021** (<https://arxiv.org/abs/2103.03098>) is the most directly
  actionable. Randomising each variance source 200× across 5 tasks, they find **"Bootstrapping
  data stands out as the most important source of variance. In contrast, model initialization
  generally is less than 50% of the variance of bootstrap"** — i.e. *the data split matters more
  than the weight-init seed*. On detection quality, "The single-point comparison has both a high
  rate of false positives in the left region (≈10%) and a high rate of false negative on the
  right (≈75%)". Their recommendations: "Randomize as many sources of variations as possible" and
  "Use multiple data splits". **[E]**
- Henderson et al., AAAI 2018 (<https://arxiv.org/abs/1709.06560>) show seed-only variation can
  produce statistically distinct learning curves (t = −9.09, p = 0.0016) — but that is deep RL,
  where the effect is much larger than in supervised image classification. Cite it for the
  principle, not the magnitude. **[E]**

**The lucky part, specific to this repo.** In `model.py`, `cfg.seed` controls *both* the
stratified split (`train_test_split(..., random_state=cfg.seed)`, twice) *and* the weight
initialisation / augmentation (`tf.keras.utils.set_random_seed(cfg.seed)`). So **changing that
one number varies the data split and the initialisation together** — which is precisely the
multi-source randomisation Bouthillier recommends, and it costs nothing to implement.

**Recommendation:** run `seed: 42`, `43`, `44` (three configs, or one `--seed` CLI override).
Report the median test AUC and the min–max range, and state in the thesis that the interval
captures *data split + initialisation + augmentation order*, three runs, range not SD — which is
exactly what NeurIPS Q7's guidelines ask you to state. Three Colab runs. Ranked #7.

If the budget truly allows only one run, the checklists still permit it — but you must write
"single run, seed 42" explicitly, and you must not claim any improvement smaller than a seed's
worth of noise.

### 6.3 Determinism flags — and why you should *not* enable op determinism

`train()` already calls `tf.keras.utils.set_random_seed(cfg.seed)`. TF's docs
(<https://www.tensorflow.org/api_docs/python/tf/keras/utils/set_random_seed>) say this "Sets all
random seeds (Python, NumPy, and backend framework)" and that it "can be used to make almost any
Keras program fully deterministic" — but with the caveat that limits apply "when certain
non-deterministic cuDNN ops are involved". It does **not** give bitwise-identical GPU results.

`tf.config.experimental.enable_op_determinism`
(<https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism>)
guarantees "if an op is run multiple times with the same inputs on the same hardware, it will
have the exact same outputs each time". The documented cost, verbatim:

> "Enabling determinism will likely make your model or your `tf.data` data processing slower. For
> example, `Dataset.map` can become **several orders of magnitude slower** when the map function
> has random ops or other stateful ops."

This repo's `augment_and_mask` calls `tf.random.uniform` three times inside a
`map(..., num_parallel_calls=AUTOTUNE)`. Per the TF docs, with op determinism on, "the function
will run serially instead of in parallel... the `num_parallel_calls` argument to map and
interleave is effectively ignored", and `prefetch` "will be disabled if any function run as part
of the input pipeline has certain stateful ops". Your carefully-designed cache-then-augment
pipeline would be destroyed.

Also note TF's determinism requirements include "Use the same hardware configuration in every
run" and "determinism is not guaranteed across different versions of TensorFlow" — neither of
which Colab can offer (GPU types "vary over time").

**Verdict: do NOT call `enable_op_determinism()`.** Nobody requires bitwise determinism; the
checklists require *disclosure*. Say in the thesis: "seeds are set via
`tf.keras.utils.set_random_seed`; op-level GPU determinism is not enabled, so runs are not
bitwise reproducible — we report the spread across N seeds instead." That is a stronger and more
honest position than a fake determinism guarantee. **[E]**

*Minor, optional:* TF's random-numbers guide
(<https://www.tensorflow.org/guide/random_numbers>) warns that "The old RNGs from TF 1.x such as
`tf.random.uniform` and `tf.random.normal` are not yet deprecated but **strongly discouraged**".
Migrating `augment_and_mask` to `tf.random.Generator` / stateless ops would be more correct. It
is not worth doing before the deadline.

### 6.4 Environment pinning and data versioning

The checklists ask for much less than tooling vendors imply.

- Pineau: "Specification of dependencies."
- NeurIPS Q5 guideline: "The instructions should contain the **exact command and environment**
  needed to run to reproduce the results." Q12: "Remember to state which version of the asset
  you're using." That is the entirety of the requirement — no lockfiles, no containers, no
  dataset versioning tooling appears anywhere in the checklist.
- Papers with Code item 1: `requirements.txt` / `environment.yml` / `setup.py`. Docker is
  explicitly optional: "If you wish to provide whole reproducible environments, you **might want
  to consider** using Docker."

This repo's `requirements.txt` already pins `numpy==2.0.2`, `scikit-learn==1.6.1`,
`PyYAML==6.0.2`, `matplotlib==3.10.0`, `pytest==8.3.4` and constrains
`tensorflow>=2.19,<3 # verified on 2.20`. **That satisfies the requirement.** The one gap is
that Colab ships its own TF, so the *actual* version at runtime may differ from the pin — which
is exactly why §4's `metrics["env"]["tf_version"]` matters more than tightening the pin.

Data versioning: the dataset lives in Drive and is the least-controlled artefact in the project.
The checklist-satisfying minimum is Pineau's "The relevant statistics, such as number of
examples" + "The details of train / validation / test splits" + "an explanation of any data that
were excluded, and all pre-processing step". `build_datasets` already prints the per-split class
counts; write them into `metrics.json` and you are done. **[C]**

### 6.5 Compute disclosure — a hard requirement you can satisfy in 20 minutes

NeurIPS Q8, verbatim: *"For each experiment, does the paper provide sufficient information on the
computer resources (type of compute workers, memory, time of execution) needed to reproduce the
experiments?"* Guidelines add: indicate CPU/GPU type, memory and storage; "provide the amount of
compute required for each of the individual experimental runs as well as estimate the total
compute"; and disclose "whether the full research project required more compute than the
experiments reported in the paper (e.g., preliminary or failed experiments)".

`evaluate.py` already records `device`. Add TF version, CUDA/cuDNN if reachable, precision
(FP32 vs mixed), whether XLA/`jit_compile` is on, host RAM, and per-run wall-clock. Ranked #3
and #9.

---

## 7. Benchmarking the efficiency claim

This is the section the thesis lives or dies on, so it gets the most specific treatment.

### 7.1 The established rules: MLPerf Inference

MLPerf's rules (<https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc>;
paper <https://arxiv.org/abs/1911.02549>) are the closest thing to a standard. **[C]** for
applicability, **[E]** for the statistics behind the query counts.

Scenarios and metrics:

| Scenario | Duration | Samples/query | Tail latency | Metric |
|---|---|---|---|---|
| SingleStream | 600 s | 1 | **90%** | 90th-%ile latency |
| MultiStream | 600 s | 8 | 99% | 99th-%ile latency |
| Server | 600 s | 1 | 99% | max Poisson throughput |
| Offline | 600 s, ≥24,576 samples | all | N/A | measured throughput |

Minimum query counts, from their own confidence-interval table:

| tail percentile | confidence | margin of error | rounded inferences |
|---|---|---|---|
| 90% | 99% | 0.50% | **24,576** |
| 99% | 99% | 0.05% | 270,336 |

And the sentence that matters most here, verbatim: an early-stopping criterion allows short runs
"**with the penalty that the effective computed percentile will be slightly higher. This penalty
counteracts the increased variance inherent to runs with few queries, where there is a higher
probability that a particular run will, by chance, report a lower latency than the system should
reliably support.**"

**Applied: `runs=100` is too few to support a p95.** A p95 over 100 samples is the 5th-largest
observation — driven by about five numbers — and MLPerf's own reasoning says few queries bias
latency *optimistically*, which is the direction that flatters your claim and the direction a
panel will suspect. Fix: raise `runs` to ≥1000 (at a few ms per inference that is seconds of
GPU time), or keep n=100 and report **median + interquartile range only**, explicitly
disclaiming the tail. One-line change; ranked #4.

MLPerf has **no warm-up rule** (the string "warm" does not appear in the rules document) — it
dissolves the problem with a 600 s minimum duration instead. So cite the PyTorch recipe's
2775.5 µs → 22.4 µs measurement for warm-up, not MLPerf.

MLPerf also mandates a machine-readable system description with meaningful values for
`accelerator_model_name` (e.g. "Nvidia Tesla V100"), `accelerator_memory_capacity`, `framework`
(their example is exact: "TensorFlow 1.14 commit hash = faf9db5..."), `other_software_stack`
("cuda 10.2.0.163, cudnn 7.6.0.64..."), and `operating_system`. That is the template for what to
put in `eval.json`.

### 7.2 FLOPs/params as proxies — what the papers say

- **ShuffleNet V2** (ECCV 2018, <https://arxiv.org/abs/1807.11164>): *"FLOPs is an **indirect**
  metric. It is an approximation of, but usually not equivalent to the **direct** metric that we
  really care about, such as speed or latency."* Reasons: memory access cost (MAC) and degree of
  parallelism are invisible to FLOPs, and "operations with the same FLOPs could have different
  running time, depending on the platform." Their four guidelines: G1 equal channel width
  minimises MAC; G2 excessive group convolution increases MAC; G3 "fragmentation reduces the
  speed significantly"; G4 "element-wise operations occupy considerable amount of time... They
  have small FLOPs but relatively heavy MAC." Benchmarked on two *named* platforms (GTX 1080Ti +
  CUDNN 7.0, Snapdragon 810). **[E]**
- **MnasNet** (<https://arxiv.org/abs/1807.11626>): *"FLOPS is often an inaccurate proxy: for
  example, MobileNet and NASNet have similar FLOPS (575M vs. 564M), but their latencies are
  significantly different (113ms vs. 183ms)."* **[E]**
- **EfficientNetV2** (<https://arxiv.org/abs/2104.00298>): *"Depthwise convolutions have fewer
  parameters and FLOPs than regular convolutions, but they often cannot fully utilize modern
  accelerators."* **[E]** — note this one cuts **in your favour**: your model uses plain dense
  convolutions, which utilise a GPU well, whereas EfficientNet-B0 is built from depthwise
  separable blocks. That is a plausible mechanism by which you beat B0 on measured latency
  despite losing on MACs. **Test it; if it holds, it is a genuine finding worth a paragraph.**
- **RepVGG** (<https://arxiv.org/abs/2101.03697>): FLOPs "does not precisely reflect the actual
  speed"; models with lower FLOPs "may not run faster". Reports examples/second on a named 1080Ti.
- **HW-NAS-Bench** (<https://arxiv.org/abs/2103.10584>) supplies the hard numbers: Kendall τ
  between FLOPs and measured latency of **0.3571** (Edge GPU), **0.1847** (Edge TPU), and
  **0.0149** on one search space — essentially zero. **[E]**

### 7.3 Is reporting FLOPs alongside params expected? Yes.

Every efficiency paper checked reports **three columns plus named hardware**:

- MobileNetV2 (<https://ar5iv.labs.arxiv.org/html/1801.04381>): Params / MAdds / CPU ms, with
  "running time in milliseconds (ms) for a single large core of the Google Pixel 1 phone".
- MobileNetV3: MAdds / Params / latency on three Pixel generations, "single large core with a
  batch size of one".
- EfficientNetV2: Params / FLOPs / inference ms / training h, measured on "V100 FP16, batch size
  16", same codebase, **same machine**.
- Keras's own Applications table (<https://keras.io/api/applications/>) reports Size (MB),
  accuracy, Parameters, and "Time (ms) per inference step" for CPU and GPU, disclosing hardware
  (Tesla A100), batch size 32, and method: "Time per inference step is the average of 30 batches
  and 10 repetitions."

**Verdict: reporting params without FLOPs/MACs is below the field norm [C]**, and in this
repo's case it actively hides the finding in §0.

Two implementation notes:

1. **State your convention.** MobileNet/MnasNet report **MAdds** (multiply-adds); EfficientNet's
   "FLOPs" column is also multiply-adds. If you write "2.4 GFLOP" while a baseline table means
   MACs, you have a silent 2× error in your headline number. Pick one, label it, use it
   everywhere.
2. **Automated FLOPs counting will undercount your FFT branch.** TF's profiler documentation
   (<https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/g3doc/profile_model_architecture.md>)
   states an op is only counted if "It must have `RegisterStatistics('flops')` defined in
   TensorFlow". `tf.signal.fft2d` is unlikely to have one. *(I did not verify this for fft2d
   specifically — check it, don't assume.)* The FFT itself is cheap (~5·N²log₂N ≈ 2.6 MFLOP per
   256² image, negligible against 2.4 GFLOP), so report the analytic conv count as the primary
   number and note the FFT separately.

### 7.4 The baseline runner — the specification

The README already says no baseline runner exists and that it belongs beside `build_model`.
Correct. Here is what it must control, each item being an attack vector if uncontrolled:

1. **Same GPU, same session.** Non-negotiable; Colab's GPU type varies by session
   (<https://research.google.com/colaboratory/faq.html>). Record `_device_name()` into every
   result.
2. **Same input resolution.** Xception is natively 299², EfficientNet-B0 and ResNet-50 are 224².
   Running your model at 256² against baselines at their native sizes makes part of the latency
   gap a resolution artifact. Either fix all models at 256² (Keras Applications accept
   `input_shape`) or report resolution per row and discuss it. **This is the single easiest
   methodological objection to raise against the table, so pre-empt it.**
3. **Same call path.** Every model wrapped in the same `tf.function(lambda x: m(x,
   training=False))`, same warm-up count, same `runs`, same sync mechanism. Never mix
   `model.predict()` with a raw call.
4. **Same precision and compile flags.** All FP32, or all mixed — and state whether `jit_compile`
   / XLA is on.
5. **Same test split**, from `build_datasets(cfg)`.
6. **Randomise or interleave model order** rather than running each model's iterations as one
   fixed block, so thermal drift and co-tenancy on a shared VM do not systematically favour
   whichever model runs first.
7. **Repeat the whole measurement ≥3 times, ideally across sessions,** and report between-run
   spread. MLPerf can require a single run because each run is 600 s and ≥24,576 queries; you
   cannot.
8. **Report both latency (bs=1) and throughput (large batch).** These are different claims —
   MLPerf SingleStream vs Offline. The repo already reports throughput at batch 144; keep it,
   label it, and disclose the batch size (it already does).
9. **Sweep bs = 1, 8, 32.** At bs=1 a 1M-param model on a modern GPU is plausibly
   launch-latency-bound rather than compute-bound, meaning your advantage may shrink or invert as
   batch grows. Finding the crossover point turns the deepest likely attack into a contribution.

### 7.5 What reviewers attack, in likely order

1. **"Fewer parameters — but did you measure MACs, or are you inferring speed from size?"**
   (Green AI + Efficiency Misnomer.) See §0. This is the first and most dangerous question.
2. **"n=100 — what is your confidence interval on that p95?"** (MLPerf's 24,576 table.)
3. **"Colab. Which GPU, which session, and did the baselines run on the same one?"**
4. **"Same input resolution? Same precision? Same call path?"**
5. **"Is your model memory-bound rather than compute-bound at bs=1?"**
6. **"Is any baseline number quoted from another paper?"** If yes, it is inadmissible.
7. **"Your val AUC is 0.9995 on a two-folder dataset. What did the model actually learn?"** —
   see §8.

Dodge et al., *Show Your Work* (EMNLP 2019, <https://arxiv.org/abs/1909.03004>) is worth knowing
for calibration: reviewing 50 EMNLP 2018 papers, "none of the papers reported all of the items we
suggest". The bar is low. Clearing it visibly is cheap and it reads as rigour.

**One honest tension to acknowledge in the thesis rather than paper over.** *Green AI* argues the
opposite of the latency-first position: it rejects elapsed real time as a reportable metric
because it is "highly influenced by factors such as the underlying hardware, other jobs running
on the same machine, and the number of cores used", and recommends hardware-agnostic FPO instead.
The Efficiency Misnomer resolves this by saying report *multiple* indicators. **Sources disagree
here, and the defensible position is to say so:** FLOPs/MACs is the hardware-agnostic
comparability number, measured latency is the deployment-truth number, neither substitutes for
the other, report both. A thesis that names this disagreement explicitly looks more competent
than one that picks a side silently.

---

## 8. The 0.9995 problem — the shortcut risk

Not one of the six questions, but it is the largest threat to the thesis and it interacts with
everything above, so it gets a short section.

Reals and fakes live in two separate folders, and val AUC is 0.9995. Geirhos et al., *Shortcut
Learning in Deep Neural Networks* (Nature Machine Intelligence 2020,
<https://arxiv.org/abs/2004.07780>) defines shortcuts as "decision rules that perform well on
standard benchmarks but fail to transfer to more challenging testing conditions", and its central
methodological point is that **i.i.d. test performance is insufficient** — shortcuts only reveal
themselves under distribution shift. **[E]**

In GAN-face detection specifically, the field norm for a credible claim is **train on one
generator, test on unseen ones**: Wang et al., *CNN-generated images are surprisingly easy to
spot... for now* (CVPR 2020, <https://arxiv.org/abs/1912.11035>) train on ProGAN only and
evaluate across 11 generators. Gragnaniello et al., *Are GAN generated images easy to detect? A
critical analysis of the state-of-the-art* (ICME 2021, <https://arxiv.org/abs/2104.02617>)
devotes itself to "realistic and challenging scenarios, like media uploaded on social networks or
generated by new and unseen architectures" — i.e. precisely the gap between 0.9995 on your own
split and any claim about detection. **[C]**

The repo has already built three of the four right instruments:

1. `audit_dataset.py` — metadata shortcuts. **Built.**
2. `background_only` control — "if it still works, it wasn't reading the face". **Built, run
   pending.** Finish this; ranked #6.
3. Grad-CAM under `face_only`. **Built.**
4. **Held-out generator.** Not built. This is the one that actually answers the question, and the
   README already says so: "a held-out generator is the real test." Ranked #10 — highest
   scientific value, real data cost.

A useful framing for the write-up: if the held-out-generator number is much lower than 0.9995,
that is **not** a failed thesis. It is a correct and publishable finding that reproduces the
field's known generalisation gap, and it makes the efficiency claim *more* credible by showing
you are not simply reporting a saturated benchmark.

---

## 9. Anti-patterns and what is genuinely not worth doing

### Do not do these

| Thing | Why not, at this scale |
|---|---|
| **Hydra / OmegaConf / gin** | §4. No sweeps, no config composition, no remote launching. `Config` already gives dataclass type-safety and validation. |
| **MLflow / W&B / Neptune** | §5. Adds a second, weaker provenance store beside `metrics.json`, plus an account and a network dependency, for two experiments. |
| **DVC** | §5. Real idea, disproportionate machinery. A filename-list hash printed into the run folder gives the same guarantee in 10 lines. |
| **`src/` package + `setup.py` + `pip install -e .`** | §3. PyPA: "requires installation of the project to be able to run its code." All its documented benefits are packaging-distribution concerns you do not have. Pure cost on Colab. |
| **Docker** | Colab does not run your container. Papers with Code lists Docker as "might want to consider". Nobody requires it. |
| **`enable_op_determinism()`** | §6.3. TF documents `Dataset.map` becoming "several orders of magnitude slower" with random ops, and it would disable your `prefetch`. Nobody requires bitwise determinism. |
| **Chasing test coverage** | Wilson et al. explicitly leave unit tests off their core list for solo/small exploratory work, noting "the lack of a test directory in Noble's rules". Two tests that caught two real bugs is the right amount. |
| **Splitting `model.py` into `data.py`/`layers.py`/`train.py`** | 337 lines with section banners is navigable. Splitting creates import ceremony and merge conflicts between three people, and buys nothing measurable — §3 found *no* study measuring layout against any outcome. |
| **Type hints / mypy / pre-commit / linting config** | Zero evidence of benefit at this scale and deadline. |
| **A model registry, feature store, serving layer, k8s** | Not applicable to a thesis. |
| **Refactoring "for the next student"** | You do not know who they are or what they will change. The README's "Group split" section already does this job in prose, for free. |

### Borderline, your call

- **GitHub Actions running `pytest tests`** — 10 minutes to set up, ~3s per run, no GPU needed.
  Wilson et al. say CI "only starts to pay off as projects grow larger" **[C]**, and they are
  probably right here. But it is genuinely cheap and it means the `if training:` mask bug can
  never come back silently across three contributors. Optional; slight lean toward yes.
- **TensorBoard callback** — one line, free, useful during long Colab runs. Not needed to defend
  anything.
- **`img_size` / `native_size` in the cache key** — the README already documents that changing
  `img_size` "needs a manual clear". A documented footgun is still a footgun, and the failure
  mode is *silently training on wrong-size cached data*, which produces a plausible-looking wrong
  number. One line in `Config.cache_path`. Cheap insurance; lean yes.

### The general principle

Wilson et al.'s warning, worth re-reading before adopting anything in this document:

> "The problem comes when those experienced developers give advice to people who haven't already
> mastered the tools and don't realize (yet) that they will save time if and when their project
> grows. In that situation, advocating unit testing with coverage checking and continuous
> integration is more likely to overwhelm newcomers rather than aid them."

Every tool above is a tool someone would save time with *on a different project*.

---

## 10. The three buckets

### Needed to defend the efficiency claim
1. MACs/FLOPs reported alongside params, for this model **and** every baseline, at the same input
   resolution, with the MAC-vs-FLOP convention stated. (§0, §7.3)
2. The baseline runner, to the specification in §7.4 — same session, same GPU, same resolution,
   same call path, order interleaved, ≥3 repeats.
3. System-under-test recorded into `eval.json`: GPU model, TF version, CUDA/cuDNN, precision,
   XLA, host RAM, batch size, n, warmup. (§7.1, §6.5)
4. `runs` ≥ 1000, or drop p95 and report median + IQR with an explicit disclaimer. (§7.1)
5. Latency at bs = 1, 8, 32. (§7.4)
6. An explicit sentence naming *which* resource is saved — parameters and memory, yes; total
   compute versus EfficientNet-B0, no.

### Needed to defend the accuracy claim
7. Report **test** AUC as headline, val AUC labelled as the selection metric. (§6.1) — free.
8. Finish the `background_only` control run. (§8)
9. Three seeds; report median and range; state what the interval captures. (§6.2)
10. Held-out generator evaluation, if time exists. (§8)

### Nice to have
11. README results table generated from `experiments/*/eval.json`. (§5)
12. `git_sha` / `tf_version` / `device` / wall-clock / timestamp into `metrics.json`. (§4)
13. Per-split class counts and a dataset filename-list hash into `metrics.json`. (§6.4)
14. `pythonpath = ["."]` in a three-line `pyproject.toml`. (§3)
15. `img_size` in the cache key. (§9)
16. TensorBoard callback; optional CI on the two guard tests.

### Do not bother
Hydra, OmegaConf, gin, MLflow, W&B, DVC, `src/` package layout, `setup.py`, Docker,
`enable_op_determinism()`, coverage targets, splitting `model.py`, type-checking infrastructure,
speculative refactoring.

---

## 11. Where the evidence is thin — stated rather than papered over

- **Repo layout has no empirical support in either direction.** No study I could find treats
  directory structure as an independent variable against reproducibility, defects, or onboarding.
  Wilson, Noble, cookiecutter-data-science and the Turing Way are all consensus or self-declared
  opinion. The *only* measured findings in that space concern relative paths, pinned
  dependencies, clean-environment testing, and getting code out of notebooks (Trisovic 2022;
  Pimentel 2019) — all of which this repo already satisfies.
- **No source names a project size at which Hydra starts paying off.** Any specific threshold is
  **[U]**. The usable test is behavioural: do you run combinatorial sweeps, or compose configs
  from independently-varying groups? Here, no.
- **The reproducibility checklists themselves are unvalidated.** Pineau et al.'s own JMLR report:
  "we do not have concluding evidence that these processes indeed have an impact on the quality
  of the work". Follow them because examiners expect them, not because they are proven.
- **Green AI and the Efficiency Misnomer disagree about whether measured latency should be
  reported at all.** Report both indicators and name the disagreement.
- **The seed-variance literature's magnitudes do not transfer cleanly.** Henderson's effect sizes
  are deep-RL-sized. Picard's ~0.5-point CIFAR-10 figure is the right order of magnitude for
  supervised vision. Neither was measured on a near-saturated binary task at AUC 0.9995, where
  the variance structure is different — near the ceiling, seed variance will look small, which
  is *not* evidence that the result is robust.
- **Official TensorFlow benchmarking guidance on warm-up and synchronisation could not be
  found.** The TF GPU performance guide has none. The CUDA-asynchrony argument is sound but is
  documented by PyTorch; cite it as a CUDA property.
- **Two things in this document I could not verify on this machine** and flag as such: (a)
  whether bare `pytest tests -q` imports `model` successfully in the project's environment —
  neither pytest nor TensorFlow is installed here, and `.pytest_cache` suggests it worked
  somewhere; (b) whether `tf.signal.fft2d` registers FLOPs statistics with the TF profiler.
  Check both; do not take them on faith.

---

## Sources

**Repo organisation and research software practice**
- Wilson et al., *Good Enough Practices in Scientific Computing*, PLOS Comp Bio 2017 — <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005510>
- Wilson et al., *Best Practices for Scientific Computing*, PLOS Biology 2014 — <https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1001745>
- Noble, *A Quick Guide to Organizing Computational Biology Projects*, PLOS Comp Bio 2009 — <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000424>
- cookiecutter-data-science, Opinions — <https://cookiecutter-data-science.drivendata.org/opinions/>
- The Turing Way, Research Compendia — <https://book.the-turing-way.org/reproducible-research/compendia/>
- Trisovic et al., *A large-scale study on research code quality and execution*, Scientific Data 2022 — <https://www.nature.com/articles/s41597-022-01143-6>
- Pimentel et al., *A Large-scale Study about Quality and Reproducibility of Jupyter Notebooks*, MSR 2019 — <https://www.ic.uff.br/~leomurta/papers/pimentel2019a.pdf>
- PyPA, src layout vs flat layout — <https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/>
- pytest, Good Integration Practices — <https://docs.pytest.org/en/stable/explanation/goodpractices.html>

**Config and tracking**
- Hydra docs — <https://hydra.cc/docs/intro/>, multirun <https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/>, working directory <https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory/>
- OmegaConf — <https://omegaconf.readthedocs.io/en/latest/>
- gin-config — <https://github.com/google/gin-config>
- MLflow Tracking — <https://mlflow.org/docs/latest/ml/tracking/>
- TensorBoard — <https://www.tensorflow.org/tensorboard/get_started>
- DVC — <https://doc.dvc.org/use-cases/versioning-data-and-models>
- Idowu et al., *ML experiment management tools: a mixed-methods empirical study*, EMSE 2024 — <https://dl.acm.org/doi/10.1007/s10664-024-10444-w>, abstract <https://research.chalmers.se/en/publication/544636>

**Reproducibility**
- Pineau, ML Reproducibility Checklist v2.0 — <https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf>
- Pineau et al., *Improving Reproducibility in ML Research*, JMLR 22(164) 2021 — <https://jmlr.org/papers/v22/20-303.html>
- NeurIPS Paper Checklist — <https://neurips.cc/public/guides/PaperChecklist>; Code Submission Policy — <https://neurips.cc/public/guides/CodeSubmissionPolicy>
- ICLR Author Guide — <https://iclr.cc/Conferences/2025/AuthorGuide>
- Papers with Code, ML Code Completeness Checklist — <https://github.com/paperswithcode/releasing-research-code>
- Cawley & Talbot, *On Over-fitting in Model Selection...*, JMLR 11 (2010) — <https://www.jmlr.org/papers/v11/cawley10a.html>
- Bouthillier et al., *Accounting for Variance in ML Benchmarks*, MLSys 2021 — <https://arxiv.org/abs/2103.03098>
- Picard, *torch.manual_seed(3407) is all you need* — <https://arxiv.org/abs/2109.08203>
- Henderson et al., *Deep Reinforcement Learning that Matters*, AAAI 2018 — <https://arxiv.org/abs/1709.06560>
- TF: `set_random_seed` — <https://www.tensorflow.org/api_docs/python/tf/keras/utils/set_random_seed>; `enable_op_determinism` — <https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism>; random numbers guide — <https://www.tensorflow.org/guide/random_numbers>; Keras reproducibility recipe — <https://keras.io/examples/keras_recipes/reproducibility_recipes/>

**Efficiency benchmarking**
- MLPerf Inference rules — <https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc>; paper — <https://arxiv.org/abs/1911.02549>
- Dehghani et al., *The Efficiency Misnomer*, ICLR 2022 — <https://arxiv.org/abs/2110.12894>
- Schwartz et al., *Green AI*, CACM 2020 — <https://arxiv.org/abs/1907.10597>
- Ma et al., *ShuffleNet V2*, ECCV 2018 — <https://arxiv.org/abs/1807.11164>
- Tan & Le, *EfficientNet*, ICML 2019 — <https://arxiv.org/abs/1905.11946>; *EfficientNetV2* — <https://arxiv.org/abs/2104.00298>
- Tan et al., *MnasNet* — <https://arxiv.org/abs/1807.11626>; *MobileNetV2* — <https://arxiv.org/abs/1801.04381>; *MobileNetV3* — <https://arxiv.org/abs/1905.02244>
- Ding et al., *RepVGG* — <https://arxiv.org/abs/2101.03697>
- Li et al., *HW-NAS-Bench*, ICLR 2021 — <https://arxiv.org/abs/2103.10584>
- Dodge et al., *Show Your Work*, EMNLP 2019 — <https://arxiv.org/abs/1909.03004>
- PyTorch CUDA semantics — <https://docs.pytorch.org/docs/stable/notes/cuda.html>; benchmark recipe — <https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html>
- TF `get_memory_info` — <https://www.tensorflow.org/api_docs/python/tf/config/experimental/get_memory_info>; profiler FLOPs caveats — <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/g3doc/profile_model_architecture.md>
- Keras Applications table — <https://keras.io/api/applications/>
- Colab FAQ (GPU types vary) — <https://research.google.com/colaboratory/faq.html>

**Shortcut learning / GAN detection**
- Geirhos et al., *Shortcut Learning in Deep Neural Networks*, Nature MI 2020 — <https://arxiv.org/abs/2004.07780>
- Wang et al., *CNN-generated images are surprisingly easy to spot... for now*, CVPR 2020 — <https://arxiv.org/abs/1912.11035>
- Gragnaniello et al., *Are GAN generated images easy to detect?*, ICME 2021 — <https://arxiv.org/abs/2104.02617>
