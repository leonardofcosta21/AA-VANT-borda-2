# What changed from the previous repository

The previous code base (`17_06/`) could run parts of the campaign but
could not deliver it: three partial copies of the adaptation loop, one
runner script per experiment family, and no owner of the question "did
every planned run actually happen". This document records what changed
and why, so the decisions can be defended rather than re-litigated.

---

## Defects found and fixed

**The uncertainty threshold never applied.** `BALDOnlySampling` and
`BALDDiversitySampling` computed the quantile threshold
$\tau_k = \mathrm{Quantile}_{1-\alpha}$ and then kept the filtered set
only `if len(filtered) >= budget`. With $\alpha = 0.1$ on a pool of ~350
candidates that yields ~35 survivors against a budget of 50, so the
condition failed on every cycle and the filter was discarded. The
documented two-stage selection — threshold, then diversity — was
effectively single-stage for the entire preliminary campaign. Fixed by
flooring the retained fraction at $b/|\mathcal{C}_k|$
(`apply_uncertainty_prefilter`, with a regression test).

**Latency was measured on random noise.** `Evaluator.measure_latency`
timed repeated inference on one buffer of uniform noise. Noise produces
almost no candidate boxes, so NMS and post-processing — a real share of
YOLO's per-frame cost — were never exercised, and reusing one buffer let
caches flatter the numbers. It also never synchronised CUDA, so on GPU it
timed kernel *queueing*. Fixed to time real test images with
synchronisation, and to report the distribution rather than only the
mean.

**An empty results table with no explanation.** `_aggregate_shift_runs`
indexed `r["summary"]` directly, so one crashed seed raised and aborted
the aggregation of the whole condition. This is why the previous campaign
left per-strategy shift files on disk alongside an empty
`all_shift_aggregated.json`. Failed runs are now journalled with their
traceback, aggregation proceeds from what succeeded, and the shortfall is
recorded in the aggregate (`n_seeds` vs `n_expected`).

**Results depended on the numpy version.** `np.trapz` and `np.trapezoid`
were called from different modules; the former is deprecated in numpy 2.
Centralised in `metrics.trapezoid`.

**Dead code that no longer matched its own helpers.** `src/pipeline.py`
called `IncrementalTrainer(config, work_dir=...)` against a class whose
signature was `(model_name, img_size, device)`, and imported
`dataset_utils` functions that had moved. Removed; `src/al_loop.py` is
now the single implementation.

**Duplicated module.** `uncertainty_extended.py` existed at the
repository root and under `src/`, with divergent contents. The `src/`
copy is kept.

**Silent name mismatches in the reporting layer.** Aggregation appends
`_mean` to every run-level scalar, so `diversity_coverage_mean` becomes
`diversity_coverage_mean_mean`; the table builder checked the un-suffixed
name and emitted an empty diversity table. A baseline recall of exactly
0.0 was rendered as missing data by a truthiness check. Both were caught
by running the campaign on the mock backend, which is why that backend
exists.

---

## What is new

| Area | What was missing | What now exists |
|---|---|---|
| Orchestration | one script per family, no resume, no manifest | `configs/campaign.yaml` + `run_campaign.py`: declarative matrix, resumable, journalled, provenance per run |
| H4 measurement | mean latency only | latency percentiles, peak RAM and VRAM, CPU and GPU utilisation, power and energy, network payload, thermal, fine-tuning wall clock, Jetson `tegrastats` |
| Statistics | means over a single run | bootstrap CIs, exact Wilcoxon on seed-paired values, Holm-Bonferroni, Cliff's $\delta$, shortfall reporting |
| Diversity | implemented but never measured | cluster coverage, selection entropy, redundancy rate, representativeness — per cycle, for every strategy |
| Continual learning | mentioned in the text only | per-domain evaluation matrix, BWT, forgetting measure, retention, stability gap, plasticity |
| Uncertainty methods | MC Dropout, with BSB/PSB/conformal/SWAG unwired | all wired through one loop, plus Deep Ensembles, each with its cost profile |
| AUC | raw integral only (240.65) | raw *and* budget-normalised, side by side in every table |
| Data | failures surfaced deep inside the loader | `audit_datasets.py`: pairing, coordinates, class range, capacity, taxonomy |
| Figures | matplotlib defaults, shrunk into LaTeX floats | authored at final width, 13/14/15 pt, marker and line-style encoding, decluttered direct labels, CVD-validated palette |
| Formalisation | prose description | `docs/ALGORITHM.md`: notation table, equations, Algorithm 1 in pseudocode and LaTeX, complexity table |
| Verification | none | 42 unit tests, plus a mock backend that exercises the whole campaign without a GPU |

---

## Board items and where they are answered

Generated in full in [`EXPERIMENT_MATRIX.md`](EXPERIMENT_MATRIX.md). The
short version:

| Revision-plan item | Answered by |
|---|---|
| Normalise AUC (240.65) | `metrics.auc_normalized`, every table |
| $L_0 < 100$, $B = 32$, find the sweet spot | `E1_main_grid`, `tables/sweet_spot.tex` |
| More runs and robust statistical validation | 5 seeds everywhere + `src/stats.py` |
| Distribution shift experiments (H2, C1) | `E2a`, `E2b`, `E2c` |
| Broaden H4 beyond latency | `E6` + `src/profiling.py` |
| Concrete embedded profiling | `E6` runs unmodified on Jetson with `tegrastats` |
| Address diversity explicitly | `src/diversity_metrics.py`, `tables/diversity.tex` |
| Catastrophic forgetting / continual learning | `E5` + `src/forgetting.py` |
| MC Dropout as *approximate* Bayesian; is it affordable? | `E3` + `docs/ALGORITHM.md` §2.1 |
| Reassess Deep Ensembles | `src/deep_ensemble.py`, in `E3` with costs |
| Conformal prediction, SWAG | wired into `E3` |
| BSB/PSB, portfolio of criteria | wired into `E3` |
| Pseudocode / flowchart of the pipeline | `docs/ALGORITHM.md` §4 |
| More mathematical formalisation | `docs/ALGORITHM.md` §2–3 |
| What makes the pipeline edge-native (C3) | `docs/ALGORITHM.md` §5 + `E6` |
| Reproducibility as an explicit contribution (C6) | this repository, `docs/REPRODUCIBILITY.md` |
| Hyperparameter sensitivity (thesis §5.3.6.1) | `E4` |

---

## Items the code cannot answer

These are text and framing changes for the thesis itself, listed here so
nothing is lost between the revision plan and the repository:

- Position the work as scientific research rather than product
  engineering; state the expected TRL.
- State explicitly that the work does not forecast disasters, plan
  operational responses, or handle last-mile delivery of detections.
- Enumerate the object and event categories the system targets (fire,
  flooding, gas cylinders, scavenging birds, victims).
- Note that crowd simulation is an established field and that collecting
  faithful disaster data is complex, risky and ethically sensitive —
  which justifies the use of public datasets.
- Replace control-theoretic vocabulary (*closed-loop architecture*,
  *operational control signal*, *cyber-physical system*) with the
  conceptual software-architecture framing the work actually supports.
  `docs/ALGORITHM.md` §3 states the limitation in the form it can be
  defended.
- Use *covariate shift* where the text currently says *concept drift*
  (pp. 34–35), consistent with p. 70.
- Add BALD to the abstract's acronym list; fix the `subsubsectadaptation`
  typo on p. 43; remove or justify the "Why?" note on p. 98; discuss
  Figure 8 (blockchain agent layer) in the main text; remove redundancy
  in Section 1.4; soften "all existing Active Learning techniques are
  employed in offline settings".
- Reposition H1 and H3, which are already well established in the Active
  Learning literature, and restate C1, C2, C3 and C5 to match what the
  experiments demonstrate.
- Add the publication plan and revise the schedule.
