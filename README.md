# Continuous Model Adaptation in UAV–Edge Systems via Uncertainty-Guided Active Learning

Experimental code, configurations and reproduction instructions for the
doctoral thesis of the same name (UFC/MDCC).

This repository is contribution **C6**: everything needed to reproduce
every number, table and figure in the thesis, from a single command per
experiment block.

---

## What this is

A closed-loop pipeline in which a UAV-borne detector is adapted during a
mission: it estimates its own epistemic uncertainty with approximate
Bayesian inference (MC Dropout), selects the most informative *and* most
diverse frames for human annotation under a fixed budget, and
incrementally fine-tunes itself when either enough new labels have
accumulated or a distribution shift is detected.

The repository runs the full experimental campaign behind four
hypotheses:

| | Hypothesis | Primary evidence |
|---|---|---|
| **H1** | Uncertainty-guided selection is more annotation-efficient | normalised AUC across the $L_0 \times B$ grid |
| **H2** | Approximate Bayesian uncertainty is more stable under covariate shift | inter-cycle recall variance on real cross-dataset pairs |
| **H3** | Diversity adds over pure uncertainty | cluster coverage and redundancy of the selected batches |
| **H4** | Continuous edge adaptation is operationally feasible | latency distribution, memory, utilisation, power, network payload |

---

## Quick start

```bash
git clone <this-repo> && cd uav-edge-al
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. What data do I actually have?
python scripts/audit_datasets.py

# 2. What would the campaign cost?
python run_campaign.py plan

# 3. Validate the whole pipeline without a GPU (seconds, synthetic data)
python run_campaign.py run --all --mock && python run_campaign.py report

# 4. Run for real, highest-priority block first
python run_campaign.py run --block E1_main_grid

# 5. Build every table and figure from whatever has completed
python run_campaign.py report
```

Or with `make`: `make audit`, `make plan`, `make smoke`, `make e1`, `make report`.

---

## Getting the data

```bash
python scripts/download_datasets.py --all      # SARD, VisDrone, UAVDT, AIDER, FloodNet, LADI
python scripts/prepare_datasets.py             # convert to YOLO + build the unified taxonomy
python scripts/audit_datasets.py               # verify before spending GPU hours
```

`audit_datasets.py` is not optional decoration. It checks pairing,
coordinate normalisation, class-id range, empty-label rate and whether
each dataset is large enough for the $L_0$ values the campaign asks for,
and it exits non-zero if anything is unusable. Most lost campaign time
comes from discovering a half-converted dataset at hour six.

Some sources need manual acquisition (registration or a licence). The
downloader says so per dataset and skips rather than failing.

---

## The campaign

Every experiment is declared in [`configs/campaign.yaml`](configs/campaign.yaml),
which carries, for each block, the examination board items it addresses
and why that experiment is the right answer to them.
[`docs/EXPERIMENT_MATRIX.md`](docs/EXPERIMENT_MATRIX.md) is generated from
that file, so the mapping can never drift from what the code runs.

| Block | What it settles | Priority |
|---|---|---|
| `E1_main_grid` | H1/H3 grid: 4 strategies × $L_0 \in \{25..400\}$ × $B \in \{32,50\}$ × 5 seeds, with diversity diagnostics | 1 |
| `E2a_shift_baseline` | the no-adaptation reference under shift | 1 |
| `E2b_shift_real` | H2 on real cross-dataset pairs (VisDrone→FloodNet and others) | 1 |
| `E2c_shift_intensity` | graded synthetic shift: how degradation scales | 2 |
| `E3_uncertainty_methods` | MC Dropout vs Deep Ensembles, conformal, SWAG, BSB/PSB — with costs | 2 |
| `E4_sensitivity` | $T$, $b$, $K_{\text{clusters}}$, $\tau_{\text{shift}}$ | 3 |
| `E5_long_horizon` | $K = 20$ cycles, catastrophic forgetting and retention | 2 |
| `E6_operational_profiling` | H4: latency tail, memory, CPU/GPU, power, energy, network | 1 |

Blocks are **resumable**. Completed runs are journalled, so re-running a
block picks up where it stopped; a campaign interrupted at hour 30 of 40
costs ten hours to finish, not forty.

```bash
python run_campaign.py status               # progress per block
python run_campaign.py run --block E2b_shift_real --limit 4   # try a few first
python run_campaign.py run --block E1_main_grid --force       # recompute
```

### Validate before you spend GPU hours

```bash
python run_campaign.py run --all --mock
```

The mock backend replaces YOLO with a synthetic trajectory generator and
exercises expansion, scheduling, resume, aggregation, statistics, tables
and figures end to end in seconds. Every mock record is stamped
`"mock": true`, so synthetic numbers cannot be mistaken for
measurements. Run it after any change to the pipeline.

---

## Outputs

```
results/<block>/runs/<run_id>.json   one record per run: config, seed, git commit,
                                     platform, full trajectory, profiling, warnings
results/campaign_state.json          the resume journal
tables/*.csv, tables/*.tex           thesis-ready, regenerated from the records
figures/*.pdf, figures/*.png         vector for the document, 300 dpi for slides
```

Tables and figures are **never** edited by hand. `python run_campaign.py
report` rebuilds them all from stored records, so a number in the thesis
is always the number the code produced, traceable through the run record
to the commit and machine that produced it.

---

## Repository layout

```
configs/campaign.yaml      the experiment matrix (single source of truth)
run_campaign.py            plan / run / status / report / matrix
src/
  campaign.py              expansion, scheduling, resume, provenance, mock backend
  executors.py             RunSpec -> real experiment
  al_loop.py               the canonical instrumented Active Learning loop
  shift_pipeline.py        covariate-shift variant (source -> target)
  acquisition.py           Random / Deterministic / BALD / BALD+Diversity
  mc_dropout.py            MC Dropout estimator, BALD, feature extraction
  uncertainty_extended.py  BSB / PSB, conformal prediction, SWAG
  deep_ensemble.py         Deep Ensembles, with its cost profile
  metrics.py               AUC (raw and normalised), stability, label efficiency
  diversity_metrics.py     cluster coverage, entropy, redundancy, representativeness
  forgetting.py            BWT, forgetting measure, retention, stability gap
  profiling.py             latency percentiles, memory, CPU/GPU, power, network, Jetson
  stats.py                 bootstrap CIs, Wilcoxon, Holm-Bonferroni, Cliff's delta
  aggregate.py             per-run records -> per-condition statistics
scripts/
  audit_datasets.py        pre-flight data check
  download_datasets.py     acquisition
  prepare_datasets.py      YOLO conversion + unified taxonomy
  make_tables.py           every LaTeX table
  make_figures.py          every figure
docs/
  ALGORITHM.md             formal notation, Algorithm 1 (with LaTeX), complexity
  EXPERIMENT_MATRIX.md     generated: board item -> experiment
  REPRODUCIBILITY.md       exact steps to reproduce each thesis claim
```

---

## Reporting conventions

**AUC is reported twice.** The raw integral of Equation 5.1 scales with
the annotation budget, which is why the preliminary abstract carries a
value of 240.65 — a number a reader cannot interpret without knowing the
budget. Every table therefore also reports the normalised form (the
integral divided by the annotation span), which lies in $[0,1]$ and reads
as mean detection quality per unit of annotation.

**No mean travels alone.** Every aggregate carries its standard deviation
across seeds; the primary tables also carry a bootstrap confidence
interval ($n = 1000$, $\alpha = 0.05$). Comparisons use the exact
Wilcoxon signed-rank test on seed-paired values with Holm-Bonferroni
correction, and report Cliff's $\delta$ alongside — because at five seeds
the smallest attainable two-sided $p$ is 0.0625, and a large effect with
a non-significant $p$ is an underpowered comparison, not evidence of no
difference.

**Shortfalls are visible.** If a condition planned for five seeds
completed three, the aggregate records `n_seeds: 3` and `n_expected: 5`.
A failed run is journalled with its traceback and the block still
aggregates from what succeeded, reporting the gap. An empty results table
with no explanation is not reachable from this code.

**Profiling states its platform.** Every profiling record carries the
machine it was measured on. Workstation figures bound the embedded case
from below and are not a substitute for it; `E6_operational_profiling`
runs unmodified on a Jetson and reads `tegrastats` rail power there.

---

## Reproducing the thesis

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the exact
command behind each claim. In short:

```bash
python run_campaign.py run --block E1_main_grid E2a_shift_baseline \
                                   E2b_shift_real E6_operational_profiling
python run_campaign.py run --block E3_uncertainty_methods E5_long_horizon E2c_shift_intensity
python run_campaign.py run --block E4_sensitivity
python run_campaign.py report
```

Seeds are `base_seed + k` with `base_seed = 42`, identical across
strategies so the statistical pairing is real. cuDNN determinism is
enabled. Each run record embeds the git commit, the resolved config and
the platform fingerprint.

---

## Citation

```bibtex
@phdthesis{costa2026uavedge,
  author = {Leonardo Ferreira da Costa},
  title  = {Continuous Model Adaptation in UAV--Edge Systems via
            Uncertainty-Guided Active Learning},
  school = {Universidade Federal do Cear\'a (UFC)},
  year   = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE). Source datasets keep their own licences.
