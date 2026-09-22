# Reproducibility

Contribution **C6** is the claim that every number in the thesis can be
reproduced. This document is what makes that checkable: for each claim,
the command that produces it and the file it lands in.

---

## 0. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Tested on Python 3.10–3.12 with PyTorch ≥ 2.0 and Ultralytics ≥ 8.1 on a
single NVIDIA GPU. The reporting layer (`make report`) runs on CPU alone,
so tables and figures can be rebuilt anywhere from stored records.

**What is pinned and what is not.** Seeds, splits, hyperparameters and the
code commit are pinned and recorded per run. cuDNN determinism is
enabled. GPU floating-point reductions are still not bit-reproducible
across driver and hardware versions, so expect agreement to about three
decimal places rather than exact equality — which is why every reported
comparison goes through the seed-paired statistics rather than through a
single run.

---

## 1. Data

```bash
python scripts/download_datasets.py --all
python scripts/prepare_datasets.py
python scripts/audit_datasets.py --json results/dataset_audit.json
```

`audit_datasets.py` exits non-zero if any dataset is unusable, and names
what is wrong: missing labels, unnormalised coordinates, class ids
outside the taxonomy, or too few images for the $L_0$ values the campaign
asks for. Run it before committing GPU time.

Datasets requiring registration (LADI v2, parts of AIDER) are skipped by
the downloader with a message; fetch those manually into the paths
declared in `configs/campaign.yaml` and re-run the audit.

The unified multi-source dataset is built by `prepare_datasets.py` under
the three-superclass taxonomy (PERSON, VEHICLE, HAZARD) described in
Section 5.2.2 of the thesis.

---

## 2. Validate the pipeline before spending GPU hours

```bash
python -m pytest tests/ -q          # metrics, statistics, expansion, forgetting
make smoke                          # full campaign on the mock backend, seconds
```

`make smoke` writes to `results/_mock`, `tables/_mock` and
`figures/_mock`, so it cannot contaminate real results. Every mock record
carries `"mock": true`. Clean up with `make clean-mock`.

---

## 3. The campaign

```bash
python run_campaign.py plan         # runs and wall-clock estimate per block
make priority1                      # E1, E2a, E2b, E6
make e3 e5 e2                       # then the priority-2 blocks
make e4                             # sensitivity last
python run_campaign.py status       # what is done, failed, pending
```

Interrupt freely. Completed runs are journalled in
`results/campaign_state.json`, and re-running a block skips them. A run
whose configuration changed is re-run even under the same id, because the
journal stores a fingerprint of the resolved parameters.

---

## 4. Claim → command → artefact

| Thesis claim | Block | Command | Artefact |
|---|---|---|---|
| H1 annotation efficiency | E1 | `make e1` | `tables/main_grid.tex`, `figures/h1_learning_curves_*.pdf` |
| Sweet spot of $L_0$ and $B$ | E1 | `make e1` | `tables/sweet_spot.tex`, `figures/sweet_spot_heatmap_*.pdf` |
| H3 diversity gain | E1 | `make e1` | `tables/diversity.tex`, `figures/h3_diversity_*.pdf` |
| Domain gap without adaptation | E2a | `make e2` | `tables/shift_h2.tex` (column *Recall (no adapt.)*) |
| H2 stability under shift | E2b | `make e2` | `tables/shift_h2.tex`, `figures/h2_shift_stability.pdf` |
| Shift dose-response | E2c | `make e2` | `tables/intensity/shift_h2.tex` |
| MC Dropout vs alternatives | E3 | `make e3` | `tables/uncertainty_methods.tex` |
| Hyperparameter sensitivity | E4 | `make e4` | `tables/sensitivity.tex`, `figures/sensitivity.pdf` |
| Catastrophic forgetting | E5 | `make e5` | `tables/forgetting.tex`, `figures/forgetting_matrix.pdf` |
| H4 operational feasibility | E6 | `make e6` | `tables/h4_profiling.tex`, `figures/h4_*.pdf` |
| Significance of every comparison | all | `make report` | `tables/significance.tex` |

Rebuild every table and figure from whatever has completed:

```bash
python run_campaign.py report
```

---

## 5. What a run record contains

`results/<block>/runs/<run_id>.json`:

```
run_id, block_id, kind, seed, seed_index
params            the fully resolved configuration
fingerprint       hash of kind + params + seed
provenance        git commit, branch, dirty flag, argv, timestamp
                  platform: host, CPU, RAM, GPU, CUDA, torch, ultralytics
status            completed | failed
result            trajectory, curve metrics, profiling, diversity, forgetting,
                  network accounting, warnings
error, traceback  present only on failure
duration_s
```

Any figure in the thesis can therefore be traced to the exact code
revision and machine that produced it.

---

## 6. Where reproduction stops, and why

These are limitations to state in the thesis, not gaps to paper over.

**Embedded hardware.** All figures reported from this repository were
measured on the platform recorded in their run records. Workstation
latency, memory and power bound the embedded case from below; they are
not a substitute for it. `E6_operational_profiling` runs unmodified on a
Jetson and reads `tegrastats` rail power there, so the embedded numbers
require executing that block on the device, not re-deriving them.

**No real UAV.** Experiments run on public datasets, not on imagery from
a drone flown for this work. Crowd and disaster simulation is an
established field precisely because collecting faithful data in real
disaster conditions is dangerous, expensive and ethically fraught; the
absence of a dedicated flight campaign is a scope boundary, and it does
not affect the internal validity of the comparisons, which all use
identical splits and seeds.

**Oracle annotation.** The human-in-the-loop step is simulated by
revealing ground-truth labels. This isolates the acquisition policy from
annotator variability — which is what the hypotheses are about — but it
means annotation latency and label noise are not modelled, and any claim
about end-to-end mission timing inherits that assumption.

**Statistical power.** Five seeds per condition is what the schedule
allows. The smallest attainable two-sided $p$ from an exact Wilcoxon test
at $n = 5$ is 0.0625, so some real differences cannot reach significance.
Every comparison therefore reports Cliff's $\delta$ alongside the
$p$-value, and a large effect with a non-significant $p$ should be
described as underpowered, never as evidence of no difference.

**Sensitivity is one-factor-at-a-time.** A full factorial over $T$, $b$,
$K_{\text{clusters}}$ and $\tau_{\text{shift}}$ is 144 cells before
replication. Interaction effects are out of scope and are declared as
such rather than being implied by the marginal results.

**Ensemble refresh.** Deep Ensemble members are trained once and not
retrained each cycle; retraining $M$ models per cycle is not affordable
within the campaign. Its uncertainty estimates therefore become
progressively staler than MC Dropout's, which is recorded in the run and
must be stated when the two are compared.
