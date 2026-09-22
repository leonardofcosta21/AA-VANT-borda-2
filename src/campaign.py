"""
Campaign engine: expansion, scheduling, resume and provenance.

The previous repository had one runner script per experiment family, each
with its own CLI, its own output layout and its own aggregation code. That
is how the last campaign ended with per-strategy shift files on disk and
an empty combined table: nothing owned the question "did every run that
was supposed to happen actually happen".

This module makes the campaign a first-class object.

    spec (configs/campaign.yaml)
        -> expand_blocks()   one RunSpec per atomic run
        -> CampaignRunner    executes, records, resumes
        -> results/<block>/runs/<run_id>.json

Guarantees
----------
Resumable
    Every completed run is journalled. Re-invoking the same block skips
    what is already on disk, so a campaign interrupted at hour 30 of 40
    costs ten hours to finish, not forty.

Nothing fails silently
    A crashed run is recorded as a failed run with its traceback, and the
    block still aggregates from what succeeded while reporting the
    shortfall. An empty results table is now impossible to produce
    without also producing the explanation.

Provenance
    Each run carries the git commit, the resolved configuration, the
    platform fingerprint and the seed. A number in the thesis can be
    traced back to the exact code and machine that produced it, which is
    contribution C6.

Testable without a GPU
    ``backend="mock"`` swaps the YOLO training loop for a synthetic
    trajectory generator with plausible learning dynamics. It validates
    expansion, scheduling, resume, aggregation, statistics, tables and
    figures end to end in seconds. It exists to test the plumbing and its
    output is stamped ``"mock": true`` in every record, so mock numbers
    can never be mistaken for measurements.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import yaml

__all__ = [
    "RunSpec",
    "CampaignSpec",
    "CampaignRunner",
    "load_campaign",
    "expand_blocks",
]


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------

@dataclass
class RunSpec:
    """One atomic, independently schedulable unit of work."""

    run_id: str
    block_id: str
    kind: str
    params: Dict = field(default_factory=dict)
    seed: int = 42
    seed_index: int = 0
    priority: int = 5

    def fingerprint(self) -> str:
        payload = json.dumps(
            {"kind": self.kind, "params": self.params, "seed": self.seed},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha1(payload.encode()).hexdigest()[:12]

    def as_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CampaignSpec:
    raw: Dict
    path: Path

    @property
    def name(self) -> str:
        return self.raw.get("campaign", {}).get("name", "campaign")

    @property
    def base_seed(self) -> int:
        return int(self.raw.get("campaign", {}).get("base_seed", 42))

    @property
    def n_seeds(self) -> int:
        return int(self.raw.get("campaign", {}).get("n_seeds", 5))

    @property
    def output_root(self) -> Path:
        return Path(self.raw.get("campaign", {}).get("output_root", "results"))

    @property
    def defaults(self) -> Dict:
        return dict(self.raw.get("defaults", {}))

    @property
    def datasets(self) -> Dict[str, str]:
        return dict(self.raw.get("datasets", {}))

    @property
    def blocks(self) -> List[Dict]:
        return list(self.raw.get("blocks", []))

    @property
    def reporting(self) -> Dict:
        return dict(self.raw.get("reporting", {}))

    def block(self, block_id: str) -> Optional[Dict]:
        for b in self.blocks:
            if b.get("id") == block_id:
                return b
        return None


def load_campaign(path: str = "configs/campaign.yaml") -> CampaignSpec:
    p = Path(path)
    with open(p) as fh:
        raw = yaml.safe_load(fh)
    return CampaignSpec(raw=raw, path=p)


# ---------------------------------------------------------------------------
# Expansion
# ---------------------------------------------------------------------------

def expand_blocks(
    spec: CampaignSpec, block_ids: Optional[Iterable[str]] = None
) -> List[RunSpec]:
    """Turn the declarative matrix into the concrete list of runs.

    Expansion is pure and deterministic: calling it twice yields the same
    run ids in the same order, which is what makes resume reliable and
    lets ``plan`` print an accurate cost estimate before anything runs.
    """
    wanted = set(block_ids) if block_ids else None
    runs: List[RunSpec] = []

    for block in spec.blocks:
        block_id = block.get("id")
        if wanted and block_id not in wanted:
            continue
        kind = block.get("kind")
        expander = _EXPANDERS.get(kind)
        if expander is None:
            raise ValueError(
                f"block '{block_id}' has unknown kind '{kind}'. "
                f"Known kinds: {sorted(_EXPANDERS)}"
            )
        runs.extend(expander(spec, block))

    runs.sort(key=lambda r: (r.priority, r.block_id, r.run_id))
    return runs


def _merged(spec: CampaignSpec, block: Dict, extra: Dict) -> Dict:
    params = spec.defaults.copy()
    for key in (
        "cycles",
        "budget",
        "T",
        "collect_diversity_metrics",
        "collect_profiling",
        "collect_per_domain_eval",
        "per_domain_eval_every",
        "link_mbps",
        "realtime_budget_ms",
        "ensemble_members",
    ):
        if key in block:
            params[key] = block[key]
    params.update(extra)
    return params


def _seeds(spec: CampaignSpec, block: Dict) -> List[int]:
    n = int(block.get("n_seeds", spec.n_seeds))
    return [spec.base_seed + k for k in range(n)]


def _resolve_dataset(spec: CampaignSpec, name: str) -> str:
    return spec.datasets.get(name, name)


def _expand_al_grid(spec: CampaignSpec, block: Dict) -> List[RunSpec]:
    runs: List[RunSpec] = []
    datasets = [block.get("dataset")] + list(block.get("also_on", []))
    datasets = [d for d in datasets if d]
    strategies = block.get("strategies", ["bald_diversity"])
    l0_sizes = block.get("l0_sizes", [100])
    budgets = block.get("budgets", [block.get("budget", spec.defaults.get("budget", 50))])
    seeds = _seeds(spec, block)

    for dataset, strategy, l0, budget in product(datasets, strategies, l0_sizes, budgets):
        for k, seed in enumerate(seeds):
            params = _merged(
                spec,
                block,
                {
                    "dataset": dataset,
                    "data_root": _resolve_dataset(spec, dataset),
                    "strategy": strategy,
                    "l0_size": l0,
                    "budget": budget,
                },
            )
            runs.append(
                RunSpec(
                    run_id=f"{block['id']}__{dataset}__{strategy}__L0{l0}__B{budget}__s{seed}",
                    block_id=block["id"],
                    kind="al_run",
                    params=params,
                    seed=seed,
                    seed_index=k,
                    priority=int(block.get("priority", 5)),
                )
            )
    return runs


def _expand_shift_grid(spec: CampaignSpec, block: Dict) -> List[RunSpec]:
    runs: List[RunSpec] = []
    strategies = block.get("strategies", ["bald_diversity"])
    l0_sizes = block.get("l0_sizes", [100])
    seeds = _seeds(spec, block)

    conditions: List[Dict] = []
    if block.get("simulate"):
        source = block.get("source")
        for intensity in block.get("intensities", ["moderate"]):
            conditions.append(
                {
                    "source": source,
                    "target": None,
                    "simulate": True,
                    "intensity": intensity,
                    "label": f"{source}_sim_{intensity}",
                }
            )
    else:
        for pair in block.get("pairs", []):
            conditions.append(
                {
                    "source": pair["source"],
                    "target": pair["target"],
                    "simulate": False,
                    "intensity": "real",
                    "label": f"{pair['source']}_to_{pair['target']}",
                }
            )

    for condition, strategy, l0 in product(conditions, strategies, l0_sizes):
        for k, seed in enumerate(seeds):
            params = _merged(
                spec,
                block,
                {
                    "source": condition["source"],
                    "source_root": _resolve_dataset(spec, condition["source"]),
                    "target": condition["target"],
                    "target_root": _resolve_dataset(spec, condition["target"])
                    if condition["target"]
                    else None,
                    "simulate_shift": condition["simulate"],
                    "shift_intensity": condition["intensity"],
                    "pair_label": condition["label"],
                    "strategy": strategy,
                    "l0_size": l0,
                },
            )
            runs.append(
                RunSpec(
                    run_id=f"{block['id']}__{condition['label']}__{strategy}__L0{l0}__s{seed}",
                    block_id=block["id"],
                    kind="shift_run",
                    params=params,
                    seed=seed,
                    seed_index=k,
                    priority=int(block.get("priority", 5)),
                )
            )
    return runs


def _expand_shift_baseline(spec: CampaignSpec, block: Dict) -> List[RunSpec]:
    """Source-trained model evaluated on the target with no adaptation."""
    runs: List[RunSpec] = []
    seeds = _seeds(spec, block)
    for pair in block.get("pairs", []):
        for l0 in block.get("l0_sizes", [100]):
            for k, seed in enumerate(seeds):
                label = f"{pair['source']}_to_{pair['target']}"
                params = _merged(
                    spec,
                    block,
                    {
                        "source": pair["source"],
                        "source_root": _resolve_dataset(spec, pair["source"]),
                        "target": pair["target"],
                        "target_root": _resolve_dataset(spec, pair["target"]),
                        "pair_label": label,
                        "l0_size": l0,
                        "cycles": 0,          # no adaptation: that is the point
                        "strategy": "none",
                    },
                )
                runs.append(
                    RunSpec(
                        run_id=f"{block['id']}__{label}__L0{l0}__s{seed}",
                        block_id=block["id"],
                        kind="shift_baseline_run",
                        params=params,
                        seed=seed,
                        seed_index=k,
                        priority=int(block.get("priority", 5)),
                    )
                )
    return runs


def _expand_uncertainty(spec: CampaignSpec, block: Dict) -> List[RunSpec]:
    runs: List[RunSpec] = []
    seeds = _seeds(spec, block)
    dataset = block.get("dataset", "sard")
    for method, l0 in product(block.get("methods", []), block.get("l0_sizes", [100])):
        for k, seed in enumerate(seeds):
            params = _merged(
                spec,
                block,
                {
                    "dataset": dataset,
                    "data_root": _resolve_dataset(spec, dataset),
                    "method": method,
                    "l0_size": l0,
                    "ensemble_members": block.get("ensemble_members", 5),
                },
            )
            runs.append(
                RunSpec(
                    run_id=f"{block['id']}__{method}__L0{l0}__s{seed}",
                    block_id=block["id"],
                    kind="uncertainty_run",
                    params=params,
                    seed=seed,
                    seed_index=k,
                    priority=int(block.get("priority", 5)),
                )
            )
    return runs


def _expand_sensitivity(spec: CampaignSpec, block: Dict) -> List[RunSpec]:
    """One-factor-at-a-time sweeps around the baseline configuration.

    A full factorial over four parameters would be 4x3x4x3 = 144 cells
    before seeds, which does not fit the schedule. OFAT costs the sum
    rather than the product and still answers the question the thesis
    asks -- which single parameter matters -- while the absence of
    interaction terms is declared rather than hidden.
    """
    runs: List[RunSpec] = []
    seeds = _seeds(spec, block)
    dataset = block.get("dataset", "sard")
    strategy = block.get("strategy", "bald_diversity")

    for param_name, values in (block.get("sweeps") or {}).items():
        for value in values:
            for l0 in block.get("l0_sizes", [100]):
                for k, seed in enumerate(seeds):
                    extra = {
                        "dataset": dataset,
                        "data_root": _resolve_dataset(spec, dataset),
                        "strategy": strategy,
                        "l0_size": l0,
                        "sweep_param": param_name,
                        "sweep_value": value,
                    }
                    if param_name != "cluster_divisor":
                        extra[param_name] = value
                    else:
                        extra["cluster_divisor"] = value
                    params = _merged(spec, block, extra)
                    runs.append(
                        RunSpec(
                            run_id=f"{block['id']}__{param_name}_{value}__L0{l0}__s{seed}",
                            block_id=block["id"],
                            kind="al_run",
                            params=params,
                            seed=seed,
                            seed_index=k,
                            priority=int(block.get("priority", 5)),
                        )
                    )
    return runs


def _expand_profiling(spec: CampaignSpec, block: Dict) -> List[RunSpec]:
    runs: List[RunSpec] = []
    seeds = _seeds(spec, block)
    dataset = block.get("dataset", "sard")
    for strategy, l0 in product(
        block.get("strategies", ["bald_diversity"]), block.get("l0_sizes", [100])
    ):
        for k, seed in enumerate(seeds):
            params = _merged(
                spec,
                block,
                {
                    "dataset": dataset,
                    "data_root": _resolve_dataset(spec, dataset),
                    "strategy": strategy,
                    "l0_size": l0,
                    "measure": block.get("measure", []),
                    "collect_profiling": True,
                },
            )
            runs.append(
                RunSpec(
                    run_id=f"{block['id']}__{strategy}__L0{l0}__s{seed}",
                    block_id=block["id"],
                    kind="profiling_run",
                    params=params,
                    seed=seed,
                    seed_index=k,
                    priority=int(block.get("priority", 5)),
                )
            )
    return runs


_EXPANDERS: Dict[str, Callable[[CampaignSpec, Dict], List[RunSpec]]] = {
    "al_grid": _expand_al_grid,
    "shift_grid": _expand_shift_grid,
    "shift_baseline": _expand_shift_baseline,
    "uncertainty_comparison": _expand_uncertainty,
    "sensitivity": _expand_sensitivity,
    "profiling": _expand_profiling,
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class CampaignRunner:
    """Executes RunSpecs, journals them, and can pick up where it stopped."""

    def __init__(
        self,
        spec: CampaignSpec,
        backend: str = "real",
        output_root: Optional[str] = None,
        dry_run: bool = False,
        continue_on_error: bool = True,
    ):
        self.spec = spec
        self.backend_name = backend
        self.dry_run = dry_run
        self.continue_on_error = continue_on_error
        self.root = Path(output_root or spec.output_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.journal_path = self.root / "campaign_state.json"
        self.journal = self._load_journal()
        self._backend = _get_backend(backend)

    # -- journal ---------------------------------------------------------
    def _load_journal(self) -> Dict:
        if self.journal_path.exists():
            try:
                with open(self.journal_path) as fh:
                    return json.load(fh)
            except Exception:
                # A corrupted journal must not block a campaign; it is a
                # cache of what is already on disk, and the run files are
                # the source of truth.
                backup = self.journal_path.with_suffix(".corrupt.json")
                self.journal_path.rename(backup)
        return {"runs": {}, "started": _now(), "campaign": self.spec.name}

    def _save_journal(self) -> None:
        tmp = self.journal_path.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            json.dump(self.journal, fh, indent=2, default=str)
        tmp.replace(self.journal_path)

    def run_path(self, run: RunSpec) -> Path:
        return self.root / run.block_id / "runs" / f"{run.run_id}.json"

    def is_done(self, run: RunSpec) -> bool:
        path = self.run_path(run)
        if not path.exists():
            return False
        entry = self.journal["runs"].get(run.run_id)
        if entry and entry.get("status") == "completed":
            # Re-run when the spec changed under the same id, otherwise a
            # config edit would be silently ignored on resume.
            return entry.get("fingerprint") == run.fingerprint()
        return False

    # -- execution -------------------------------------------------------
    def execute(
        self, runs: List[RunSpec], force: bool = False, limit: Optional[int] = None
    ) -> Dict:
        pending = [r for r in runs if force or not self.is_done(r)]
        skipped = len(runs) - len(pending)
        if limit:
            pending = pending[:limit]

        print(f"[campaign] backend={self.backend_name} total={len(runs)} "
              f"pending={len(pending)} already_done={skipped}")
        if self.dry_run:
            for r in pending:
                print(f"  would run: {r.run_id}")
            return {"planned": len(pending), "skipped": skipped, "dry_run": True}

        provenance = _provenance()
        completed, failed = 0, 0
        t_campaign = time.time()

        for i, run in enumerate(pending, 1):
            print(f"\n[{i}/{len(pending)}] {run.run_id}")
            started = time.time()
            record = {
                "run_id": run.run_id,
                "block_id": run.block_id,
                "kind": run.kind,
                "seed": run.seed,
                "seed_index": run.seed_index,
                "params": run.params,
                "fingerprint": run.fingerprint(),
                "provenance": provenance,
                "backend": self.backend_name,
                "mock": self.backend_name == "mock",
                "started_at": _now(),
            }
            try:
                result = self._backend(run)
                record["status"] = "completed"
                record["result"] = result
                completed += 1
            except KeyboardInterrupt:
                print("\n[campaign] interrupted; progress is journalled")
                raise
            except Exception as exc:
                record["status"] = "failed"
                record["error"] = str(exc)
                record["traceback"] = traceback.format_exc()
                failed += 1
                print(f"  FAILED: {exc}")
                if not self.continue_on_error:
                    self._write_record(run, record)
                    raise
            record["duration_s"] = time.time() - started
            record["finished_at"] = _now()
            self._write_record(run, record)
            print(f"  {record['status']} in {record['duration_s']:.1f}s")

        summary = {
            "total": len(runs),
            "attempted": len(pending),
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "wall_clock_s": time.time() - t_campaign,
        }
        print(
            f"\n[campaign] completed={completed} failed={failed} "
            f"skipped={skipped} in {summary['wall_clock_s'] / 60:.1f} min"
        )
        if failed:
            print(
                f"[campaign] {failed} run(s) failed. Their tracebacks are in the "
                f"run JSONs under {self.root}; aggregation will report the shortfall."
            )
        return summary

    def _write_record(self, run: RunSpec, record: Dict) -> None:
        path = self.run_path(run)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(record, fh, indent=2, default=_json_default)
        self.journal["runs"][run.run_id] = {
            "status": record["status"],
            "fingerprint": record["fingerprint"],
            "path": str(path),
            "finished_at": record.get("finished_at"),
            "duration_s": record.get("duration_s"),
            "backend": self.backend_name,
        }
        self.journal["updated"] = _now()
        self._save_journal()

    # -- inspection ------------------------------------------------------
    def status(self, runs: List[RunSpec]) -> Dict:
        by_block: Dict[str, Dict[str, int]] = {}
        for run in runs:
            bucket = by_block.setdefault(
                run.block_id, {"total": 0, "completed": 0, "failed": 0, "pending": 0}
            )
            bucket["total"] += 1
            entry = self.journal["runs"].get(run.run_id)
            if entry is None:
                bucket["pending"] += 1
            elif entry["status"] == "completed":
                bucket["completed"] += 1
            else:
                bucket["failed"] += 1
        return by_block

    def load_block_results(self, block_id: str) -> List[Dict]:
        """Every completed record for a block, for aggregation."""
        directory = self.root / block_id / "runs"
        if not directory.exists():
            return []
        records = []
        for path in sorted(directory.glob("*.json")):
            try:
                with open(path) as fh:
                    record = json.load(fh)
            except Exception:
                continue
            if record.get("status") == "completed":
                records.append(record)
        return records


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def _get_backend(name: str) -> Callable[[RunSpec], Dict]:
    if name == "mock":
        return _mock_backend
    if name == "real":
        return _real_backend
    raise ValueError(f"unknown backend '{name}' (expected 'real' or 'mock')")


def _real_backend(run: RunSpec) -> Dict:
    """Dispatch to the actual experiment implementations.

    Imported lazily so that planning, status and mock runs work on a
    machine without torch or ultralytics installed.
    """
    from src import executors

    dispatch = {
        "al_run": executors.run_al,
        "shift_run": executors.run_shift,
        "shift_baseline_run": executors.run_shift_baseline,
        "uncertainty_run": executors.run_uncertainty,
        "profiling_run": executors.run_profiling,
    }
    fn = dispatch.get(run.kind)
    if fn is None:
        raise ValueError(f"no executor registered for kind '{run.kind}'")
    return fn(run)


def _mock_backend(run: RunSpec) -> Dict:
    """Synthetic but plausible results, for validating the plumbing.

    The learning curve is a saturating exponential in the number of
    labels with seed-dependent noise, and informed strategies are given a
    modest advantage. The shape matters only insofar as it exercises
    every downstream consumer -- AUC, stability, statistics, tables,
    figures -- with realistic values.
    """
    from src.metrics import curve_summary, final_metrics

    params = run.params
    rng = np.random.default_rng(run.seed + hash(run.run_id) % 10_000)

    strategy = params.get("strategy", params.get("method", "random"))
    l0 = int(params.get("l0_size", 100))
    budget = int(params.get("budget", 50))
    cycles = int(params.get("cycles", 8))

    advantage = {
        "random": 0.0,
        "none": 0.0,
        "deterministic": -0.01,
        "bald_only": 0.02,
        "bald_diversity": 0.035,
        "mc_dropout": 0.03,
        "deep_ensemble": 0.04,
        "conformal": 0.025,
        "swag": 0.028,
        "bsb": 0.022,
        "psb": 0.02,
        "bsb_diversity": 0.033,
    }.get(strategy, 0.0)

    ceiling = 0.78 + advantage + rng.normal(0, 0.01)
    rate = 400.0
    trajectory = []
    for k in range(cycles + 1):
        n_labelled = l0 + k * budget
        base = ceiling * (1 - np.exp(-n_labelled / rate))
        noise = rng.normal(0, 0.02)
        map50 = float(np.clip(base + noise, 0.0, 1.0))
        trajectory.append(
            {
                "cycle": k,
                "n_labelled": n_labelled,
                "n_new": 0 if k == 0 else budget,
                "mAP50": map50,
                "mAP50_95": round(map50 * 0.47, 6),
                "precision": float(np.clip(map50 * 1.12 + rng.normal(0, 0.02), 0, 1)),
                "recall": float(np.clip(map50 * 1.02 + rng.normal(0, 0.02), 0, 1)),
                "update_time": float(20 + rng.normal(0, 2)),
                "trigger_reason": "initial" if k == 0 else f"accumulation(n={budget})",
            }
        )

    l_max = budget * cycles
    result: Dict = {
        "strategy": strategy,
        "l0_size": l0,
        "budget": budget,
        "cycles": cycles,
        "dataset": params.get("dataset"),
        "trajectory": trajectory,
        "inference_latency_ms": float(5.6 + rng.normal(0, 0.2)),
    }
    result.update(curve_summary(trajectory, l_max=l_max))
    result.update(final_metrics(trajectory))

    if params.get("collect_diversity_metrics"):
        spread = 0.85 if "diversity" in strategy else 0.42
        result["diversity_coverage_mean"] = float(
            np.clip(spread + rng.normal(0, 0.05), 0, 1)
        )
        result["diversity_selection_entropy_mean"] = float(
            np.clip(spread - 0.05 + rng.normal(0, 0.05), 0, 1)
        )
        result["diversity_redundancy_rate_mean"] = float(
            np.clip((1 - spread) * 0.4 + rng.normal(0, 0.02), 0, 1)
        )

    if params.get("collect_profiling") or run.kind == "profiling_run":
        latency = result["inference_latency_ms"]
        # Key names mirror executors._flatten_profiling exactly, so the
        # mock exercises every column of the H4 table rather than
        # leaving some blank and hiding a naming mismatch.
        result["profiling"] = {
            "latency_ms_mean": latency,
            "latency_ms_p95": latency * 1.35,
            "latency_ms_p99": latency * 1.9,
            "throughput_fps": float(1000.0 / latency),
            "frames_over_budget_pct": float(max(0.0, rng.normal(0.4, 0.3))),
            "meets_budget_at_p95": True,
            "gpu_energy_j_per_cycle": float(420 + rng.normal(0, 40)),
            "finetune_s_total": float((24 + rng.normal(0, 3)) * cycles),
            "network_mb_total": float((budget * 0.35 + 6.2) * cycles),
            "peak_host_memory_mb": float(2400 + rng.normal(0, 150)),
            "peak_gpu_memory_mb": float(1750 + rng.normal(0, 90)),
            "cpu_util_pct_mean": float(38 + rng.normal(0, 5)),
            "gpu_util_pct_mean": float(64 + rng.normal(0, 8)),
            "gpu_power_w_mean": float(78 + rng.normal(0, 6)),
            "gpu_energy_j": float(420 + rng.normal(0, 40)),
            "finetune_s_per_cycle": float(24 + rng.normal(0, 3)),
            "network_mb_per_cycle": float(budget * 0.35 + 6.2),
            "mc_overhead_ratio": float(params.get("T", 8)) if "bald" in strategy else 1.0,
            "platform": {"mock": True},
        }

    if params.get("collect_per_domain_eval"):
        domains = ["sard", "visdrone", "floodnet", "aider"]
        n_evals = max(2, cycles // max(int(params.get("per_domain_eval_every", 1)), 1))
        matrix = []
        for k in range(n_evals):
            row = []
            for d, _ in enumerate(domains):
                # Prior domains drift down slowly, the adapted domain climbs.
                trend = -0.012 * k if d > 0 else 0.02 * k
                row.append(float(np.clip(0.62 + trend + rng.normal(0, 0.015), 0, 1)))
            matrix.append(row)
        arr = np.array(matrix)
        prior = arr[:, 1:]
        result["forgetting"] = {
            "metric": "mAP50",
            "domains": domains,
            "cycles": list(range(0, n_evals * int(params.get("per_domain_eval_every", 1)),
                                 max(int(params.get("per_domain_eval_every", 1)), 1))),
            "performance_matrix": arr.tolist(),
            "backward_transfer": float(np.mean(prior[-1] - prior[0])),
            "forgetting_measure": float(np.mean(prior.max(axis=0) - prior[-1])),
            "stability_gap": float(np.max(-np.diff(prior, axis=0))),
            "retention_mean_prior_domains": float(
                np.mean(prior[-1] / np.maximum(prior.max(axis=0), 1e-9))
            ),
            "plasticity": float(arr[-1, 0] - arr[0, 0]),
            "source_retention": float(arr[-1, 0] / max(arr[:, 0].max(), 1e-9)),
        }

    if run.kind in ("shift_run", "shift_baseline_run"):
        drop = rng.uniform(0.18, 0.42)
        result["baseline_target_mAP50"] = float(
            max(0.0, trajectory[-1]["mAP50"] - drop)
        )
        result["baseline_target_recall"] = float(
            max(0.0, trajectory[-1]["recall"] - drop)
        )
        result["pair_label"] = params.get("pair_label")
        result["shift_intensity"] = params.get("shift_intensity")

    result["mock"] = True
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _provenance() -> Dict:
    """Code and machine fingerprint stored with every run (C6)."""
    info: Dict = {
        "timestamp": _now(),
        "python": sys.version.split()[0],
        "argv": " ".join(sys.argv),
    }
    for key, cmd in (
        ("git_commit", ["git", "rev-parse", "HEAD"]),
        ("git_branch", ["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        ("git_dirty", ["git", "status", "--porcelain"]),
    ):
        try:
            out = subprocess.run(
                cmd, capture_output=True, text=True, timeout=5
            ).stdout.strip()
            info[key] = bool(out) if key == "git_dirty" else out
        except Exception:
            info[key] = None
    try:
        from src.profiling import describe_platform

        info["platform"] = describe_platform()
    except Exception:
        pass
    return info


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)
