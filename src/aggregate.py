"""
Aggregation: from per-run records to thesis-ready statistics.

One rule governs this module: a condition is grouped by everything except
the seed. Runs that differ only in their seed are replicates; runs that
differ in anything else are different conditions. That single rule
replaces the ad-hoc grouping code that each old runner carried, and it is
what makes the seed pairing used by the statistical tests valid.

Shortfalls are reported, never hidden. If a condition was planned for
five seeds and three succeeded, the aggregate says ``n_seeds: 3`` and
``n_expected: 5``, so a table can never quietly present a three-seed mean
as a five-seed one.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from src.stats import bootstrap_ci, compare_strategies, significance_table

__all__ = [
    "group_runs",
    "aggregate_condition",
    "aggregate_block",
    "build_dataframe",
    "run_statistics",
]

# Condition key: every param that defines a distinct experimental cell.
CONDITION_KEYS = [
    "dataset",
    "strategy",
    "method",
    "estimator",
    "l0_size",
    "budget",
    "cycles",
    "pair_label",
    "shift_intensity",
    "sweep_param",
    "sweep_value",
]

# Scalars aggregated as mean, std and bootstrap CI across seeds.
METRIC_KEYS = [
    "auc",
    "auc_normalized",
    "final_mAP50",
    "final_mAP50_95",
    "final_precision",
    "final_recall",
    "recall_variance",
    "recall_monotonicity",
    "map_monotonicity",
    "map_max_drawdown",
    "map_smoothness",
    "inference_latency_ms",
    "mc_overhead_ratio",
    "labels_to_map50",
    "labels_to_map60",
    "labels_to_map70",
    "total_update_time_s",
    "diversity_coverage_mean",
    "diversity_selection_entropy_mean",
    "diversity_redundancy_rate_mean",
    "diversity_mean_pairwise_distance_mean",
    "baseline_target_mAP50",
    "baseline_target_recall",
]

PROFILING_KEYS = [
    "latency_ms_mean",
    "latency_ms_p95",
    "latency_ms_p99",
    "throughput_fps",
    "frames_over_budget_pct",
    "peak_host_memory_mb",
    "peak_gpu_memory_mb",
    "cpu_util_pct_mean",
    "gpu_util_pct_mean",
    "gpu_power_w_mean",
    "gpu_energy_j_per_cycle",
    "finetune_s_per_cycle",
    "network_mb_per_cycle",
    "mc_overhead_ratio",
]


def _condition_key(record: Dict) -> tuple:
    params = record.get("params", {})
    result = record.get("result", {})
    values = []
    for key in CONDITION_KEYS:
        value = params.get(key, result.get(key))
        values.append((key, _hashable(value)))
    return tuple(values)


def _hashable(value):
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, default=str)
    return value


def group_runs(records: Iterable[Dict]) -> Dict[tuple, List[Dict]]:
    """Group completed run records into conditions (everything but seed)."""
    groups: Dict[tuple, List[Dict]] = defaultdict(list)
    for record in records:
        if record.get("status") != "completed":
            continue
        groups[_condition_key(record)].append(record)
    for runs in groups.values():
        runs.sort(key=lambda r: r.get("seed", 0))
    return dict(groups)


def _extract(result: Dict, key: str):
    """Pull a metric from a result, looking inside `profiling` too."""
    if key in result and isinstance(result[key], (int, float)):
        return float(result[key])
    profiling = result.get("profiling") or {}
    if key in profiling and isinstance(profiling[key], (int, float)):
        return float(profiling[key])
    return None


def aggregate_condition(
    runs: List[Dict],
    n_expected: Optional[int] = None,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> Dict:
    """Mean, std and bootstrap CI for one condition across its seeds."""
    if not runs:
        return {}

    first = runs[0]
    params = first.get("params", {})
    result = first.get("result", {})

    agg: Dict = {
        "block_id": first.get("block_id"),
        "kind": first.get("kind"),
        "n_seeds": len(runs),
        "seeds": [r.get("seed") for r in runs],
        "mock": bool(first.get("mock")),
    }
    for key in CONDITION_KEYS:
        value = params.get(key, result.get(key))
        if value is not None:
            agg[key] = value
    if n_expected is not None:
        agg["n_expected"] = n_expected
        if len(runs) < n_expected:
            agg["shortfall"] = n_expected - len(runs)
            agg["shortfall_note"] = (
                f"{n_expected - len(runs)} of {n_expected} seeds did not complete; "
                "report this alongside the mean"
            )

    for key in METRIC_KEYS + PROFILING_KEYS:
        values = [
            v
            for v in (_extract(r.get("result", {}), key) for r in runs)
            if v is not None and not np.isnan(v)
        ]
        if not values:
            continue
        agg[f"{key}_mean"] = float(np.mean(values))
        agg[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        agg[f"{key}_values"] = values
        if len(values) > 1:
            ci = bootstrap_ci(values, n_boot=n_bootstrap, alpha=alpha)
            agg[f"{key}_ci_low"] = ci.lower
            agg[f"{key}_ci_high"] = ci.upper

    # Mean learning curve with a per-cycle standard deviation band.
    trajectories = [r.get("result", {}).get("trajectory", []) for r in runs]
    trajectories = [t for t in trajectories if t]
    if trajectories:
        max_len = max(len(t) for t in trajectories)
        for metric in ("mAP50", "recall", "precision"):
            matrix = np.full((len(trajectories), max_len), np.nan)
            for i, traj in enumerate(trajectories):
                for j, step in enumerate(traj):
                    value = step.get(metric)
                    if isinstance(value, (int, float)):
                        matrix[i, j] = value
            agg[f"curve_{metric}_mean"] = _nanmean(matrix)
            agg[f"curve_{metric}_std"] = _nanstd(matrix)
        labels = np.full((len(trajectories), max_len), np.nan)
        for i, traj in enumerate(trajectories):
            for j, step in enumerate(traj):
                value = step.get("n_labelled")
                if isinstance(value, (int, float)):
                    labels[i, j] = value
        agg["curve_n_labelled"] = _nanmean(labels)

    # Forgetting summaries, when the block collected them.
    forgetting = [
        r.get("result", {}).get("forgetting")
        for r in runs
        if r.get("result", {}).get("forgetting")
    ]
    if forgetting:
        for key in (
            "backward_transfer",
            "forgetting_measure",
            "stability_gap",
            "retention_mean_prior_domains",
            "plasticity",
            "source_retention",
        ):
            values = [
                f[key] for f in forgetting
                if isinstance(f.get(key), (int, float)) and not np.isnan(f[key])
            ]
            if values:
                agg[f"{key}_mean"] = float(np.mean(values))
                agg[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    warnings = []
    for r in runs:
        warnings.extend(r.get("result", {}).get("warnings", []) or [])
    if warnings:
        agg["warnings"] = sorted(set(warnings))
    return agg


def aggregate_block(
    records: List[Dict],
    n_expected_per_condition: Optional[int] = None,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> List[Dict]:
    """Aggregate every condition in a block."""
    groups = group_runs(records)
    return [
        aggregate_condition(
            runs, n_expected_per_condition, n_bootstrap=n_bootstrap, alpha=alpha
        )
        for runs in groups.values()
    ]


def build_dataframe(aggregates: List[Dict], drop_arrays: bool = True):
    """Tidy DataFrame from aggregates, ready for CSV or LaTeX."""
    import pandas as pd

    rows = []
    for agg in aggregates:
        row = {}
        for key, value in agg.items():
            if drop_arrays and (
                key.endswith("_values")
                or key.startswith("curve_")
                or key == "seeds"
            ):
                continue
            if isinstance(value, (list, dict)):
                continue
            row[key] = value
        rows.append(row)
    df = pd.DataFrame(rows)
    sort_cols = [
        c for c in ("dataset", "pair_label", "l0_size", "budget", "strategy", "method")
        if c in df.columns
    ]
    return df.sort_values(sort_cols).reset_index(drop=True) if sort_cols else df


def run_statistics(
    aggregates: List[Dict],
    metrics: Sequence[str],
    reference: str = "bald_diversity",
    group_by: Sequence[str] = ("dataset", "l0_size", "budget", "pair_label"),
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> List[Dict]:
    """Paired comparisons within each experimental context.

    Comparisons run *within* a context (same dataset, same L0, same
    budget, same shift pair) and never across them: comparing a strategy
    at L0=25 against another at L0=400 would answer no question the
    thesis asks and would inflate the multiplicity correction with
    meaningless tests.
    """
    contexts: Dict[tuple, Dict[str, Dict]] = defaultdict(dict)
    for agg in aggregates:
        context = tuple((k, agg.get(k)) for k in group_by)
        name = agg.get("strategy") or agg.get("method") or "unknown"
        contexts[context][name] = agg

    rows: List[Dict] = []
    for context, by_strategy in contexts.items():
        if len(by_strategy) < 2:
            continue
        context_label = ", ".join(f"{k}={v}" for k, v in context if v is not None)
        for metric in metrics:
            values = {
                name: agg.get(f"{metric}_values", [])
                for name, agg in by_strategy.items()
            }
            values = {k: v for k, v in values.items() if len(v) >= 2}
            if len(values) < 2:
                continue
            higher_is_better = metric not in {
                "recall_variance",
                "map_smoothness",
                "map_max_drawdown",
                "latency_ms_mean",
                "labels_to_map50",
                "labels_to_map60",
                "labels_to_map70",
            }
            results = compare_strategies(
                values,
                metric=metric,
                reference=reference if reference in values else None,
                alpha=alpha,
                n_boot=n_bootstrap,
                higher_is_better=higher_is_better,
            )
            for row in significance_table(results):
                row["context"] = context_label
                rows.append(row)
    return rows


def _nanmean(matrix: np.ndarray) -> List[float]:
    with np.errstate(all="ignore"):
        return [
            float(v) if not np.isnan(v) else None
            for v in np.nanmean(matrix, axis=0)
        ]


def _nanstd(matrix: np.ndarray) -> List[float]:
    with np.errstate(all="ignore"):
        return [
            float(v) if not np.isnan(v) else None
            for v in np.nanstd(matrix, axis=0)
        ]


def load_records(root: str, block_id: str) -> List[Dict]:
    """Read completed run records for a block from disk."""
    directory = Path(root) / block_id / "runs"
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
