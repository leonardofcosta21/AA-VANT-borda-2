"""
Executors: translate a RunSpec into a real experiment.

Thin by design. Each function resolves paths, builds the loop config and
delegates to the canonical implementation, so that the campaign engine
stays free of experiment-specific logic and the experiment code stays
free of scheduling logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from src.campaign import RunSpec

__all__ = [
    "run_al",
    "run_shift",
    "run_shift_baseline",
    "run_uncertainty",
    "run_profiling",
]


def _require_dataset(path: str, label: str) -> str:
    """Fail loudly and usefully when a dataset is missing.

    A missing dataset is the most common reason a campaign stalls, and
    the previous code surfaced it as an opaque exception deep inside the
    dataset manager. Saying which dataset and which command prepares it
    turns a debugging session into a one-line fix.
    """
    if not path:
        raise FileNotFoundError(f"no path configured for dataset '{label}'")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"dataset '{label}' not found at {p.resolve()}. "
            f"Run `python scripts/audit_datasets.py` to see what is present, "
            f"then `python scripts/download_datasets.py --dataset {label}` "
            f"followed by `python scripts/prepare_datasets.py` to build it."
        )
    return str(p)


def _output_dir(run: RunSpec) -> str:
    return str(Path("results") / run.block_id / "workdirs")


def run_al(run: RunSpec) -> Dict:
    """Standard closed-loop Active Learning run (E1, E4, E5)."""
    from src.al_loop import ALLoopConfig, run_al_loop

    params = dict(run.params)
    params["data_root"] = _require_dataset(
        params.get("data_root"), params.get("dataset", "?")
    )
    params.setdefault("output_dir", _output_dir(run))

    if params.get("collect_per_domain_eval"):
        params["per_domain_roots"] = _existing_domain_roots(params)

    cfg = ALLoopConfig.from_params(params, seed=run.seed)
    result = run_al_loop(cfg, run_id=run.run_id)
    if "sweep_param" in run.params:
        result["sweep_param"] = run.params["sweep_param"]
        result["sweep_value"] = run.params["sweep_value"]
    return result


def run_shift(run: RunSpec) -> Dict:
    """Covariate-shift adaptation run (E2b, E2c)."""
    from src.shift_pipeline import run_shift_experiment

    params = dict(run.params)
    source = _require_dataset(params.get("source_root"), params.get("source", "?"))
    target = params.get("target_root")
    if target:
        target = _require_dataset(target, params.get("target", "?"))

    config = _shift_config(params, run)
    result = run_shift_experiment(
        config=config,
        source_dir=source,
        target_dir=target,
        strategy=params["strategy"],
        l0_size=int(params["l0_size"]),
        run_id=run.seed_index,
        simulate_shift=bool(params.get("simulate_shift", target is None)),
        shift_intensity=params.get("shift_intensity", "moderate"),
        domain_fraction=float(params.get("domain_fraction", 0.5)),
        class_names=params.get("class_names"),
    )
    result["pair_label"] = params.get("pair_label")
    result["shift_intensity"] = params.get("shift_intensity")
    # Promote the summary to the top level so aggregation treats shift
    # runs and standard runs identically.
    summary = result.get("summary", {})
    result["auc"] = summary.get("auc")
    result["auc_normalized"] = summary.get("auc_normalized")
    result["final_recall"] = summary.get("final_recall")
    result["final_mAP50"] = summary.get("final_mAP50")
    result["recall_variance"] = summary.get("recall_variance")
    return result


def run_shift_baseline(run: RunSpec) -> Dict:
    """Train on source, evaluate on target, adapt nothing (E2a).

    The reference point for every shift claim: it isolates the domain gap
    from the effect of adaptation. Implemented as a zero-cycle shift run
    so the split construction is byte-identical to the adaptive runs it
    will be compared against.
    """
    from src.shift_pipeline import run_shift_experiment

    params = dict(run.params)
    source = _require_dataset(params.get("source_root"), params.get("source", "?"))
    target = _require_dataset(params.get("target_root"), params.get("target", "?"))

    config = _shift_config(params, run)
    config["active_learning"]["num_cycles"] = 0

    result = run_shift_experiment(
        config=config,
        source_dir=source,
        target_dir=target,
        strategy="random",       # never used: no cycle runs
        l0_size=int(params["l0_size"]),
        run_id=run.seed_index,
        simulate_shift=False,
        class_names=params.get("class_names"),
    )
    baseline = result.get("baseline_target_metrics", {})
    result["pair_label"] = params.get("pair_label")
    result["baseline_target_mAP50"] = baseline.get("mAP50")
    result["baseline_target_recall"] = baseline.get("recall")
    result["baseline_target_precision"] = baseline.get("precision")
    result["adaptation"] = "none"
    return result


def run_uncertainty(run: RunSpec) -> Dict:
    """Uncertainty-estimator comparison run (E3).

    Every method uses the same acquisition strategy family and the same
    splits; only the estimator changes. That is what makes the resulting
    table a comparison of estimators rather than of pipelines.
    """
    from src.al_loop import ALLoopConfig, run_al_loop

    params = dict(run.params)
    params["data_root"] = _require_dataset(
        params.get("data_root"), params.get("dataset", "?")
    )
    params.setdefault("output_dir", _output_dir(run))

    method = params["method"]
    # Margin-based criteria are acquisition strategies, not estimators;
    # everything else swaps the estimator behind a fixed strategy.
    strategy_methods = {"bsb": "bsb", "psb": "psb", "bsb_diversity": "bsb_diversity"}
    if method in strategy_methods:
        params["strategy"] = strategy_methods[method]
        params["estimator"] = "mc_dropout"
    elif method == "deterministic":
        params["strategy"] = "deterministic"
        params["estimator"] = "deterministic"
    else:
        params["strategy"] = "bald_diversity"
        params["estimator"] = method

    cfg = ALLoopConfig.from_params(params, seed=run.seed)
    result = run_al_loop(cfg, run_id=run.run_id)
    result["method"] = method

    if method == "deep_ensemble":
        result["cost_note"] = (
            f"{params.get('ensemble_members', 5)} independently trained models: "
            "storage and training cost scale linearly with the member count"
        )
    return result


def run_profiling(run: RunSpec) -> Dict:
    """Operational profiling run (E6).

    Runs the loop with full instrumentation and then extracts the H4
    figures into a flat block, so the profiling table is built from one
    well-defined structure rather than by digging through nested reports.
    """
    from src.al_loop import ALLoopConfig, run_al_loop

    params = dict(run.params)
    params["data_root"] = _require_dataset(
        params.get("data_root"), params.get("dataset", "?")
    )
    params["collect_profiling"] = True
    params.setdefault("output_dir", _output_dir(run))

    cfg = ALLoopConfig.from_params(params, seed=run.seed)
    result = run_al_loop(cfg, run_id=run.run_id)
    result["profiling"] = _flatten_profiling(result)
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shift_config(params: Dict, run: RunSpec) -> Dict:
    return {
        "experiment": {
            "seed": run.seed - run.seed_index,  # pipeline adds run_id back
            "device": params.get("device", "auto"),
            "output_dir": str(Path("results") / run.block_id),
        },
        "model": {
            "name": params.get("model", "yolov8n.pt"),
            "input_size": int(params.get("img_size", 640)),
            "conf_threshold": float(params.get("conf_threshold", 0.25)),
            "iou_threshold": float(params.get("iou_threshold", 0.5)),
        },
        "uncertainty": {
            "T": int(params.get("T", 8)),
            "dropout_rate": float(params.get("dropout_rate", 0.1)),
            "alpha": float(params.get("alpha", 0.1)),
        },
        "active_learning": {
            "budget_per_cycle": int(params.get("budget", 50)),
            "num_cycles": int(params.get("cycles", 8)),
        },
        "diversity": {"beta": int(params.get("beta", 5))},
        "training": {
            "batch_size": int(params.get("batch_size", 16)),
            "initial_lr": float(params.get("initial_lr", 0.01)),
            "incremental_lr": float(params.get("incremental_lr", 0.001)),
            "n_min": int(params.get("n_min", 50)),
            "shift_window": int(params.get("shift_window", 3)),
            "tau_shift_percentile": float(params.get("tau_shift_percentile", 90)),
            "epochs_per_cycle": int(params.get("epochs_per_cycle", 5)),
        },
        "dataset": {"test_ratio": float(params.get("test_fraction", 0.2))},
    }


def _existing_domain_roots(params: Dict) -> Dict[str, str]:
    """Domain test sets for forgetting analysis, skipping absent datasets."""
    import yaml as _yaml

    campaign_path = Path("configs/campaign.yaml")
    if not campaign_path.exists():
        return {}
    with open(campaign_path) as fh:
        datasets = (_yaml.safe_load(fh) or {}).get("datasets", {})
    return {
        name: path
        for name, path in datasets.items()
        if name != "unified" and Path(path).exists()
    }


def _flatten_profiling(result: Dict) -> Dict:
    """Collapse the nested instrumentation into the H4 reporting block."""
    inference = result.get("inference_profile", {})
    cycles = result.get("profiling_per_cycle", [])
    network = result.get("network", {})

    def peak(path_a: str, key: str):
        values = []
        for entry in cycles:
            block = entry.get(path_a) or {}
            value = block.get(key)
            if isinstance(value, (int, float)):
                values.append(value)
        return max(values) if values else None

    def mean(path_a: str, key: str):
        values = []
        for entry in cycles:
            block = entry.get(path_a) or {}
            value = block.get(key)
            if isinstance(value, (int, float)):
                values.append(value)
        return sum(values) / len(values) if values else None

    update_times = [e.get("update_time_s", 0.0) for e in cycles]
    return {
        "latency_ms_mean": inference.get("latency_ms_mean"),
        "latency_ms_p95": inference.get("latency_ms_p95"),
        "latency_ms_p99": inference.get("latency_ms_p99"),
        "throughput_fps": inference.get("throughput_fps"),
        "frames_over_budget_pct": inference.get("frames_over_budget_pct"),
        "meets_budget_at_p95": inference.get("meets_budget_at_p95"),
        "mc_overhead_ratio": result.get("mc_overhead_ratio"),
        "peak_host_memory_mb": peak("finetune", "rss_mb_peak")
        or peak("acquisition", "rss_mb_peak"),
        "peak_gpu_memory_mb": peak("finetune", "gpu_mem_used_mb_peak")
        or peak("acquisition", "gpu_mem_used_mb_peak"),
        "cpu_util_pct_mean": mean("finetune", "cpu_pct_mean"),
        "gpu_util_pct_mean": mean("finetune", "gpu_util_pct_mean"),
        "gpu_power_w_mean": mean("finetune", "gpu_power_w_mean"),
        "gpu_energy_j_per_cycle": mean("finetune", "gpu_energy_j"),
        "finetune_s_per_cycle": (
            sum(update_times) / len(update_times) if update_times else None
        ),
        "finetune_s_total": sum(update_times) if update_times else None,
        "network_mb_total": network.get("total_mb"),
        "network_mb_per_cycle": network.get("mb_per_cycle"),
        "network_transfer_s_total": network.get("total_transfer_s"),
        "platform": result.get("platform", {}),
    }
