"""
Distribution Shift Adaptation Pipeline (H2).

Extends the baseline closed-loop pipeline to support controlled
covariate-shift experiments. The model is:
  1. Trained on the SOURCE domain (L0 + U source)
  2. Evaluated on the TARGET domain at every cycle (shift benchmark)
  3. Adapted on TARGET domain samples via the same AL strategies

This directly tests Hypothesis H2:
  "Approximate Bayesian uncertainty via MC Dropout produces more stable
   learning curves under covariate shift than deterministic methods."

Metrics collected per cycle on the TARGET test set:
  - mAP@50, precision, recall (performance)
  - Inter-cycle variance of recall (H2 stability criterion)
  - Learning curve monotonicity (H2 smoothness criterion)
  - All baseline metrics from the non-shift pipeline

Output JSON includes a 'shift_metrics' block compatible with the
baseline results format, enabling direct AUC/recall comparisons.
"""

import os
import sys
import json
import random
import shutil
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

from src.shift_dataset_manager import ShiftDatasetManager
from src.mc_dropout import (
    MCDropoutEstimator,
    enable_mc_dropout,
    disable_mc_dropout,
)
from src.acquisition import get_strategy
from src.trainer import ModelTrainer, UpdateTrigger
from src.metrics import trapezoid


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Per-run shift experiment
# ---------------------------------------------------------------------------

def run_shift_experiment(
    config: dict,
    source_dir: str,
    target_dir: str,            # pass None for simulated shift
    strategy: str,
    l0_size: int,
    run_id: int = 0,
    simulate_shift: bool = False,
    shift_intensity: str = "moderate",
    domain_fraction: float = 0.5,
    class_names: list = None,
) -> dict:
    """
    Run one complete shift experiment.

    Parameters
    ----------
    config : dict
        Experiment config (loaded from config.yaml).
    source_dir : str
        Path to source YOLO dataset root.
    target_dir : str or None
        Path to target YOLO dataset root.
        None → use simulated shift on source.
    strategy : str
        Acquisition strategy name.
    l0_size : int
        Initial labelled set size on the source domain.
    run_id : int
        Seed offset for this run (0..N_SEEDS-1).
    simulate_shift : bool
        Whether to apply synthetic shift augmentation.
    shift_intensity : str
        "none" | "mild" | "moderate" | "severe"
    domain_fraction : float
        Source/target split when simulating shift.
    class_names : list or None
        Class names to use; inferred if None.

    Returns
    -------
    dict  Full results dict compatible with the baseline pipeline output.
    """
    seed = config["experiment"]["seed"] + run_id
    set_seed(seed)

    device = config["experiment"].get("device", "cpu")
    output_dir = Path(config["experiment"]["output_dir"]) / "shift_experiments"
    exp_name = f"shift_{strategy}_L0{l0_size}_run{run_id}"
    exp_dir = output_dir / exp_name
    if exp_dir.exists():
        shutil.rmtree(exp_dir)
    exp_dir.mkdir(parents=True)

    work_dir = exp_dir / "work"
    work_dir.mkdir()

    print(f"\n{'='*70}")
    print(f"SHIFT EXPERIMENT: {exp_name}")
    print(f"  Source: {source_dir}")
    print(f"  Target: {target_dir or 'simulated'}")
    print(f"  Strategy: {strategy} | L0: {l0_size} | Seed: {seed}")
    print(f"{'='*70}")

    # ----------------------------------------------------------------
    # 1. Dataset setup
    # ----------------------------------------------------------------
    print("\n[1/6] Setting up shift dataset manager...")
    mgr = ShiftDatasetManager(
        source_root=source_dir,
        target_root=target_dir,
        class_names=class_names,
        simulate_shift=simulate_shift,
        domain_fraction=domain_fraction,
        shift_intensity=shift_intensity,
    )
    print(f"  {mgr.describe()}")

    source_splits = mgr.get_source_splits(
        l0_size=l0_size,
        test_fraction=config["dataset"].get("test_ratio", 0.2),
        seed=seed,
    )
    target_splits = mgr.get_target_splits(
        test_fraction=config["dataset"].get("test_ratio", 0.2),
        seed=seed,
    )

    # Build workspace directories
    source_yaml = mgr.prepare_source_workspace(source_splits, str(work_dir))
    target_yaml = mgr.prepare_target_workspace(target_splits, str(work_dir))

    # ----------------------------------------------------------------
    # 2. Initial training on SOURCE domain (L0)
    # ----------------------------------------------------------------
    print("\n[2/6] Initial training on SOURCE domain (L0)...")
    trainer = ModelTrainer(
        model_name=config["model"]["name"],
        img_size=config["model"]["input_size"],
        device=device,
    )
    weights_path = trainer.initial_training(
        yaml_path=source_yaml,
        epochs=20,
        batch_size=config["training"]["batch_size"],
        lr=config["training"]["initial_lr"],
        output_dir=str(exp_dir / "runs"),
    )

    # ----------------------------------------------------------------
    # 3. Baseline: evaluate on TARGET test set BEFORE any adaptation
    # ----------------------------------------------------------------
    print("\n[3/6] Baseline evaluation on TARGET domain (no adaptation)...")
    baseline_metrics = trainer.evaluate(target_yaml, split="val")
    print(
        f"  Baseline → mAP@50={baseline_metrics['mAP50']:.4f} | "
        f"P={baseline_metrics['precision']:.4f} | "
        f"R={baseline_metrics['recall']:.4f}"
    )

    # ----------------------------------------------------------------
    # 4. Setup uncertainty estimator, acquisition strategy, trigger
    # ----------------------------------------------------------------
    print("\n[4/6] Setting up AL components...")
    uc_estimator = MCDropoutEstimator(
        model_path=weights_path,
        T=config["uncertainty"]["T"],
        dropout_rate=config["uncertainty"]["dropout_rate"],
        conf_threshold=config["model"]["conf_threshold"],
        iou_threshold=config["model"]["iou_threshold"],
        img_size=config["model"]["input_size"],
        device=device,
    )

    acq_strategy = get_strategy(strategy, {
        "seed": seed,
        "alpha": config["uncertainty"]["alpha"],
        "beta": config["diversity"]["beta"],
    })

    trigger = UpdateTrigger(
        n_min=config["training"]["n_min"],
        window=config["training"]["shift_window"],
    )

    # Calibrate shift threshold from SOURCE L0
    if strategy in ("bald_only", "bald_diversity"):
        src_paths = [s["image"] for s in source_splits["L0"][:min(80, len(source_splits["L0"]))]]
        baseline_unc = uc_estimator.compute_mc_uncertainty(src_paths)
        trigger.calibrate_threshold(
            baseline_unc["variance"],
            percentile=config["training"].get("tau_shift_percentile", 90),
        )

    # ----------------------------------------------------------------
    # 5. Adaptation cycles on TARGET domain
    # ----------------------------------------------------------------
    num_cycles = config["active_learning"]["num_cycles"]
    budget = config["active_learning"]["budget_per_cycle"]

    target_u_pool = list(target_splits["U"])  # mutable list of sample dicts
    labelled_target = []
    cumulative_target_labels = 0

    results = {
        "experiment_id": exp_name,
        "strategy": strategy,
        "l0_size": l0_size,
        "run_id": run_id,
        "seed": seed,
        "shift_setup": {
            "source": str(source_dir),
            "target": str(target_dir) if target_dir else "simulated",
            "simulate_shift": simulate_shift,
            "shift_intensity": shift_intensity,
            "domain_fraction": domain_fraction,
        },
        "config": {
            "T": config["uncertainty"]["T"],
            "alpha": config["uncertainty"]["alpha"],
            "budget": budget,
            "num_cycles": num_cycles,
            "epochs_per_cycle": config["training"]["epochs_per_cycle"],
        },
        "baseline_target_metrics": baseline_metrics,
        "trajectory": [{
            "cycle": 0,
            "n_labelled_target": 0,
            "n_new": 0,
            **baseline_metrics,
            "update_time": 0.0,
            "trigger_reason": "initial",
        }],
    }

    print(f"\n[5/6] Adaptation cycles on TARGET domain ({num_cycles} cycles)...")
    print(f"  Budget/cycle: {budget} | Total budget: {budget * num_cycles}")

    for cycle in range(1, num_cycles + 1):
        print(f"\n  --- Cycle {cycle}/{num_cycles} ---")

        # Progressive release of target unlabelled pool
        batch = mgr.get_target_cycle_batch(target_u_pool, cycle - 1, num_cycles)
        if not batch:
            print("  [SKIP] Target pool exhausted.")
            metrics = trainer.evaluate(target_yaml, split="val")
            results["trajectory"].append({
                "cycle": cycle,
                "n_labelled_target": cumulative_target_labels,
                "n_new": 0,
                **metrics,
                "update_time": 0.0,
                "trigger_reason": "skipped",
            })
            continue

        batch_paths = [s["image"] for s in batch]
        print(f"  Stage 1 (pool release): {len(batch)} target samples available")

        # --- Stage 2: Uncertainty scoring ---
        print(f"  Stage 2 (uncertainty): computing scores ({strategy})...")
        n_total = len(mgr._target_samples or mgr._source_samples)

        bald_scores = None
        unc_scores = None
        det_scores = None
        features = None

        if strategy == "random":
            pass  # no scores needed

        elif strategy == "deterministic":
            preds = uc_estimator.compute_deterministic_predictions(batch_paths)
            det_scores = np.zeros(n_total)
            for i, pred in enumerate(preds):
                max_conf = pred["scores"].max() if len(pred["scores"]) > 0 else 0.0
                det_scores[i] = 1.0 - max_conf

        else:  # bald_only or bald_diversity
            unc_result = uc_estimator.compute_mc_uncertainty(batch_paths)
            unc_scores = np.zeros(n_total)
            bald_scores = np.zeros(n_total)
            for i in range(len(batch_paths)):
                unc_scores[i] = unc_result["variance"][i]
                bald_scores[i] = unc_result["bald"][i]
            trigger.record_batch_uncertainty(float(unc_result["variance"].mean()))

            if strategy == "bald_diversity":
                feats = uc_estimator.extract_features(batch_paths)
                features = np.zeros((n_total, feats.shape[1]))
                for i in range(min(len(batch_paths), len(feats))):
                    features[i] = feats[i]

        # --- Stage 3: AL selection ---
        batch_indices = list(range(len(batch_paths)))  # local indices within batch
        print(f"  Stage 3 (selection): selecting up to {budget} samples...")

        selected_local = acq_strategy.select(
            candidate_indices=batch_indices,
            budget=budget,
            det_scores=det_scores,
            bald_scores=bald_scores,
            uncertainty_scores=unc_scores,
            features=features,
        )

        selected_samples = [batch[i] for i in selected_local]
        print(f"  Selected: {len(selected_samples)} samples")

        # Simulate oracle annotation (copy ground-truth labels)
        mgr.add_target_samples(selected_samples, str(work_dir))
        labelled_target.extend(selected_samples)
        cumulative_target_labels += len(selected_samples)
        trigger.add_annotations(len(selected_samples))

        # Remove selected from pool
        selected_set = {s["image"] for s in selected_samples}
        target_u_pool = [s for s in target_u_pool if s["image"] not in selected_set]

        # --- Stage 4: Update trigger ---
        should_update, reason = trigger.should_update()

        if should_update or cycle == num_cycles:
            if not should_update:
                reason = "final_cycle"
            print(f"  Stage 4 (update): triggered ({reason})")
            weights_path, update_time = trainer.incremental_update(
                yaml_path=target_yaml,
                cycle=cycle,
                epochs=config["training"]["epochs_per_cycle"],
                batch_size=config["training"]["batch_size"],
                lr=config["training"]["incremental_lr"],
                output_dir=str(exp_dir / "runs" / "updates"),
            )
            # Refresh uncertainty estimator with new weights
            uc_estimator = MCDropoutEstimator(
                model_path=weights_path,
                T=config["uncertainty"]["T"],
                dropout_rate=config["uncertainty"]["dropout_rate"],
                conf_threshold=config["model"]["conf_threshold"],
                iou_threshold=config["model"]["iou_threshold"],
                img_size=config["model"]["input_size"],
                device=device,
            )
            trigger.reset_after_update()
        else:
            update_time = 0.0
            reason = "no_update"
            print(f"  Stage 4: no update (accumulated: {trigger.n_accumulated})")

        # Evaluate on TARGET test set
        metrics = trainer.evaluate(target_yaml, split="val")
        print(
            f"  → mAP@50={metrics['mAP50']:.4f} | "
            f"P={metrics['precision']:.4f} | R={metrics['recall']:.4f}"
        )

        results["trajectory"].append({
            "cycle": cycle,
            "n_labelled_target": cumulative_target_labels,
            "n_new": len(selected_samples),
            **metrics,
            "update_time": update_time,
            "trigger_reason": reason,
        })

    # ----------------------------------------------------------------
    # 6. Post-experiment statistics
    # ----------------------------------------------------------------
    print("\n[6/6] Computing shift statistics...")

    traj = results["trajectory"]
    recalls = [t["recall"] for t in traj]
    map50s = [t["mAP50"] for t in traj]
    n_labels = [t["n_labelled_target"] for t in traj]

    # AUC (trapezoid over cumulative target labels), plus the
    # budget-normalised form the examination board asked for.
    auc = trapezoid(map50s, n_labels)
    span = float(n_labels[-1] - n_labels[0]) if len(n_labels) > 1 else 0.0
    auc_norm = float(auc / span) if span > 0 else 0.0

    # Recall stability (H2 primary criterion)
    recall_diffs = [recalls[i+1] - recalls[i] for i in range(len(recalls)-1)]
    recall_variance = float(np.var(recalls))
    recall_std = float(np.std(recalls))
    monotonicity = (
        float(sum(1 for d in recall_diffs if d >= 0) / len(recall_diffs))
        if recall_diffs else 0.0
    )

    # Latency (final model)
    latency_ms = trainer.measure_latency(
        [s["image"] for s in target_splits["T"][:100]]
    )

    results["summary"] = {
        "final_mAP50":    float(map50s[-1]),
        "final_precision": float(traj[-1]["precision"]),
        "final_recall":    float(recalls[-1]),
        "auc":             auc,
        "auc_normalized":  auc_norm,
        "recall_variance": recall_variance,
        "recall_std":      recall_std,
        "recall_mean":     float(np.mean(recalls)),
        "monotonicity":    monotonicity,
        "latency_ms":      latency_ms,
        "total_labelled":  cumulative_target_labels,
        "recall_drop_from_baseline": (
            float(baseline_metrics["recall"]) - float(recalls[-1])
        ),
        "mAP_drop_from_baseline": (
            float(baseline_metrics["mAP50"]) - float(map50s[-1])
        ),
    }

    print(f"\n  {'='*55}")
    print(f"  SHIFT RESULTS: {exp_name}")
    print(f"  {'='*55}")
    print(f"  Baseline recall (no adapt.): {baseline_metrics['recall']:.4f}")
    print(f"  Final recall (after adapt.): {recalls[-1]:.4f}")
    print(f"  Recall variance (stability): {recall_variance:.6f}")
    print(f"  Monotonicity:               {monotonicity:.2f}")
    print(f"  AUC:                        {auc:.4f}")
    print(f"  Mean Latency:               {latency_ms:.1f} ms")
    print(f"  {'='*55}")

    # Save results
    results_path = exp_dir / "shift_results.json"
    _save_json(results, results_path)
    print(f"  Results saved to: {results_path}")

    return results


# ---------------------------------------------------------------------------
# Multi-seed shift runner
# ---------------------------------------------------------------------------

def run_shift_with_seeds(
    config: dict,
    source_dir: str,
    target_dir: str,
    strategy: str,
    l0_size: int,
    n_seeds: int = 5,
    seeds: list = None,
    simulate_shift: bool = False,
    shift_intensity: str = "moderate",
    domain_fraction: float = 0.5,
    class_names: list = None,
) -> dict:
    """
    Run a shift experiment with multiple seeds and aggregate statistics.

    Returns
    -------
    dict with per-seed results + aggregated mean/std summary.
    """
    if seeds is None:
        seeds = list(range(n_seeds))

    all_runs = []
    for run_id, seed_offset in enumerate(seeds):
        # Override seed in config for this run
        cfg = dict(config)
        cfg["experiment"] = dict(config["experiment"])
        cfg["experiment"]["seed"] = config["experiment"].get("seed", 42) + seed_offset

        result = run_shift_experiment(
            config=cfg,
            source_dir=source_dir,
            target_dir=target_dir,
            strategy=strategy,
            l0_size=l0_size,
            run_id=run_id,
            simulate_shift=simulate_shift,
            shift_intensity=shift_intensity,
            domain_fraction=domain_fraction,
            class_names=class_names,
        )
        all_runs.append(result)

    return _aggregate_shift_runs(all_runs, strategy, l0_size)


def _aggregate_shift_runs(runs: list, strategy: str, l0_size: int) -> dict:
    """Aggregate N seed results into mean ± std statistics.

    Runs that crashed before writing a summary are dropped and counted
    rather than raising: a single failed seed used to abort the whole
    aggregation and leave an empty ``all_shift_aggregated.json``, which is
    how the previous campaign ended up with per-strategy files on disk but
    an empty combined table.
    """
    valid = [r for r in runs if isinstance(r, dict) and "summary" in r]
    n_failed = len(runs) - len(valid)
    if not valid:
        return {
            "strategy": strategy,
            "l0_size": l0_size,
            "n_seeds": 0,
            "n_failed": n_failed,
            "error": "every seed failed; no summary produced",
        }
    runs = valid
    summaries = [r["summary"] for r in runs]

    keys = ["final_mAP50", "final_precision", "final_recall",
            "auc", "auc_normalized", "recall_variance", "recall_std",
            "monotonicity", "latency_ms", "recall_drop_from_baseline"]

    agg = {
        "strategy": strategy,
        "l0_size": l0_size,
        "n_seeds": len(runs),
        "n_failed": n_failed,
        "shift_setup": runs[0].get("shift_setup", {}),
        "per_run_summaries": summaries,
    }

    for k in keys:
        vals = [s[k] for s in summaries if k in s]
        if vals:
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))

    # Recall trajectory: mean ± std per cycle
    max_cycles = max(len(r["trajectory"]) for r in runs)
    recall_matrix = np.full((len(runs), max_cycles), np.nan)
    mAP_matrix = np.full((len(runs), max_cycles), np.nan)

    for i, r in enumerate(runs):
        for j, step in enumerate(r["trajectory"]):
            recall_matrix[i, j] = step.get("recall", np.nan)
            mAP_matrix[i, j] = step.get("mAP50", np.nan)

    agg["trajectory_recall_mean"] = np.nanmean(recall_matrix, axis=0).tolist()
    agg["trajectory_recall_std"] = np.nanstd(recall_matrix, axis=0).tolist()
    agg["trajectory_mAP50_mean"] = np.nanmean(mAP_matrix, axis=0).tolist()
    agg["trajectory_mAP50_std"] = np.nanstd(mAP_matrix, axis=0).tolist()

    return agg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=convert)
