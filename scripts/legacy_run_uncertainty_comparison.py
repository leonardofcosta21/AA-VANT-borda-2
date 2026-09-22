"""
run_uncertainty_comparison.py
==============================
Compara os 5 métodos de quantificação de incerteza mantendo
a mesma estratégia de seleção (BALD+Diversity) e pipeline idêntico:

  1. MC Dropout (BALD)        — método proposto na tese
  2. Conformal Prediction     — alternativa black-box (Seção 9.1)
  3. SWAG                     — alternativa Bayesiana (Seção 9.1)
  4. BSB                      — critério de margem (Seção 9.2)
  5. PSB                      — critério de margem com histórico (Seção 9.2)

Todos usam BALD+Diversity como estratégia de seleção (exceto BSB/PSB
que usam BSB+Diversity) para isolar o efeito do estimador de incerteza.

Uso:
    python run_uncertainty_comparison.py \\
        --data_root datasets/_prepared/SARD \\
        --n_seeds 5 --l0_size 100 \\
        --output_dir results/uncertainty_comparison \\
        --device 0

    # Apenas métodos específicos:
    python run_uncertainty_comparison.py \\
        --data_root datasets/_prepared/SARD \\
        --methods mc_dropout bsb conformal \\
        --n_seeds 3 --l0_size 100 \\
        --output_dir results/uncertainty_comparison \\
        --device 0

    # Rápido (1 seed, para testar):
    python run_uncertainty_comparison.py \\
        --data_root datasets/_prepared/SARD \\
        --n_seeds 1 --l0_size 100 --cycles 5 --T 4 \\
        --output_dir results/uncertainty_comparison_fast \\
        --device 0

Saída:
    results/uncertainty_comparison/
      per_run/
        {method}_L0{l0}_run{k}/results.json
      aggregated/
        {method}_L0{l0}_agg.json
      all_methods_aggregated.json
      uncertainty_comparison_table.csv    ← tabela para a tese
      uncertainty_comparison_table.tex    ← LaTeX pronto
"""

import os
import sys
import json
import random
import shutil
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import ExperimentConfig
from src.pipeline import run_experiment, set_seed
from src.dataset_utils import discover_dataset, partition_dataset, prepare_yolo_split, write_data_yaml
from src.trainer import ModelTrainer as IncrementalTrainer
from src.trainer import ModelTrainer
from src.evaluator import Evaluator

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

ALL_METHODS = ["mc_dropout", "conformal", "swag", "bsb", "psb"]


# ---------------------------------------------------------------------------
# Per-method experiment runner
# ---------------------------------------------------------------------------

def run_one_method(
    method: str,
    data_root: str,
    l0_size: int,
    config: ExperimentConfig,
    output_dir: Path,
    run_id: int = 0,
) -> dict:
    """
    Run one complete experiment with a specific uncertainty estimator.

    Uses the same pipeline as run_experiment() but swaps out the
    uncertainty estimator based on `method`.

    mc_dropout / conformal / swag → strategy = "bald_diversity"
    bsb / psb                     → strategy = "bsb_diversity"
    """
    seed = config.seed + run_id
    set_seed(seed)

    device = config.device
    exp_name = f"{method}_L0{l0_size}_run{run_id}"
    exp_dir  = output_dir / "per_run" / exp_name
    if exp_dir.exists():
        shutil.rmtree(exp_dir)
    exp_dir.mkdir(parents=True)
    work_dir = exp_dir / "work"
    work_dir.mkdir()

    print(f"\n{'='*65}")
    print(f"  METHOD: {method} | L0={l0_size} | seed={seed} (run {run_id})")
    print(f"{'='*65}")

    # ------------------------------------------------------------------
    # 1. Dataset setup
    # ------------------------------------------------------------------
    samples, class_names = discover_dataset(data_root)
    if not class_names:
        class_names = ["person", "vehicle", "hazard"]

    L0, U, T = partition_dataset(
        samples, l0_size,
        test_ratio=config.dataset.test_fraction,
        seed=seed,
    )
    print(f"  L0={len(L0)} | U={len(U)} | T={len(T)}")

    data_dir  = work_dir / "data"
    prepare_yolo_split(L0, data_dir, "train")
    prepare_yolo_split(T,  data_dir, "val")
    data_yaml = write_data_yaml(
        data_dir,
        data_dir / "images" / "train",
        data_dir / "images" / "val",
        class_names,
    )

    # ------------------------------------------------------------------
    # 2. Initial training (identical for all methods)
    # ------------------------------------------------------------------
    print("  [1/4] Initial training on L0 (20 epochs)...")
    model = YOLO(config.model.model_name)
    model.train(
        data=str(data_yaml),
        epochs=20,
        batch=config.training.batch_size,
        imgsz=config.model.img_size,
        lr0=config.training.initial_lr,
        project=str(work_dir / "initial_train"),
        name="initial",
        exist_ok=True,
        verbose=False,
        plots=False,
        device=device,
    )
    # Ultralytics may save to runs/detect/ regardless of project= setting.
    # Search recursively from work_dir first, then from cwd.
    def _find_weights(search_dirs, name="best.pt"):
        for base in search_dirs:
            base = Path(base)
            if not base.exists():
                continue
            candidates = sorted(base.rglob(name), key=lambda p: p.stat().st_mtime)
            if candidates:
                return str(candidates[-1])
        return None

    weights_path = (
        _find_weights([work_dir / "initial_train"], "best.pt") or
        _find_weights([work_dir / "initial_train"], "last.pt") or
        _find_weights([Path("runs")], "best.pt") or
        _find_weights([Path("runs")], "last.pt")
    )
    if weights_path is None:
        raise FileNotFoundError(
            f"Could not find trained weights. "
            f"Searched in {work_dir / 'initial_train'} and runs/"
        )
    print(f"  Weights found: {weights_path}")
    model = YOLO(weights_path)

    # ------------------------------------------------------------------
    # 3. Build uncertainty estimator for this method
    # ------------------------------------------------------------------
    print(f"  [2/4] Building uncertainty estimator: {method}...")
    uc_estimator = _build_estimator(
        method, weights_path, config, L0, data_yaml
    )

    # ------------------------------------------------------------------
    # 4. Adaptation cycles
    # ------------------------------------------------------------------
    trainer = ModelTrainer(
        model_name=config.model.model_name,
        img_size=config.model.img_size,
        device=device,
    )
    from src.trainer import UpdateTrigger
    trigger = UpdateTrigger(
        n_min=config.training.n_min,
        window=config.training.shift_window,
    )
    evaluator = Evaluator(data_yaml, device=device)

    # Cycle 0 baseline
    metrics_0 = evaluator.evaluate(model, cycle_idx=0, cumulative_labels=len(L0))
    print(f"  Cycle 0: mAP50={metrics_0['mAP50']:.4f} | "
          f"R={metrics_0['recall']:.4f} | "
          f"lat={metrics_0['latency_ms']:.1f}ms")

    num_cycles = config.active_learning.num_cycles
    budget     = config.active_learning.budget_per_cycle
    cumulative_labels = len(L0)

    # PSB needs second-best history across cycles
    psb_strategy = None
    if method == "psb":
        from src.uncertainty_extended import PSBSampling
        psb_strategy = PSBSampling(alpha=config.uncertainty.alpha)

    print(f"  [3/4] {num_cycles} adaptation cycles (budget={budget})...")

    for cycle in range(1, num_cycles + 1):
        print(f"    Cycle {cycle}/{num_cycles}", end=" ", flush=True)

        # Progressive pool release
        from src.dataset_utils import create_cycle_pool
        batch = create_cycle_pool(U, cycle - 1, num_cycles)
        batch_paths = [s["image"] for s in batch]

        if not batch_paths:
            metrics = evaluator.evaluate(
                model, cycle_idx=cycle, cumulative_labels=cumulative_labels
            )
            print(f"→ pool empty")
            continue

        # Compute uncertainty with the chosen estimator
        unc_result = uc_estimator.compute_mc_uncertainty(batch_paths)
        n_total    = len(batch_paths)

        # Select samples based on method
        selected_paths = _select_samples(
            method        = method,
            batch         = batch,
            batch_paths   = batch_paths,
            unc_result    = unc_result,
            budget        = budget,
            config        = config,
            uc_estimator  = uc_estimator,
            psb_strategy  = psb_strategy,
        )

        # Simulate oracle annotation
        selected_set = set(selected_paths)
        newly_labelled = [s for s in batch if s["image"] in selected_set]
        cumulative_labels += len(newly_labelled)
        trigger.add_annotations(len(newly_labelled))

        # Update PSB history
        if method == "psb" and psb_strategy is not None:
            sb = unc_result.get("second_best")
            if sb is not None:
                psb_strategy.update_history(sb)

        # Add to training set
        prepare_yolo_split(newly_labelled, data_dir, "train")
        trigger.record_batch_uncertainty(float(unc_result["variance"].mean()))

        # Incremental update trigger
        triggered, reasons = trigger.should_update()
        if triggered or cycle == num_cycles:
            upd_dir = str(work_dir / "incremental_runs")
            weights_path = trainer.incremental_update(
                yaml_path  = str(data_yaml),
                cycle      = cycle,
                epochs     = config.training.epochs_per_cycle,
                batch_size = config.training.batch_size,
                lr         = config.training.incremental_lr,
                output_dir = upd_dir,
            )
            trigger.reset_after_update()
            if weights_path:
                model = YOLO(weights_path)
                trainer.current_weights_path = weights_path
                # Rebuild estimator with new weights
                uc_estimator = _build_estimator(
                    method, weights_path, config, L0, data_yaml
                )

        metrics = evaluator.evaluate(
            model, cycle_idx=cycle, cumulative_labels=cumulative_labels
        )
        print(f"→ mAP50={metrics['mAP50']:.4f} R={metrics['recall']:.4f}")

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    summary = evaluator.get_summary()
    summary["method"]  = method
    summary["l0_size"] = l0_size
    summary["seed"]    = seed
    summary["run_id"]  = run_id

    results_path = evaluator.save_results(exp_dir / "results.json")
    print(f"  Saved: {results_path}")
    return summary


# ---------------------------------------------------------------------------
# Estimator factory
# ---------------------------------------------------------------------------

def _build_estimator(method, weights_path, config, L0, data_yaml):
    """Build the appropriate uncertainty estimator for the given method."""
    from src.mc_dropout import MCDropoutEstimator

    base_kwargs = dict(
        model_path      = weights_path,
        conf_threshold  = config.model.conf_threshold,
        iou_threshold   = config.model.iou_threshold,
        img_size        = config.model.img_size,
        device          = config.device,
    )

    if method == "mc_dropout":
        return MCDropoutEstimator(
            T            = config.uncertainty.T,
            dropout_rate = config.uncertainty.dropout_rate,
            **base_kwargs,
        )

    elif method == "conformal":
        from src.uncertainty_extended import ConformalUncertaintyEstimator
        base = MCDropoutEstimator(T=1, dropout_rate=0.0, **base_kwargs)
        cp   = ConformalUncertaintyEstimator(
            base, epsilon=getattr(config.uncertainty, "epsilon", 0.1)
        )
        # Calibrate on L0 (use up to 200 samples)
        cal_paths = [s["image"] for s in L0[:min(200, len(L0))]]
        cp.calibrate(cal_paths)
        return cp

    elif method == "swag":
        from src.uncertainty_extended import SWAGEstimator
        swag = SWAGEstimator(
            K      = getattr(config.uncertainty, "swag_K", 10),
            S      = getattr(config.uncertainty, "swag_S", 5),
            rank   = getattr(config.uncertainty, "swag_rank", 5),
            **base_kwargs,
        )
        # Collect SWAG statistics with short fine-tuning
        swag.collect_swag_statistics(
            yaml_path = str(data_yaml),
            epochs    = max(config.uncertainty.T, 10),
            lr        = config.training.incremental_lr,
        )
        return swag

    elif method in ("bsb", "psb"):
        from src.uncertainty_extended import BSBScorer
        return BSBScorer(
            conf_threshold = 0.05,   # low threshold to see near-miss detections
            iou_threshold  = base_kwargs["iou_threshold"],
            img_size       = base_kwargs["img_size"],
            device         = base_kwargs["device"],
            model_path     = weights_path,
        )

    else:
        raise ValueError(f"Unknown method: {method}")


def _select_samples(
    method, batch, batch_paths, unc_result, budget,
    config, uc_estimator, psb_strategy,
):
    """Select samples using the appropriate strategy for each method."""
    budget = min(budget, len(batch_paths))

    if method in ("mc_dropout", "conformal", "swag"):
        # Use BALD+Diversity with the uncertainty scores from the estimator
        from src.uncertainty_extended import BALDDiversitySampling
        strategy = BALDDiversitySampling(
            alpha = config.uncertainty.alpha,
            beta  = config.active_learning.beta,
            seed  = config.seed,
        )
        n = len(batch_paths)
        bald_scores = unc_result.get("bald",     np.zeros(n))
        unc_scores  = unc_result.get("variance", np.zeros(n))

        # Try to get features for diversity
        features = None
        try:
            features_raw = uc_estimator.extract_features(batch_paths)
            features = np.zeros((n, features_raw.shape[1]))
            features[:len(features_raw)] = features_raw
        except Exception:
            pass

        selected_idx = strategy.select(
            candidate_indices  = list(range(n)),
            budget             = budget,
            bald_scores        = bald_scores,
            uncertainty_scores = unc_scores,
            features           = features,
        )
        return [batch_paths[i] for i in selected_idx]

    elif method == "bsb":
        from src.uncertainty_extended import BSBDiversitySampling
        strategy = BSBDiversitySampling(
            alpha = config.uncertainty.alpha,
            beta  = config.active_learning.beta,
            seed  = config.seed,
        )
        n          = len(batch_paths)
        bsb_scores = unc_result.get("bsb_scores", unc_result.get("variance", np.zeros(n)))

        features = None
        try:
            features_raw = uc_estimator.extract_features(batch_paths)
            features = np.zeros((n, features_raw.shape[1]))
            features[:len(features_raw)] = features_raw
        except Exception:
            pass

        selected_idx = strategy.select(
            candidate_indices = list(range(n)),
            budget            = budget,
            bsb_scores        = bsb_scores,
            features          = features,
        )
        return [batch_paths[i] for i in selected_idx]

    elif method == "psb":
        n           = len(batch_paths)
        bsb_scores  = unc_result.get("bsb_scores",  unc_result.get("variance", np.zeros(n)))
        second_best = unc_result.get("second_best", np.zeros(n))

        if psb_strategy is not None:
            selected_idx = psb_strategy.select(
                candidate_indices  = list(range(n)),
                budget             = budget,
                bsb_scores         = bsb_scores,
                second_best_scores = second_best,
            )
        else:
            selected_idx = np.argsort(bsb_scores)[::-1][:budget].tolist()

        return [batch_paths[i] for i in selected_idx]

    else:
        return random.sample(batch_paths, budget)


def _find_best_weights(runs_dir: Path) -> str:
    """Find the most recent best.pt in incremental runs."""
    candidates = sorted(runs_dir.rglob("best.pt"), key=lambda p: p.stat().st_mtime)
    return str(candidates[-1]) if candidates else None


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------

def aggregate_method_runs(runs: list, method: str, l0_size: int) -> dict:
    """Aggregate N seed runs for one method."""
    keys = ["final_mAP50", "final_recall", "final_precision",
            "auc", "mean_latency_ms"]
    agg  = {"method": method, "l0_size": l0_size, "n_runs": len(runs), "runs": runs}

    for k in keys:
        vals = [r[k] for r in runs if k in r]
        if vals:
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"]  = float(np.std(vals))

    # Recall variance (stability) per run
    recall_vars = []
    for r in runs:
        traj = r.get("trajectory", [])
        if len(traj) > 1:
            recalls = [t.get("recall", np.nan) for t in traj]
            recall_vars.append(float(np.nanvar(recalls)))
    if recall_vars:
        agg["recall_variance_mean"] = float(np.mean(recall_vars))
        agg["recall_variance_std"]  = float(np.std(recall_vars))

    # Monotonicity
    monots = []
    for r in runs:
        traj = r.get("trajectory", [])
        if len(traj) > 1:
            m50s = [t.get("mAP50", np.nan) for t in traj]
            diffs = np.diff([x for x in m50s if not np.isnan(x)])
            if len(diffs):
                monots.append(float((diffs >= 0).mean()))
    if monots:
        agg["monotonicity_mean"] = float(np.mean(monots))
        agg["monotonicity_std"]  = float(np.std(monots))

    return agg


# ---------------------------------------------------------------------------
# Output tables
# ---------------------------------------------------------------------------

def build_comparison_table(all_agg: list) -> pd.DataFrame:
    rows = []
    for a in all_agg:
        rows.append({
            "method":                a.get("method"),
            "n_seeds":               a.get("n_runs"),
            "AUC_mean":              a.get("auc_mean",              np.nan),
            "AUC_std":               a.get("auc_std",               np.nan),
            "final_mAP50_mean":      a.get("final_mAP50_mean",      np.nan),
            "final_mAP50_std":       a.get("final_mAP50_std",       np.nan),
            "final_recall_mean":     a.get("final_recall_mean",     np.nan),
            "final_recall_std":      a.get("final_recall_std",      np.nan),
            "recall_variance_mean":  a.get("recall_variance_mean",  np.nan),
            "monotonicity_mean":     a.get("monotonicity_mean",     np.nan),
            "latency_mean_ms":       a.get("mean_latency_ms_mean",  np.nan),
        })
    return pd.DataFrame(rows)


def build_latex_table(all_agg: list) -> str:
    METHOD_LABELS = {
        "mc_dropout": "MC Dropout (BALD) — proposed",
        "conformal":  "Conformal Prediction",
        "swag":       "SWAG",
        "bsb":        "BSB",
        "psb":        "PSB",
    }
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Uncertainty estimator comparison (mean $\pm$ std, $N$ seeds). "
        r"All methods use the same selection strategy (BALD+Diversity / BSB+Diversity). "
        r"AUC: area under mAP@50 learning curve. "
        r"Recall Var.: inter-cycle variance (lower = more stable).}",
        r"\label{tab:uncertainty_comparison}",
        r"\begin{tabular}{l rr rr rr r}",
        r"\toprule",
        r"Method & AUC $\uparrow$ & $\pm$ & Recall $\uparrow$ & $\pm$ "
        r"& mAP@50 $\uparrow$ & $\pm$ & Recall Var. $\downarrow$ \\",
        r"\midrule",
    ]

    order = ["mc_dropout", "conformal", "swag", "bsb", "psb"]
    for method in order:
        match = [a for a in all_agg if a.get("method") == method]
        if not match:
            continue
        a   = match[0]
        lbl = METHOD_LABELS.get(method, method)
        if method == "mc_dropout":
            lbl = r"\textbf{" + lbl + r"}"
        auc_m  = a.get("auc_mean",             0.0)
        auc_s  = a.get("auc_std",              0.0)
        rec_m  = a.get("final_recall_mean",    0.0)
        rec_s  = a.get("final_recall_std",     0.0)
        map_m  = a.get("final_mAP50_mean",     0.0)
        map_s  = a.get("final_mAP50_std",      0.0)
        var_m  = a.get("recall_variance_mean", 0.0)
        lines.append(
            f"  {lbl} "
            f"& {auc_m:.2f} & {auc_s:.2f} "
            f"& {rec_m:.4f} & {rec_s:.4f} "
            f"& {map_m:.4f} & {map_s:.4f} "
            f"& {var_m:.6f} \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def conv(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray):    return o.tolist()
        return o
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=conv)


def _config_to_dict(config: ExperimentConfig) -> dict:
    return {
        "experiment": {"seed": config.seed, "device": config.device,
                       "output_dir": "./results"},
        "model":      {"name": config.model.model_name,
                       "input_size": config.model.img_size,
                       "conf_threshold": config.model.conf_threshold,
                       "iou_threshold": config.model.iou_threshold},
        "uncertainty":{"T": config.uncertainty.T,
                       "dropout_rate": config.uncertainty.dropout_rate,
                       "alpha": config.uncertainty.alpha},
        "active_learning": {"budget_per_cycle": config.active_learning.budget_per_cycle,
                            "num_cycles": config.active_learning.num_cycles},
        "diversity":  {"beta": config.active_learning.beta},
        "training":   {"batch_size": config.training.batch_size,
                       "initial_lr": config.training.initial_lr,
                       "incremental_lr": config.training.incremental_lr,
                       "n_min": config.training.n_min,
                       "shift_window": config.training.shift_window,
                       "tau_shift_percentile": config.training.tau_shift_percentile,
                       "epochs_per_cycle": config.training.epochs_per_cycle},
        "dataset":    {"test_ratio": config.dataset.test_fraction},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare uncertainty estimators: BALD vs Conformal vs SWAG vs BSB vs PSB"
    )
    parser.add_argument("--data_root",  required=True)
    parser.add_argument("--output_dir", default="./results/uncertainty_comparison")
    parser.add_argument("--methods",    nargs="+", default=ALL_METHODS,
                        choices=ALL_METHODS,
                        help="Methods to compare (default: all 5)")
    parser.add_argument("--n_seeds",    type=int,   default=5)
    parser.add_argument("--base_seed",  type=int,   default=42)
    parser.add_argument("--l0_size",    type=int,   default=100)
    parser.add_argument("--device",     default="auto")
    parser.add_argument("--T",          type=int,   default=8,
                        help="MC Dropout passes / SWAG K epochs")
    parser.add_argument("--cycles",     type=int,   default=8)
    parser.add_argument("--budget",     type=int,   default=50)
    parser.add_argument("--swag_S",     type=int,   default=5,
                        help="SWAG inference samples (default 5, use 10 for more accuracy)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ExperimentConfig()
    config.device = args.device
    config.uncertainty.T = args.T
    config.active_learning.num_cycles = args.cycles
    config.active_learning.budget_per_cycle = args.budget
    config.uncertainty.swag_K = args.T
    config.uncertainty.swag_S = args.swag_S

    all_agg  = []
    total    = len(args.methods) * args.n_seeds
    current  = 0

    for method in args.methods:
        runs = []
        for run_id in range(args.n_seeds):
            current += 1
            print(f"\n[{current}/{total}] {method} | run {run_id+1}/{args.n_seeds}")

            cfg = ExperimentConfig()
            cfg.device = config.device
            cfg.seed   = args.base_seed + run_id
            cfg.uncertainty.T            = config.uncertainty.T
            cfg.uncertainty.swag_K       = config.uncertainty.swag_K
            cfg.uncertainty.swag_S       = config.uncertainty.swag_S
            cfg.active_learning.num_cycles       = config.active_learning.num_cycles
            cfg.active_learning.budget_per_cycle = config.active_learning.budget_per_cycle
            cfg.training.batch_size              = config.training.batch_size
            cfg.training.epochs_per_cycle        = config.training.epochs_per_cycle

            try:
                result = run_one_method(
                    method     = method,
                    data_root  = args.data_root,
                    l0_size    = args.l0_size,
                    config     = cfg,
                    output_dir = output_dir,
                    run_id     = run_id,
                )
                runs.append(result)
            except Exception as e:
                print(f"  [ERROR] {method} run {run_id}: {e}")
                import traceback; traceback.print_exc()

        if runs:
            agg = aggregate_method_runs(runs, method, args.l0_size)
            all_agg.append(agg)
            _save_json(agg, output_dir / "aggregated" / f"{method}_L0{args.l0_size}_agg.json")

    # Save all
    _save_json(all_agg, output_dir / "all_methods_aggregated.json")

    # CSV table
    df = build_comparison_table(all_agg)
    df.to_csv(output_dir / "uncertainty_comparison_table.csv", index=False, float_format="%.4f")

    # LaTeX table
    latex = build_latex_table(all_agg)
    (output_dir / "uncertainty_comparison_table.tex").write_text(latex)

    # Print summary
    print(f"\n{'='*80}")
    print("UNCERTAINTY ESTIMATOR COMPARISON")
    print(f"{'='*80}")
    pd.set_option("display.float_format", "{:.4f}".format)
    print(df.to_string(index=False))
    print(f"{'='*80}")
    print(f"\nTabela CSV: {output_dir / 'uncertainty_comparison_table.csv'}")
    print(f"Tabela LaTeX: {output_dir / 'uncertainty_comparison_table.tex'}")


if __name__ == "__main__":
    main()
