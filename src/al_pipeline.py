"""
Main Experiment Runner.
Implements the complete closed-loop adaptive cycle from Chapter 3,
evaluated according to the protocol in Chapter 4.

Usage:
    python run_experiment.py --data_root ./datasets/SARD --l0_size 200 --strategy bald_diversity
    python run_experiment.py --data_root ./datasets/SARD --run_all
"""

import os, sys, json, time, shutil, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import ExperimentConfig
from src.dataset_manager import DatasetManager
from src.mc_dropout import MCDropoutEstimator
from src.acquisition import get_strategy
from src.trainer import ModelTrainer, UpdateTrigger


def set_seed(seed):
    import random, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_single_experiment(
    data_root: str,
    strategy_name: str,
    l0_size: int,
    config: ExperimentConfig,
    output_dir: str,
    class_names: list = None,
):
    """
    Run one complete experiment: initial training + K adaptation cycles.

    Returns dict with full results trajectory.
    """
    exp_id = f"{strategy_name}_L0{l0_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = Path(output_dir) / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    data_dir = exp_dir / "data"

    print(f"\n{'#'*70}")
    print(f"# Experiment: {exp_id}")
    print(f"# Strategy: {strategy_name}, L0: {l0_size}")
    print(f"{'#'*70}\n")

    # ── Step 1: Dataset preparation ──
    dm = DatasetManager(data_root, img_size=config.dataset.img_size, seed=config.seed)
    splits = dm.create_splits(l0_size=l0_size, test_fraction=config.dataset.test_fraction)
    yaml_path = dm.prepare_yolo_dataset(splits, str(data_dir), class_names=class_names)

    # Track which indices are in training
    labelled_indices = list(splits["L0"])
    u_pool = list(splits["U"])

    # ── Step 2: Initial training on L0 ──
    trainer = ModelTrainer(
        model_name=config.model.model_name,
        img_size=config.model.img_size,
        device=config.device
    )
    weights_path = trainer.initial_training(
        yaml_path=yaml_path,
        epochs=30,
        batch_size=config.training.batch_size,
        lr=config.training.initial_lr,
        output_dir=str(exp_dir / "runs")
    )

    # ── Step 3: Evaluate initial model ──
    metrics_0 = trainer.evaluate(yaml_path)
    print(f"\n[Cycle 0] mAP50={metrics_0['mAP50']:.4f}, "
          f"P={metrics_0['precision']:.4f}, R={metrics_0['recall']:.4f}")

    # Results trajectory
    results = {
        "experiment_id": exp_id,
        "strategy": strategy_name,
        "l0_size": l0_size,
        "config": {
            "T": config.uncertainty.T,
            "alpha": config.uncertainty.alpha,
            "budget": config.active_learning.budget_per_cycle,
            "epochs_per_cycle": config.training.epochs_per_cycle,
            "n_min": config.training.n_min,
        },
        "trajectory": [{
            "cycle": 0,
            "n_labelled": len(labelled_indices),
            "n_new": 0,
            **metrics_0,
            "update_time": 0,
            "trigger_reason": "initial",
        }],
    }

    # ── Step 4: Setup uncertainty estimator and acquisition strategy ──
    uc_estimator = MCDropoutEstimator(
        model_path=weights_path,
        T=config.uncertainty.T,
        dropout_rate=config.uncertainty.dropout_rate,
        conf_threshold=config.model.conf_threshold,
        iou_threshold=config.model.iou_threshold,
        img_size=config.model.img_size,
        device=config.device,
    )

    strategy = get_strategy(strategy_name, {
        "seed": config.seed,
        "alpha": config.uncertainty.alpha,
        "beta": config.active_learning.beta,
    })

    trigger = UpdateTrigger(
        n_min=config.training.n_min,
        window=config.training.shift_window,
    )

    # Calibrate shift threshold from L0 baseline
    if strategy_name in ("bald_only", "bald_diversity"):
        l0_paths = dm.get_image_paths_for_indices(labelled_indices[:min(100, len(labelled_indices))])
        baseline_unc = uc_estimator.compute_mc_uncertainty(l0_paths)
        trigger.calibrate_threshold(
            baseline_unc["variance"],
            percentile=config.training.tau_shift_percentile
        )

    # ── Step 5: Adaptation cycles ──
    K = config.active_learning.num_cycles
    budget = config.active_learning.budget_per_cycle

    for cycle in range(1, K + 1):
        print(f"\n{'─'*60}")
        print(f"[Cycle {cycle}/{K}] Strategy: {strategy_name}")
        print(f"{'─'*60}")

        # 5a: Get batch of unlabelled samples for this cycle
        batch_indices = dm.get_cycle_batch(u_pool, cycle - 1, K)
        if not batch_indices:
            print(f"  No more unlabelled samples. Stopping.")
            break

        batch_paths = dm.get_image_paths_for_indices(batch_indices)
        print(f"  Batch size: {len(batch_indices)}")

        # 5b: Compute uncertainty scores (if needed)
        unc_scores = None
        bald_scores = None
        features = None
        det_scores = None

        # Create full-size arrays indexed by global position
        n_total = len(dm.image_paths)

        if strategy_name == "random":
            # No scores needed
            pass

        elif strategy_name == "deterministic":
            det_preds = uc_estimator.compute_deterministic_predictions(batch_paths)
            det_scores = np.zeros(n_total)
            for i, (idx, pred) in enumerate(zip(batch_indices, det_preds)):
                max_conf = pred["scores"].max() if len(pred["scores"]) > 0 else 0.0
                det_scores[idx] = 1.0 - max_conf

        else:  # bald_only or bald_diversity
            unc_result = uc_estimator.compute_mc_uncertainty(batch_paths)
            unc_scores = np.zeros(n_total)
            bald_scores = np.zeros(n_total)
            for i, idx in enumerate(batch_indices):
                unc_scores[idx] = unc_result["variance"][i]
                bald_scores[idx] = unc_result["bald"][i]

            # Record batch uncertainty for trigger
            trigger.record_batch_uncertainty(unc_result["variance"].mean())

            # Extract features for diversity
            if strategy_name == "bald_diversity":
                batch_features = uc_estimator.extract_features(batch_paths)
                features = np.zeros((n_total, batch_features.shape[1]))
                for i, idx in enumerate(batch_indices):
                    if i < len(batch_features):
                        features[idx] = batch_features[i]

        # 5c: Select samples via acquisition strategy
        selected = strategy.select(
            candidate_indices=batch_indices,
            budget=budget,
            det_scores=det_scores,
            bald_scores=bald_scores,
            uncertainty_scores=unc_scores,
            features=features,
        )

        print(f"  Selected {len(selected)} samples for annotation")

        # 5d: Simulate annotation (oracle: use ground-truth labels)
        # Add to training set
        dm.add_samples_to_training(selected, str(data_dir))
        labelled_indices.extend(selected)

        # Remove selected from pool
        selected_set = set(selected)
        u_pool = [idx for idx in u_pool if idx not in selected_set]

        trigger.add_annotations(len(selected))

        # 5e: Check update trigger
        should_update, reason = trigger.should_update()

        if should_update or cycle == K:  # Always update on last cycle
            if not should_update:
                reason = "final_cycle"

            print(f"  Update triggered: {reason}")

            # Incremental fine-tuning
            weights_path, update_time = trainer.incremental_update(
                yaml_path=yaml_path,
                cycle=cycle,
                epochs=config.training.epochs_per_cycle,
                batch_size=config.training.batch_size,
                lr=config.training.incremental_lr,
                output_dir=str(exp_dir / "runs" / "updates"),
            )

            # Update uncertainty estimator with new weights
            uc_estimator = MCDropoutEstimator(
                model_path=weights_path,
                T=config.uncertainty.T,
                dropout_rate=config.uncertainty.dropout_rate,
                conf_threshold=config.model.conf_threshold,
                iou_threshold=config.model.iou_threshold,
                img_size=config.model.img_size,
                device=config.device,
            )

            trigger.reset_after_update()
        else:
            update_time = 0
            reason = "no_update"
            print(f"  No update triggered (accumulated: {trigger.n_accumulated})")

        # 5f: Evaluate
        metrics_k = trainer.evaluate(yaml_path)
        print(f"  mAP50={metrics_k['mAP50']:.4f}, "
              f"P={metrics_k['precision']:.4f}, R={metrics_k['recall']:.4f}")

        results["trajectory"].append({
            "cycle": cycle,
            "n_labelled": len(labelled_indices),
            "n_new": len(selected),
            **metrics_k,
            "update_time": update_time,
            "trigger_reason": reason,
        })

    # ── Step 6: Measure inference latency ──
    test_paths = dm.get_image_paths_for_indices(splits["T"][:100])
    latency = trainer.measure_latency(test_paths)
    results["inference_latency_ms"] = latency
    print(f"\n[Latency] {latency:.2f} ms/image")

    # ── Step 7: Compute AUC ──
    trajectory = results["trajectory"]
    map_values = [t["mAP50"] for t in trajectory]
    n_labels = [t["n_labelled"] for t in trajectory]
    auc = np.trapezoid(map_values, n_labels) if len(map_values) > 1 else 0.0
    results["AUC"] = float(auc)
    print(f"[AUC] {auc:.4f}")

    # ── Step 8: Save results ──
    results_path = exp_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save trajectory as CSV
    df = pd.DataFrame(trajectory)
    df.to_csv(exp_dir / "trajectory.csv", index=False)

    print(f"\n[Done] Results saved to {exp_dir}")

    return results


def run_all_experiments(data_root, config, output_dir, class_names=None):
    """Run all strategy × L0 combinations."""
    all_results = []

    for l0 in config.dataset.l0_sizes:
        for strategy in config.active_learning.strategies:
            try:
                results = run_single_experiment(
                    data_root=data_root,
                    strategy_name=strategy,
                    l0_size=l0,
                    config=config,
                    output_dir=output_dir,
                    class_names=class_names,
                )
                all_results.append(results)
            except Exception as e:
                print(f"\n[ERROR] {strategy} L0={l0}: {e}")
                import traceback
                traceback.print_exc()

    # Save combined results
    combined_path = Path(output_dir) / "all_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"All experiments complete. Results: {combined_path}")
    print(f"{'='*70}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Adaptive UAV-Edge AL Experiment")
    parser.add_argument("--data_root", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--strategy", type=str, default=None,
                        choices=["random", "deterministic", "bald_only", "bald_diversity"],
                        help="Single strategy to run (or --run_all)")
    parser.add_argument("--l0_size", type=int, default=200, help="Initial labelled set size")
    parser.add_argument("--run_all", action="store_true", help="Run all strategies × L0 sizes")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cuda, cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--budget", type=int, default=50, help="Annotation budget per cycle")
    parser.add_argument("--cycles", type=int, default=8, help="Number of adaptation cycles")
    parser.add_argument("--T", type=int, default=8, help="MC Dropout passes")
    parser.add_argument("--class_names", type=str, nargs="+", default=None,
                        help="Class names (e.g., person vehicle)")

    args = parser.parse_args()

    config = ExperimentConfig()
    config.seed = args.seed
    config.device = args.device
    config.active_learning.budget_per_cycle = args.budget
    config.active_learning.num_cycles = args.cycles
    config.uncertainty.T = args.T

    set_seed(config.seed)

    if args.run_all:
        run_all_experiments(args.data_root, config, args.output_dir, args.class_names)
    elif args.strategy:
        run_single_experiment(
            data_root=args.data_root,
            strategy_name=args.strategy,
            l0_size=args.l0_size,
            config=config,
            output_dir=args.output_dir,
            class_names=args.class_names,
        )
    else:
        print("Specify --strategy or --run_all")


if __name__ == "__main__":
    main()
