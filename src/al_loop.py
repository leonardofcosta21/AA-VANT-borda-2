"""
The canonical instrumented Active Learning loop.

This is the single implementation of the four-stage cycle from Chapter 4
(acquisition -> inference and filtering -> supervision -> update). The
repository previously carried three partial copies of this loop -- one in
``pipeline.py`` that no longer matched its own helper signatures, one in
``run_experiment.py`` and one in ``shift_pipeline.py`` -- which is why
instrumentation added to one never reached the others.

Everything the examination board asked to be measured is collected here,
in the same run that produces the accuracy figures:

  * diversity diagnostics per acquisition cycle (H3, C4)
  * resource profiling per cycle and per phase (H4, C3)
  * network payload accounting per cycle (H4)
  * per-domain evaluation for forgetting analysis (Continual Learning)
  * both raw and budget-normalised AUC (board Sec.2)

Instrumentation is opt-in through the config so a cheap run stays cheap,
but it defaults to on: a cost number measured in a separate run from the
accuracy number it is paired with is a number nobody can defend.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.diversity_metrics import analyse_selection, aggregate_cycle_diversity
from src.forgetting import ForgettingTracker
from src.metrics import curve_summary, final_metrics
from src.profiling import (
    NetworkAccountant,
    ResourceMonitor,
    describe_platform,
    profile_inference,
)

__all__ = ["ALLoopConfig", "run_al_loop"]


@dataclass
class ALLoopConfig:
    """Resolved configuration for one run. Flat by design: it is written
    verbatim into the result JSON, and a nested structure would make the
    stored record harder to diff between runs."""

    data_root: str
    strategy: str
    l0_size: int
    seed: int = 42
    budget: int = 50
    cycles: int = 8
    model: str = "yolov8n.pt"
    img_size: int = 640
    conf_threshold: float = 0.25
    iou_threshold: float = 0.5
    epochs_initial: int = 30
    epochs_per_cycle: int = 5
    batch_size: int = 16
    initial_lr: float = 0.01
    incremental_lr: float = 0.001
    n_min: int = 50
    tau_shift_percentile: float = 90
    shift_window: int = 3
    T: int = 8
    dropout_rate: float = 0.1
    alpha: float = 0.1
    beta: int = 5
    cluster_divisor: object = 5
    test_fraction: float = 0.2
    device: str = "auto"
    class_names: Optional[List[str]] = None
    output_dir: str = "results/runs"
    collect_diversity_metrics: bool = True
    collect_profiling: bool = True
    collect_per_domain_eval: bool = False
    per_domain_eval_every: int = 1
    per_domain_roots: Dict[str, str] = field(default_factory=dict)
    link_mbps: float = 10.0
    realtime_budget_ms: float = 33.0
    estimator: str = "mc_dropout"
    ensemble_members: int = 5

    @classmethod
    def from_params(cls, params: Dict, seed: int) -> "ALLoopConfig":
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in params.items() if k in known}
        kwargs["seed"] = seed
        # `beta` is the cluster divisor in the thesis notation
        # (K = min(b, |C|/beta)); the sensitivity sweep varies it under
        # the clearer name, so keep the two in sync.
        divisor = kwargs.get("cluster_divisor", kwargs.get("beta", 5))
        if isinstance(divisor, (int, float)) and divisor:
            kwargs["beta"] = int(divisor)
        return cls(**kwargs)


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Determinism costs throughput but makes a reported number
        # reproducible, which is the whole point of contribution C6.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def run_al_loop(cfg: ALLoopConfig, run_id: str = "") -> Dict:
    """Execute one complete closed-loop run and return its full record."""
    from src.dataset_manager import DatasetManager
    from src.acquisition import get_strategy
    from src.trainer import ModelTrainer, UpdateTrigger

    set_seed(cfg.seed)
    exp_id = run_id or f"{cfg.strategy}_L0{cfg.l0_size}_B{cfg.budget}_s{cfg.seed}"
    exp_dir = Path(cfg.output_dir) / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    data_dir = exp_dir / "data"

    results: Dict = {
        "experiment_id": exp_id,
        "strategy": cfg.strategy,
        "estimator": cfg.estimator,
        "l0_size": cfg.l0_size,
        "budget": cfg.budget,
        "cycles": cfg.cycles,
        "seed": cfg.seed,
        "config": {k: _plain(v) for k, v in cfg.__dict__.items()},
        "platform": describe_platform(),
        "trajectory": [],
        "diversity_per_cycle": [],
        "profiling_per_cycle": [],
        "warnings": [],
    }

    # ---- Stage 0: splits ------------------------------------------------
    dm = DatasetManager(cfg.data_root, img_size=cfg.img_size, seed=cfg.seed)
    splits = dm.create_splits(l0_size=cfg.l0_size, test_fraction=cfg.test_fraction)
    yaml_path = dm.prepare_yolo_dataset(
        splits, str(data_dir), class_names=cfg.class_names
    )
    labelled = list(splits["L0"])
    u_pool = list(splits["U"])
    results["splits"] = {
        "L0": len(splits["L0"]),
        "U": len(splits["U"]),
        "T": len(splits["T"]),
    }
    if len(u_pool) < cfg.budget * cfg.cycles:
        results["warnings"].append(
            f"unlabelled pool ({len(u_pool)}) is smaller than the requested "
            f"budget ({cfg.budget * cfg.cycles}); later cycles will be short"
        )

    net = NetworkAccountant(link_mbps=cfg.link_mbps)

    # ---- Stage 0b: initial training ------------------------------------
    trainer = ModelTrainer(
        model_name=cfg.model, img_size=cfg.img_size, device=cfg.device
    )
    with ResourceMonitor("initial_training", interval_s=0.5) as monitor:
        weights_path = trainer.initial_training(
            yaml_path=yaml_path,
            epochs=cfg.epochs_initial,
            batch_size=cfg.batch_size,
            lr=cfg.initial_lr,
            output_dir=str(exp_dir / "runs"),
        )
    results["initial_training_profile"] = (
        monitor.report.as_dict() if monitor.report else {}
    )
    net.record(0, "weights_broadcast", paths=[weights_path])

    metrics_0 = trainer.evaluate(yaml_path)
    results["trajectory"].append(
        {
            "cycle": 0,
            "n_labelled": len(labelled),
            "n_new": 0,
            **metrics_0,
            "update_time": 0.0,
            "trigger_reason": "initial",
        }
    )

    # ---- Forgetting tracker --------------------------------------------
    tracker: Optional[ForgettingTracker] = None
    domain_yamls: Dict[str, str] = {}
    if cfg.collect_per_domain_eval and cfg.per_domain_roots:
        domain_yamls = _prepare_domain_test_sets(
            cfg.per_domain_roots, exp_dir / "domain_tests", cfg
        )
        if domain_yamls:
            tracker = ForgettingTracker(domains=sorted(domain_yamls))
            tracker.record(0, _evaluate_domains(trainer, domain_yamls))
        else:
            results["warnings"].append(
                "per-domain evaluation requested but no domain test set could "
                "be prepared; forgetting metrics will be absent"
            )

    # ---- Acquisition components ----------------------------------------
    estimator = _build_estimator(cfg, weights_path, yaml_path, dm, splits)
    strategy = get_strategy(
        cfg.strategy,
        {"seed": cfg.seed, "alpha": cfg.alpha, "beta": cfg.beta},
    )
    trigger = UpdateTrigger(n_min=cfg.n_min, window=cfg.shift_window)

    if cfg.strategy in ("bald_only", "bald_diversity") or cfg.estimator != "deterministic":
        calib_paths = dm.get_image_paths_for_indices(labelled[: min(100, len(labelled))])
        try:
            baseline_unc = estimator.compute_mc_uncertainty(calib_paths)
            trigger.calibrate_threshold(
                baseline_unc["variance"], percentile=cfg.tau_shift_percentile
            )
            results["tau_shift"] = trigger.tau_shift
        except Exception as exc:
            results["warnings"].append(f"threshold calibration failed: {exc}")

    # ---- Adaptation cycles ---------------------------------------------
    for cycle in range(1, cfg.cycles + 1):
        cycle_record, u_pool, labelled, weights_path, estimator = _run_cycle(
            cycle=cycle,
            cfg=cfg,
            dm=dm,
            trainer=trainer,
            estimator=estimator,
            strategy=strategy,
            trigger=trigger,
            u_pool=u_pool,
            labelled=labelled,
            yaml_path=yaml_path,
            data_dir=data_dir,
            exp_dir=exp_dir,
            weights_path=weights_path,
            net=net,
            results=results,
        )
        if cycle_record is None:
            results["warnings"].append(f"pool exhausted at cycle {cycle}")
            break
        results["trajectory"].append(cycle_record)

        if tracker is not None and cycle % max(cfg.per_domain_eval_every, 1) == 0:
            tracker.record(cycle, _evaluate_domains(trainer, domain_yamls))

    # ---- Final measurements ---------------------------------------------
    test_paths = dm.get_image_paths_for_indices(splits["T"][:200])

    deployment = profile_inference(
        predict_fn=lambda p: trainer.model.predict(
            p, imgsz=cfg.img_size, verbose=False,
            device=None if cfg.device == "auto" else cfg.device,
        ),
        image_paths=test_paths,
        label="deterministic_deployment",
        realtime_budget_ms=cfg.realtime_budget_ms,
    )
    results["inference_profile"] = deployment
    results["inference_latency_ms"] = deployment.get("latency_ms_mean")

    if cfg.collect_profiling and cfg.strategy in ("bald_only", "bald_diversity"):
        # Acquisition-time cost, measured separately because it runs off
        # the real-time path and must not be conflated with deployment
        # latency in the H4 argument.
        acquisition = profile_inference(
            predict_fn=lambda p: estimator.compute_mc_uncertainty([p]),
            image_paths=test_paths[:40],
            warmup=3,
            n_samples=30,
            label=f"mc_dropout_T{cfg.T}_acquisition",
            realtime_budget_ms=cfg.realtime_budget_ms,
        )
        results["acquisition_profile"] = acquisition
        det_mean = deployment.get("latency_ms_mean") or 0.0
        acq_mean = acquisition.get("latency_ms_mean") or 0.0
        results["mc_overhead_ratio"] = (
            float(acq_mean / det_mean) if det_mean > 0 else None
        )

    l_max = cfg.budget * cfg.cycles
    results.update(curve_summary(results["trajectory"], l_max=l_max))
    results.update(final_metrics(results["trajectory"]))
    results["AUC"] = results.get("auc")

    if results["diversity_per_cycle"]:
        results.update(aggregate_cycle_diversity(results["diversity_per_cycle"]))

    results["network"] = net.summary()
    if tracker is not None:
        results["forgetting"] = tracker.summary(
            source_domain=_guess_source_domain(cfg)
        )

    results["total_update_time_s"] = float(
        sum(t.get("update_time", 0.0) for t in results["trajectory"])
    )

    with open(exp_dir / "results.json", "w") as fh:
        json.dump(results, fh, indent=2, default=_plain)
    return results


# ---------------------------------------------------------------------------
# One cycle
# ---------------------------------------------------------------------------

def _run_cycle(
    cycle, cfg, dm, trainer, estimator, strategy, trigger, u_pool, labelled,
    yaml_path, data_dir, exp_dir, weights_path, net, results,
):
    """Stages 1-4 of the operational loop for a single cycle."""
    batch_indices = dm.get_cycle_batch(u_pool, cycle - 1, cfg.cycles)
    if not batch_indices:
        return None, u_pool, labelled, weights_path, estimator

    batch_paths = dm.get_image_paths_for_indices(batch_indices)
    n_total = len(dm.image_paths)

    det_scores = bald_scores = unc_scores = features = None
    monitor = ResourceMonitor(f"cycle{cycle}_acquisition", interval_s=0.3)
    monitor.start()
    t_score = time.time()

    def _extract_features():
        """Embeddings for the candidate pool, laid out by global index."""
        feats = estimator.extract_features(batch_paths)
        matrix = np.zeros((n_total, feats.shape[1]))
        for i, idx in enumerate(batch_indices):
            if i < len(feats):
                matrix[idx] = feats[i]
        return matrix

    if cfg.strategy == "random":
        # Random needs no scores, but the diversity diagnostics still
        # need embeddings to measure its cluster coverage. Without this,
        # the diversity comparison the board asked for would only ever
        # contain the two BALD strategies and could not show that
        # uninformed selection spreads differently.
        if cfg.collect_diversity_metrics:
            features = _extract_features()
    elif cfg.strategy == "deterministic":
        preds = estimator.compute_deterministic_predictions(batch_paths)
        det_scores = np.zeros(n_total)
        for idx, pred in zip(batch_indices, preds):
            scores = pred.get("scores", np.array([]))
            det_scores[idx] = 1.0 - (float(scores.max()) if len(scores) else 0.0)
        if cfg.collect_diversity_metrics:
            features = _extract_features()
    else:
        unc = estimator.compute_mc_uncertainty(batch_paths)
        unc_scores = np.zeros(n_total)
        bald_scores = np.zeros(n_total)
        for i, idx in enumerate(batch_indices):
            unc_scores[idx] = unc["variance"][i]
            bald_scores[idx] = unc["bald"][i]
        trigger.record_batch_uncertainty(float(np.mean(unc["variance"])))

        if "diversity" in cfg.strategy or cfg.collect_diversity_metrics:
            features = _extract_features()

    scoring_time = time.time() - t_score

    selected = strategy.select(
        candidate_indices=batch_indices,
        budget=cfg.budget,
        det_scores=det_scores,
        bald_scores=bald_scores,
        uncertainty_scores=unc_scores,
        features=features,
    )
    acquisition_report = monitor.stop()

    # -- Diversity diagnostics (H3, C4) --------------------------------
    if cfg.collect_diversity_metrics and features is not None and selected:
        position_of = {idx: pos for pos, idx in enumerate(batch_indices)}
        try:
            diagnostic = analyse_selection(
                pool_features=features[batch_indices],
                selected_positions=[
                    position_of[i] for i in selected if i in position_of
                ],
                cycle=cycle,
                strategy=cfg.strategy,
                beta=cfg.beta,
                scores=(
                    bald_scores[batch_indices] if bald_scores is not None else None
                ),
                seed=cfg.seed,
            )
            results["diversity_per_cycle"].append(diagnostic)
        except Exception as exc:
            results["warnings"].append(
                f"diversity diagnostic failed at cycle {cycle}: {exc}"
            )

    # -- Supervision: oracle annotation ---------------------------------
    selected_paths = dm.get_image_paths_for_indices(selected)
    net.record(cycle, "candidate_uplink", paths=selected_paths)
    # Label payload: YOLO .txt files are the annotations returned by the
    # operator, so their real size on disk is the downlink cost.
    label_paths = [
        p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        for p in selected_paths
    ]
    net.record(cycle, "label_downlink", paths=label_paths, n_items=len(selected))

    dm.add_samples_to_training(selected, str(data_dir))
    labelled.extend(selected)
    selected_set = set(selected)
    u_pool = [i for i in u_pool if i not in selected_set]
    trigger.add_annotations(len(selected))

    # -- Update stage ----------------------------------------------------
    should_update, reason = trigger.should_update()
    update_time = 0.0
    update_report = None
    if should_update or cycle == cfg.cycles:
        if not should_update:
            reason = "final_cycle"
        update_monitor = ResourceMonitor(f"cycle{cycle}_finetune", interval_s=0.5)
        update_monitor.start()
        weights_path, update_time = trainer.incremental_update(
            yaml_path=yaml_path,
            cycle=cycle,
            epochs=cfg.epochs_per_cycle,
            batch_size=cfg.batch_size,
            lr=cfg.incremental_lr,
            output_dir=str(exp_dir / "runs" / "updates"),
        )
        update_report = update_monitor.stop()
        net.record(cycle, "weights_broadcast", paths=[weights_path])
        estimator = _rebuild_estimator(estimator, cfg, weights_path)
        trigger.reset_after_update()
    else:
        reason = "no_update"

    metrics = trainer.evaluate(yaml_path)

    if cfg.collect_profiling:
        results["profiling_per_cycle"].append(
            {
                "cycle": cycle,
                "scoring_time_s": scoring_time,
                "update_time_s": update_time,
                "n_candidates": len(batch_indices),
                "acquisition": acquisition_report.as_dict()
                if acquisition_report
                else {},
                "finetune": update_report.as_dict() if update_report else {},
            }
        )

    return (
        {
            "cycle": cycle,
            "n_labelled": len(labelled),
            "n_new": len(selected),
            **metrics,
            "update_time": update_time,
            "scoring_time": scoring_time,
            "trigger_reason": reason,
        },
        u_pool,
        labelled,
        weights_path,
        estimator,
    )


# ---------------------------------------------------------------------------
# Estimator construction
# ---------------------------------------------------------------------------

def _build_estimator(cfg: ALLoopConfig, weights_path, yaml_path, dm, splits):
    """Instantiate the uncertainty estimator named by the config.

    All variants expose the MCDropoutEstimator interface, so the loop
    above is written once and the comparison the board asked for (MC
    Dropout vs Deep Ensembles vs conformal vs SWAG vs BSB/PSB) runs
    through identical code paths.
    """
    from src.mc_dropout import MCDropoutEstimator

    base = MCDropoutEstimator(
        model_path=weights_path,
        T=cfg.T,
        dropout_rate=cfg.dropout_rate,
        conf_threshold=cfg.conf_threshold,
        iou_threshold=cfg.iou_threshold,
        img_size=cfg.img_size,
        device=cfg.device,
    )
    name = (cfg.estimator or "mc_dropout").lower()
    if name in ("mc_dropout", "deterministic", "bald", "bsb", "psb", "bsb_diversity"):
        return base

    if name == "deep_ensemble":
        from src.deep_ensemble import DeepEnsembleEstimator, train_ensemble_members

        trained = train_ensemble_members(
            yaml_path=yaml_path,
            n_members=cfg.ensemble_members,
            base_seed=cfg.seed,
            model_name=cfg.model,
            epochs=cfg.epochs_initial,
            batch_size=cfg.batch_size,
            lr=cfg.initial_lr,
            img_size=cfg.img_size,
            device=cfg.device,
            output_dir=str(Path(cfg.output_dir) / "ensemble"),
        )
        return DeepEnsembleEstimator(
            member_paths=trained["weights"],
            conf_threshold=cfg.conf_threshold,
            iou_threshold=cfg.iou_threshold,
            img_size=cfg.img_size,
            device=cfg.device,
        )

    from src.uncertainty_extended import get_uncertainty_estimator

    # Conformal prediction needs a calibration set that is disjoint from
    # both the training set and the pool being scored, otherwise its
    # coverage guarantee does not hold. A slice of the held-out test set
    # is the only partition that satisfies that here; it is excluded from
    # the reported test metrics by taking the tail rather than the head.
    calibration = dm.get_image_paths_for_indices(
        splits["T"][-min(200, len(splits["T"])):]
    )
    estimator = get_uncertainty_estimator(
        method=name,
        model_path=weights_path,
        config={
            "T": cfg.T,
            "dropout_rate": cfg.dropout_rate,
            "device": cfg.device,
            "img_size": cfg.img_size,
            "conf_threshold": cfg.conf_threshold,
            "iou_threshold": cfg.iou_threshold,
        },
        calibration_paths=calibration,
    )
    if name == "swag" and hasattr(estimator, "collect_swag_statistics"):
        # SWAG fits its Gaussian over the trajectory of SGD iterates, so
        # it needs a short additional training run before it can produce
        # any uncertainty at all.
        estimator.collect_swag_statistics(yaml_path, epochs=cfg.epochs_per_cycle * 2)
    return estimator


def _rebuild_estimator(current, cfg: ALLoopConfig, weights_path: str):
    """Point the estimator at the freshly fine-tuned weights.

    Uncertainty must be computed with the current model, otherwise the
    acquisition scores describe a model that no longer exists. Ensembles
    are the exception: retraining five members every cycle is not
    affordable, so the members are kept and the cost of that choice is
    recorded as a limitation rather than hidden.
    """
    from src.mc_dropout import MCDropoutEstimator

    if current.__class__.__name__ == "DeepEnsembleEstimator":
        return current
    try:
        return MCDropoutEstimator(
            model_path=weights_path,
            T=cfg.T,
            dropout_rate=cfg.dropout_rate,
            conf_threshold=cfg.conf_threshold,
            iou_threshold=cfg.iou_threshold,
            img_size=cfg.img_size,
            device=cfg.device,
        )
    except Exception:
        return current


# ---------------------------------------------------------------------------
# Per-domain evaluation
# ---------------------------------------------------------------------------

def _prepare_domain_test_sets(roots: Dict[str, str], out_dir: Path, cfg) -> Dict[str, str]:
    """Build a small fixed test set per source domain.

    These are held out once, with the split seed, and reused across all
    cycles, so the forgetting matrix compares like with like.
    """
    from src.dataset_manager import DatasetManager

    prepared: Dict[str, str] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, root in roots.items():
        if not Path(root).exists():
            continue
        try:
            dm = DatasetManager(root, img_size=cfg.img_size, seed=cfg.seed)
            splits = dm.create_splits(l0_size=1, test_fraction=cfg.test_fraction)
            yaml_path = dm.prepare_yolo_dataset(
                splits, str(out_dir / name), class_names=cfg.class_names
            )
            prepared[name] = yaml_path
        except Exception:
            continue
    return prepared


def _evaluate_domains(trainer, domain_yamls: Dict[str, str]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for name, yaml_path in domain_yamls.items():
        try:
            scores[name] = float(trainer.evaluate(yaml_path)["mAP50"])
        except Exception:
            scores[name] = float("nan")
    return scores


def _guess_source_domain(cfg: ALLoopConfig) -> Optional[str]:
    root = Path(cfg.data_root).name.lower()
    for name in cfg.per_domain_roots:
        if name.lower() in root:
            return name
    return None


def _plain(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value
