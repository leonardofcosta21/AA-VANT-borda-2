"""
Core evaluation metrics for the UAV-Edge Active Learning campaign.

This module centralises every scalar the thesis reports, so that the
definition used in a table is the definition used in a figure and in the
statistical test.

Metric families
---------------
Annotation efficiency (H1)
    ``auc_learning_curve``      raw area under mAP@50 vs cumulative labels
    ``auc_normalized``          the same area divided by the annotation
                                budget, i.e. a budget-weighted mean mAP@50
                                in [0, 1]. The examination board asked for
                                the raw AUC (240.65) to be normalised so it
                                can be read directly; ``auc_normalized``
                                is that number.
    ``label_efficiency``        labels needed to reach a target mAP@50,
                                relative to the random baseline.

Stability under shift (H2)
    ``recall_variance``         inter-cycle variance of recall
    ``monotonicity``            fraction of non-decreasing steps
    ``max_drawdown``            worst peak-to-trough drop of the curve
    ``curve_smoothness``        std of first differences

Diversity (H3, C4)
    see :mod:`src.diversity_metrics`

Operational cost (H4)
    see :mod:`src.profiling`

All functions accept plain Python lists or numpy arrays and return plain
floats, so results are JSON-serialisable without conversion helpers.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "trapezoid",
    "auc_learning_curve",
    "auc_normalized",
    "label_efficiency",
    "recall_variance",
    "monotonicity",
    "max_drawdown",
    "curve_smoothness",
    "curve_summary",
    "final_metrics",
]


def trapezoid(y: Sequence[float], x: Sequence[float]) -> float:
    """Trapezoidal integration that works on every supported numpy version.

    numpy renamed ``trapz`` to ``trapezoid`` in 2.0 and the old spelling
    emits a DeprecationWarning. The original code base called both
    spellings in different modules, which made results depend on the numpy
    version installed. This helper removes that dependency.
    """
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    valid = ~(np.isnan(y_arr) | np.isnan(x_arr))
    if valid.sum() < 2:
        return 0.0
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return float(integrate(y_arr[valid], x_arr[valid]))


# ---------------------------------------------------------------------------
# H1 -- annotation efficiency
# ---------------------------------------------------------------------------

def auc_learning_curve(
    performance: Sequence[float],
    cumulative_labels: Sequence[float],
) -> float:
    """Raw area under the learning curve, in units of mAP@50 x labels.

    This is Equation 5.1 of the thesis. The value is scale-dependent: it
    grows with the annotation budget, which is why a run with
    ``Lmax = 400`` reports numbers around 200-250 while the underlying
    mAP@50 never exceeds 1. Report :func:`auc_normalized` alongside it.
    """
    return trapezoid(performance, cumulative_labels)


def auc_normalized(
    performance: Sequence[float],
    cumulative_labels: Sequence[float],
    l_max: Optional[float] = None,
) -> float:
    """Budget-normalised AUC in [0, 1].

    Divides the raw area by the horizontal span actually integrated over,
    which turns the metric into the mean mAP@50 sustained across the
    annotation budget. Two runs with different budgets become comparable,
    and the number can be read as "average detection quality per unit of
    annotation spent".

    Parameters
    ----------
    l_max
        Span to normalise by. Defaults to the observed span
        ``cumulative_labels[-1] - cumulative_labels[0]``. Pass the nominal
        budget ``K * b_t`` to normalise every strategy by the same
        constant even when a pool runs dry early.
    """
    x = np.asarray(cumulative_labels, dtype=float)
    valid = ~np.isnan(x)
    if valid.sum() < 2:
        return 0.0
    span = float(l_max) if l_max else float(x[valid][-1] - x[valid][0])
    if span <= 0:
        return 0.0
    return auc_learning_curve(performance, cumulative_labels) / span


def label_efficiency(
    performance: Sequence[float],
    cumulative_labels: Sequence[float],
    target: float,
) -> Optional[float]:
    """Number of labels needed to first reach ``target`` mAP@50.

    Returns ``None`` when the curve never reaches the target, which the
    caller should report as "not reached" rather than silently treating as
    the budget ceiling. Linear interpolation is used between the two
    cycles that bracket the crossing.
    """
    y = np.asarray(performance, dtype=float)
    x = np.asarray(cumulative_labels, dtype=float)
    for i in range(len(y)):
        if y[i] >= target:
            if i == 0:
                return float(x[0])
            y0, y1 = y[i - 1], y[i]
            x0, x1 = x[i - 1], x[i]
            if y1 == y0:
                return float(x1)
            return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))
    return None


# ---------------------------------------------------------------------------
# H2 -- stability
# ---------------------------------------------------------------------------

def recall_variance(recalls: Sequence[float]) -> float:
    """Inter-cycle variance of recall: the board's stability criterion."""
    vals = _clean(recalls)
    return float(np.var(vals)) if len(vals) > 1 else 0.0


def monotonicity(values: Sequence[float]) -> float:
    """Fraction of consecutive steps that do not decrease.

    1.0 means the curve never regressed across the campaign; 0.5 means it
    regressed half the time. Reported for both mAP@50 and recall.
    """
    vals = _clean(values)
    if len(vals) < 2:
        return 0.0
    diffs = np.diff(vals)
    return float((diffs >= 0).mean())


def max_drawdown(values: Sequence[float]) -> float:
    """Largest drop from a running maximum.

    Complements variance: a curve can have low variance and still contain
    one catastrophic cycle, which is the failure mode that matters
    operationally (a model update that makes the detector worse mid
    mission).
    """
    vals = _clean(values)
    if len(vals) < 2:
        return 0.0
    running_max = np.maximum.accumulate(vals)
    return float(np.max(running_max - vals))


def curve_smoothness(values: Sequence[float]) -> float:
    """Standard deviation of first differences (lower is smoother)."""
    vals = _clean(values)
    if len(vals) < 2:
        return 0.0
    return float(np.std(np.diff(vals)))


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------

def curve_summary(
    trajectory: List[Dict],
    performance_key: str = "mAP50",
    label_key: str = "n_labelled",
    l_max: Optional[float] = None,
    targets: Sequence[float] = (0.5, 0.6, 0.7),
) -> Dict[str, float]:
    """Every curve-level scalar for one run, computed from its trajectory.

    Designed to be called once per run and stored verbatim in the run's
    JSON, so that aggregation and statistics never recompute metrics from
    partially aligned arrays.
    """
    perf = [t.get(performance_key, np.nan) for t in trajectory]
    labels = [t.get(label_key, np.nan) for t in trajectory]
    recalls = [t.get("recall", np.nan) for t in trajectory]

    summary: Dict[str, float] = {
        "auc": auc_learning_curve(perf, labels),
        "auc_normalized": auc_normalized(perf, labels, l_max=l_max),
        "recall_variance": recall_variance(recalls),
        "recall_monotonicity": monotonicity(recalls),
        "map_monotonicity": monotonicity(perf),
        "map_max_drawdown": max_drawdown(perf),
        "recall_max_drawdown": max_drawdown(recalls),
        "map_smoothness": curve_smoothness(perf),
    }
    for target in targets:
        key = f"labels_to_map{int(round(target * 100))}"
        value = label_efficiency(perf, labels, target)
        summary[key] = float(value) if value is not None else float("nan")
    return summary


def final_metrics(trajectory: List[Dict]) -> Dict[str, float]:
    """Last-cycle detection metrics, guarded against an empty trajectory."""
    if not trajectory:
        return {}
    last = trajectory[-1]
    return {
        f"final_{k}": float(last[k])
        for k in ("mAP50", "mAP50_95", "precision", "recall")
        if k in last and last[k] is not None
    }


def _clean(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[~np.isnan(arr)]
