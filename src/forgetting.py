"""
Catastrophic forgetting and continual-learning diagnostics.

The board's position on this was conditional: the thesis mentions
Continual Learning, so it must either treat forgetting properly or stop
foregrounding the topic. This module makes the first option available at
the cost of extra evaluation passes, so the thesis can report evidence
instead of retreating from the claim.

The measurement
---------------
Each adaptation cycle fine-tunes on newly annotated data. If that update
degrades the detector on domains it already handled, the system is
forgetting -- and in a disaster mission that means a model adapted to
flooded terrain that has quietly stopped finding people in the open
field it was trained on first.

To see this, the model is evaluated after every cycle on a *set* of
held-out domain test sets, not just the single global one:

    R[k][d] = mAP@50 of the model after cycle k on domain d

From that matrix the standard continual-learning scalars follow
(Lopez-Paz & Ranzato 2017; Chaudhry et al. 2018):

``backward_transfer`` (BWT)
    Mean change on earlier domains between the cycle they were last
    trained on and the end of the campaign. Negative means forgetting;
    positive means later data helped earlier domains too.

``forgetting_measure``
    Mean over domains of (best score ever achieved, minus final score).
    Unlike BWT this is never cancelled out by a domain that improved,
    so it is the more conservative statement.

``retention``
    Final score divided by peak score per domain, in [0, 1]. The number
    to quote in a sentence: "the adapted model retains 94% of its peak
    accuracy on the source domain".

``stability_gap``
    Worst single-cycle drop on any previously-learned domain. This is the
    operational risk metric: the deepest hole the system fell into at any
    point during the mission, which an average hides.

``plasticity``
    Mean improvement on the domain being adapted to. Reported alongside
    the forgetting numbers because the stability-plasticity trade-off is
    only interpretable when both sides are visible.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "ForgettingTracker",
    "backward_transfer",
    "forgetting_measure",
    "retention",
    "stability_gap",
]


class ForgettingTracker:
    """Accumulates the per-cycle, per-domain performance matrix.

    Usage inside the adaptation loop::

        tracker = ForgettingTracker(domains=["sard", "visdrone", "floodnet"])
        tracker.record(cycle=0, scores={"sard": 0.71, ...})
        ...
        summary = tracker.summary(adapted_domain="floodnet")

    Domains that are missing in a given cycle are stored as NaN and
    skipped by every statistic, so a campaign can evaluate expensive
    domains on a coarser schedule (for example every other cycle) without
    corrupting the metrics.
    """

    def __init__(self, domains: Sequence[str], metric: str = "mAP50"):
        self.domains = list(domains)
        self.metric = metric
        self.cycles: List[int] = []
        self.matrix: List[List[float]] = []

    def record(self, cycle: int, scores: Dict[str, float]) -> None:
        row = [float(scores.get(d, np.nan)) for d in self.domains]
        self.cycles.append(int(cycle))
        self.matrix.append(row)

    def as_array(self) -> np.ndarray:
        if not self.matrix:
            return np.zeros((0, len(self.domains)))
        return np.asarray(self.matrix, dtype=float)

    def summary(
        self,
        adapted_domain: Optional[str] = None,
        source_domain: Optional[str] = None,
    ) -> Dict:
        """Compute every continual-learning scalar from the matrix.

        Parameters
        ----------
        adapted_domain
            The domain the campaign is adapting *to* (the target under
            shift). Its trajectory measures plasticity, and it is excluded
            from the forgetting statistics, which concern prior domains.
        source_domain
            The domain the model started from. Reported separately because
            "does adaptation break the original mission capability" is the
            single question a reviewer will ask first.
        """
        R = self.as_array()
        if R.size == 0:
            return {"error": "no evaluations recorded"}

        prior_idx = [
            i for i, d in enumerate(self.domains) if d != adapted_domain
        ]

        result: Dict = {
            "metric": self.metric,
            "domains": self.domains,
            "cycles": self.cycles,
            "performance_matrix": R.tolist(),
            "n_cycles": len(self.cycles),
        }

        result["backward_transfer"] = backward_transfer(R, prior_idx)
        result["forgetting_measure"] = forgetting_measure(R, prior_idx)
        result["stability_gap"] = stability_gap(R, prior_idx)

        retentions = retention(R)
        result["retention_per_domain"] = {
            d: retentions[i] for i, d in enumerate(self.domains)
        }
        prior_retentions = [retentions[i] for i in prior_idx if not np.isnan(retentions[i])]
        result["retention_mean_prior_domains"] = (
            float(np.mean(prior_retentions)) if prior_retentions else float("nan")
        )

        if source_domain and source_domain in self.domains:
            j = self.domains.index(source_domain)
            column = R[:, j]
            valid = column[~np.isnan(column)]
            if valid.size:
                result["source_domain"] = source_domain
                result["source_initial"] = float(valid[0])
                result["source_final"] = float(valid[-1])
                result["source_peak"] = float(np.max(valid))
                result["source_drop_from_peak"] = float(np.max(valid) - valid[-1])
                result["source_retention"] = float(retentions[j])

        if adapted_domain and adapted_domain in self.domains:
            j = self.domains.index(adapted_domain)
            column = R[:, j]
            valid = column[~np.isnan(column)]
            if valid.size > 1:
                result["adapted_domain"] = adapted_domain
                result["plasticity"] = float(valid[-1] - valid[0])
                result["adapted_final"] = float(valid[-1])

        # The trade-off in one number: gain on the target per unit lost
        # on prior domains. Infinite when nothing was forgotten, which is
        # reported as None rather than a misleading large float.
        gain = result.get("plasticity")
        loss = result.get("forgetting_measure")
        if gain is not None and loss is not None and not np.isnan(loss):
            result["stability_plasticity_ratio"] = (
                float(gain / loss) if loss > 1e-9 else None
            )
        return result


def backward_transfer(R: np.ndarray, domain_idx: Optional[Sequence[int]] = None) -> float:
    """Mean change on prior domains from first evaluation to last.

    Negative values indicate forgetting. Defined over the cycles actually
    evaluated, so a domain measured only twice still contributes.
    """
    if R.shape[0] < 2:
        return float("nan")
    idx = list(domain_idx) if domain_idx is not None else list(range(R.shape[1]))
    deltas = []
    for j in idx:
        column = R[:, j]
        valid = column[~np.isnan(column)]
        if valid.size >= 2:
            deltas.append(valid[-1] - valid[0])
    return float(np.mean(deltas)) if deltas else float("nan")


def forgetting_measure(
    R: np.ndarray, domain_idx: Optional[Sequence[int]] = None
) -> float:
    """Mean of (peak - final) over domains; 0 means nothing was lost."""
    if R.shape[0] < 2:
        return float("nan")
    idx = list(domain_idx) if domain_idx is not None else list(range(R.shape[1]))
    losses = []
    for j in idx:
        column = R[:, j]
        valid = column[~np.isnan(column)]
        if valid.size >= 2:
            losses.append(max(0.0, float(np.max(valid) - valid[-1])))
    return float(np.mean(losses)) if losses else float("nan")


def retention(R: np.ndarray) -> List[float]:
    """Final / peak performance per domain, in [0, 1]."""
    out: List[float] = []
    for j in range(R.shape[1]):
        column = R[:, j]
        valid = column[~np.isnan(column)]
        if valid.size == 0 or np.max(valid) <= 0:
            out.append(float("nan"))
        else:
            out.append(float(valid[-1] / np.max(valid)))
    return out


def stability_gap(
    R: np.ndarray, domain_idx: Optional[Sequence[int]] = None
) -> float:
    """Worst single-cycle drop observed on any prior domain."""
    if R.shape[0] < 2:
        return float("nan")
    idx = list(domain_idx) if domain_idx is not None else list(range(R.shape[1]))
    worst = 0.0
    for j in idx:
        column = R[:, j]
        valid = column[~np.isnan(column)]
        if valid.size >= 2:
            drops = -np.diff(valid)
            worst = max(worst, float(np.max(drops)) if drops.size else 0.0)
    return worst
