"""
Diversity diagnostics for the acquisition step (H3, C4).

The board asked the thesis to "abordar explicitamente a promocao de
diversidade no processo de selecao de amostras". Until now the repository
implemented diversity but never measured it: BALD+Diversity could only be
defended by its downstream accuracy, which confounds the mechanism with
its effect. These metrics observe the selection itself, so the thesis can
show *that* the selected batches are more spread out, separately from
whether that spreading helped.

Metrics
-------
``cluster_coverage``
    Fraction of the k-means clusters in the candidate pool that receive at
    least one annotation this cycle. BALD-only typically concentrates in
    one or two clusters; the proposed strategy should approach 1.0. This
    is the direct evidence for the diversity claim.

``selection_entropy``
    Normalised Shannon entropy of the selected batch over clusters. 1.0
    means the budget was spread perfectly evenly, 0.0 means it all landed
    in one cluster. Complements coverage, which is blind to how lopsided
    the covered clusters are.

``mean_pairwise_distance`` / ``min_pairwise_distance``
    Geometry of the selected batch in embedding space. The minimum is the
    redundancy signal: two nearly identical frames from consecutive video
    positions show up as a near-zero minimum distance, which is exactly
    the failure mode diversity is meant to prevent in aerial footage.

``redundancy_rate``
    Fraction of selected pairs closer than a threshold derived from the
    candidate pool's own distance distribution, so it adapts to the scale
    of the embedding rather than using an arbitrary constant.

``representativeness``
    Mean distance from each *unselected* candidate to its nearest
    selected sample, normalised by the pool's mean pairwise distance.
    Lower is better: it says the annotated batch stands in well for the
    pool it was drawn from.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
except Exception:  # pragma: no cover
    KMeans = None
    silhouette_score = None

__all__ = [
    "cluster_coverage",
    "selection_entropy",
    "pairwise_distance_stats",
    "representativeness",
    "analyse_selection",
    "aggregate_cycle_diversity",
]


def _as_matrix(features: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(features, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def cluster_coverage(
    cluster_labels: Sequence[int],
    selected_positions: Sequence[int],
) -> Dict[str, float]:
    """Cluster occupancy of the selected batch.

    ``selected_positions`` indexes into ``cluster_labels``, i.e. positions
    within the candidate pool, not global dataset indices.
    """
    labels = np.asarray(cluster_labels)
    if labels.size == 0:
        return {"coverage": 0.0, "n_clusters": 0, "n_clusters_hit": 0}

    selected = np.asarray(list(selected_positions), dtype=int)
    selected = selected[(selected >= 0) & (selected < len(labels))]
    all_clusters = np.unique(labels)
    hit_clusters = np.unique(labels[selected]) if selected.size else np.array([])

    return {
        "coverage": float(len(hit_clusters) / len(all_clusters)),
        "n_clusters": int(len(all_clusters)),
        "n_clusters_hit": int(len(hit_clusters)),
    }


def selection_entropy(
    cluster_labels: Sequence[int],
    selected_positions: Sequence[int],
) -> float:
    """Normalised entropy of the batch's cluster distribution, in [0, 1]."""
    labels = np.asarray(cluster_labels)
    selected = np.asarray(list(selected_positions), dtype=int)
    selected = selected[(selected >= 0) & (selected < len(labels))]
    if selected.size == 0:
        return 0.0

    _, counts = np.unique(labels[selected], return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-12))

    n_clusters = len(np.unique(labels))
    max_entropy = np.log(max(n_clusters, 2))
    if max_entropy <= 0:
        return 0.0
    # The 1e-12 guard inside the log makes a single-cluster selection come
    # out at -1e-12 rather than exactly zero. Entropy is non-negative by
    # definition, so clamp: a negative value in a results table reads as a
    # bug even when it is float noise.
    return float(max(0.0, entropy / max_entropy))


def pairwise_distance_stats(
    features: Sequence[Sequence[float]],
    redundancy_quantile: float = 0.05,
    reference_features: Optional[Sequence[Sequence[float]]] = None,
) -> Dict[str, float]:
    """Distance geometry of a set of embeddings.

    ``reference_features`` (the full candidate pool) calibrates the
    redundancy threshold. Without it the threshold falls back to the
    selected set's own distribution, which makes the metric
    self-referential and much less informative across strategies.
    """
    matrix = _as_matrix(features)
    if len(matrix) < 2:
        return {
            "mean_pairwise_distance": 0.0,
            "min_pairwise_distance": 0.0,
            "std_pairwise_distance": 0.0,
            "redundancy_rate": 0.0,
        }

    distances = _condensed_distances(matrix)

    if reference_features is not None and len(reference_features) >= 2:
        ref = _as_matrix(reference_features)
        ref_distances = _condensed_distances(ref, max_points=400)
        threshold = float(np.quantile(ref_distances, redundancy_quantile))
    else:
        threshold = float(np.quantile(distances, redundancy_quantile))

    return {
        "mean_pairwise_distance": float(distances.mean()),
        "min_pairwise_distance": float(distances.min()),
        "std_pairwise_distance": float(distances.std()),
        "redundancy_threshold": threshold,
        "redundancy_rate": float((distances <= threshold).mean()),
    }


def _condensed_distances(matrix: np.ndarray, max_points: int = 600) -> np.ndarray:
    """Upper-triangular pairwise Euclidean distances.

    Subsamples deterministically above ``max_points`` so a large candidate
    pool cannot turn a diagnostic into the campaign's bottleneck.
    """
    if len(matrix) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(matrix), size=max_points, replace=False)
        matrix = matrix[np.sort(idx)]
    diff = matrix[:, None, :] - matrix[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    iu = np.triu_indices(len(matrix), k=1)
    return dist[iu]


def representativeness(
    pool_features: Sequence[Sequence[float]],
    selected_positions: Sequence[int],
) -> Dict[str, float]:
    """How well the selected batch covers the candidate pool.

    Computes the mean and maximum distance from each unselected candidate
    to its nearest selected neighbour (a Hausdorff-style coverage
    radius), normalised by the pool's own scale so values are comparable
    across cycles and datasets.
    """
    pool = _as_matrix(pool_features)
    selected = np.asarray(list(selected_positions), dtype=int)
    selected = selected[(selected >= 0) & (selected < len(pool))]
    if len(pool) < 2 or selected.size == 0:
        return {"coverage_radius_mean": 0.0, "coverage_radius_max": 0.0}

    mask = np.ones(len(pool), dtype=bool)
    mask[selected] = False
    unselected = pool[mask]
    if len(unselected) == 0:
        return {"coverage_radius_mean": 0.0, "coverage_radius_max": 0.0}

    chosen = pool[selected]
    diff = unselected[:, None, :] - chosen[None, :, :]
    distances = np.sqrt((diff ** 2).sum(axis=-1))
    nearest = distances.min(axis=1)

    scale = float(_condensed_distances(pool, max_points=300).mean()) or 1.0
    return {
        "coverage_radius_mean": float(nearest.mean() / scale),
        "coverage_radius_max": float(nearest.max() / scale),
        "pool_distance_scale": scale,
    }


def analyse_selection(
    pool_features: Sequence[Sequence[float]],
    selected_positions: Sequence[int],
    cycle: int = 0,
    strategy: str = "",
    n_clusters: Optional[int] = None,
    beta: int = 5,
    scores: Optional[Sequence[float]] = None,
    seed: int = 42,
) -> Dict:
    """Full diversity diagnostic for one acquisition cycle.

    Clusters the candidate pool independently of the acquisition strategy
    so that every strategy is scored against the *same* partition of the
    feature space. Without this, BALD-only would have no clusters to be
    measured against and the comparison the board asked for would be
    impossible.
    """
    pool = _as_matrix(pool_features)
    selected = [int(i) for i in selected_positions if 0 <= int(i) < len(pool)]

    result: Dict = {
        "cycle": cycle,
        "strategy": strategy,
        "n_candidates": int(len(pool)),
        "n_selected": len(selected),
    }
    if len(pool) < 2 or not selected:
        result["note"] = "pool or selection too small to analyse"
        return result

    k = n_clusters or max(2, min(len(selected), len(pool) // max(beta, 1)))
    k = int(max(2, min(k, len(pool) - 1)))

    labels = None
    if KMeans is not None:
        try:
            kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=100)
            labels = kmeans.fit_predict(pool)
        except Exception as exc:
            result["cluster_error"] = str(exc)

    if labels is not None:
        result.update(cluster_coverage(labels, selected))
        result["selection_entropy"] = selection_entropy(labels, selected)
        if silhouette_score is not None and len(np.unique(labels)) > 1:
            try:
                result["pool_silhouette"] = float(silhouette_score(pool, labels))
            except Exception:
                pass

    result.update(
        pairwise_distance_stats(pool[selected], reference_features=pool)
    )
    result.update(representativeness(pool, selected))

    if scores is not None and len(scores) == len(pool):
        arr = np.asarray(scores, dtype=float)
        chosen = arr[selected]
        result["selected_score_mean"] = float(chosen.mean())
        result["pool_score_mean"] = float(arr.mean())
        order = np.argsort(arr)[::-1][: len(selected)]
        result["score_overlap_with_topk"] = float(
            len(set(order.tolist()) & set(selected)) / max(len(selected), 1)
        )
    return result


def aggregate_cycle_diversity(cycle_reports: List[Dict]) -> Dict[str, float]:
    """Average the per-cycle diagnostics into run-level scalars."""
    if not cycle_reports:
        return {}
    keys = [
        "coverage",
        "selection_entropy",
        "mean_pairwise_distance",
        "min_pairwise_distance",
        "redundancy_rate",
        "coverage_radius_mean",
        "score_overlap_with_topk",
    ]
    summary: Dict[str, float] = {}
    for key in keys:
        values = [r[key] for r in cycle_reports if isinstance(r.get(key), (int, float))]
        if values:
            summary[f"diversity_{key}_mean"] = float(np.mean(values))
            summary[f"diversity_{key}_std"] = float(np.std(values))
    summary["diversity_n_cycles"] = len(cycle_reports)
    return summary
