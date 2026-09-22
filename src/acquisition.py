"""
Acquisition Strategies for Active Learning.
Implements the 4 baseline strategies described in Chapter 4, Section 4.3.

Controlled progression:
  1. Random Sampling (no information-based selection)
  2. Deterministic Uncertainty (single-pass confidence)
  3. BALD-only (MC Dropout, no diversity)
  4. BALD + Diversity (proposed: BALD + k-means cluster-then-select)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans


def apply_uncertainty_prefilter(
    candidate_indices: List[int],
    uncertainty_scores: Optional[np.ndarray],
    budget: int,
    alpha: float,
) -> List[int]:
    """Keep the most uncertain candidates before the acquisition step.

    This is the adaptive threshold tau_t = quantile_{1-alpha} from Section
    4.3.2 of the thesis. The original implementation kept the filtered set
    only when it already contained at least ``budget`` samples, which with
    alpha = 0.1 and a pool of ~350 candidates produced ~35 survivors
    against a budget of 50 -- so the filter was discarded on every cycle
    and the documented two-stage selection silently never ran.

    The fix keeps the threshold but guarantees the survivors can fund the
    budget: the retained fraction is at least ``budget / |C|``, so the
    filter narrows the pool whenever narrowing is possible and degrades
    gracefully to a no-op on small pools instead of being dropped
    entirely. A short pool (fewer candidates than the budget) is returned
    untouched, since there is nothing to choose between.
    """
    n = len(candidate_indices)
    if uncertainty_scores is None or n == 0 or n <= budget:
        return list(candidate_indices)

    effective_alpha = float(min(1.0, max(alpha, budget / n)))
    if effective_alpha >= 1.0:
        return list(candidate_indices)

    scores = np.asarray(
        [uncertainty_scores[i] for i in candidate_indices], dtype=float
    )
    threshold = float(np.quantile(scores, 1.0 - effective_alpha))
    filtered = [
        idx for idx, score in zip(candidate_indices, scores) if score >= threshold
    ]
    # Ties at the threshold can leave fewer than the budget; top up by
    # score order so the budget is always fundable.
    if len(filtered) < budget:
        order = np.argsort(scores)[::-1]
        filtered = [candidate_indices[i] for i in order[:budget]]
    return filtered


class AcquisitionStrategy:
    """Base class for acquisition strategies."""

    def __init__(self, name: str):
        self.name = name

    def select(
        self,
        candidate_indices: List[int],
        budget: int,
        **kwargs
    ) -> List[int]:
        raise NotImplementedError


class RandomSampling(AcquisitionStrategy):
    """
    Baseline 1: Select b_t samples uniformly at random.
    S_k ~ Uniform(U_k)
    """

    def __init__(self, seed: int = 42):
        super().__init__("random")
        self.rng = np.random.RandomState(seed)

    def select(
        self,
        candidate_indices: List[int],
        budget: int,
        **kwargs
    ) -> List[int]:
        budget = min(budget, len(candidate_indices))
        selected = self.rng.choice(
            candidate_indices, size=budget, replace=False
        )
        return selected.tolist()


class DeterministicUncertaintySampling(AcquisitionStrategy):
    """
    Baseline 2: Rank by low confidence from single deterministic pass.
    u_det(x) = 1 - max_i p_i(x)

    Prioritises samples with lowest maximum confidence.
    """

    def __init__(self):
        super().__init__("deterministic")

    def select(
        self,
        candidate_indices: List[int],
        budget: int,
        det_scores: np.ndarray = None,
        **kwargs
    ) -> List[int]:
        """
        Args:
            candidate_indices: indices of candidate samples
            budget: number of samples to select
            det_scores: deterministic uncertainty scores (1 - max_conf)
                       indexed by global image index
        """
        if det_scores is None:
            raise ValueError("det_scores required for deterministic sampling")

        budget = min(budget, len(candidate_indices))

        # Get scores for candidates
        scores = det_scores[candidate_indices]

        # Select top-budget by highest uncertainty (lowest confidence)
        top_indices = np.argsort(scores)[::-1][:budget]
        selected = [candidate_indices[i] for i in top_indices]

        return selected


class BALDOnlySampling(AcquisitionStrategy):
    """
    Baseline 3: BALD-based selection without diversity filtering.
    Select top-b_t samples by BALD score directly.

    Isolates the contribution of Bayesian epistemic uncertainty (C2)
    from the diversity mechanism, providing reference for H3.
    """

    def __init__(self, alpha: float = 0.1):
        super().__init__("bald_only")
        self.alpha = alpha

    def select(
        self,
        candidate_indices: List[int],
        budget: int,
        bald_scores: np.ndarray = None,
        uncertainty_scores: np.ndarray = None,
        **kwargs
    ) -> List[int]:
        """
        Args:
            candidate_indices: indices of candidate samples
            budget: annotation budget
            bald_scores: BALD scores indexed by global image index
            uncertainty_scores: variance scores for threshold filtering
        """
        if bald_scores is None:
            raise ValueError("bald_scores required for BALD sampling")

        candidate_indices = apply_uncertainty_prefilter(
            candidate_indices, uncertainty_scores, budget, self.alpha
        )

        budget = min(budget, len(candidate_indices))

        # Rank by BALD score
        scores = bald_scores[candidate_indices]
        top_indices = np.argsort(scores)[::-1][:budget]
        selected = [candidate_indices[i] for i in top_indices]

        return selected


class BALDDiversitySampling(AcquisitionStrategy):
    """
    Proposed method (Baseline 4): BALD + k-means cluster-then-select.

    Two-step procedure from Chapter 3, Section 3.3:
    1. Rank candidates by BALD score (informativeness)
    2. Cluster feature embeddings with k-means, select top-scoring
       from each cluster proportionally (diversity)

    Combines uncertainty-driven informativeness (C2) with
    representativeness-driven diversity (C4).
    """

    def __init__(self, alpha: float = 0.1, beta: int = 5, seed: int = 42):
        super().__init__("bald_diversity")
        self.alpha = alpha
        self.beta = beta
        self.seed = seed
        # Provenance of the most recent selection, consumed by
        # src.diversity_metrics so the thesis can report *how* the batch
        # was spread and not only what accuracy followed from it.
        self.last_selection_info: Dict = {}

    def select(
        self,
        candidate_indices: List[int],
        budget: int,
        bald_scores: np.ndarray = None,
        uncertainty_scores: np.ndarray = None,
        features: np.ndarray = None,
        **kwargs
    ) -> List[int]:
        """
        Args:
            candidate_indices: indices of candidate samples
            budget: annotation budget b_t
            bald_scores: BALD scores indexed by global image index
            uncertainty_scores: variance scores for threshold filtering
            features: feature embeddings (n_total x d) indexed by global image index
        """
        if bald_scores is None:
            raise ValueError("bald_scores required")
        if features is None:
            raise ValueError("features required for diversity sampling")

        # Step 0: adaptive uncertainty threshold (Section 4.3.2)
        candidate_indices = apply_uncertainty_prefilter(
            candidate_indices, uncertainty_scores, budget, self.alpha
        )

        n_candidates = len(candidate_indices)
        budget = min(budget, n_candidates)

        if budget <= 0:
            return []

        # Step 1: Get BALD scores for candidates
        cand_bald = bald_scores[candidate_indices]

        # Step 2: Extract features for candidates
        cand_features = features[candidate_indices]

        # Step 3: Determine number of clusters
        # K = min(b_t, floor(|C_t| / beta))
        K = min(budget, max(1, n_candidates // self.beta))
        K = max(1, min(K, n_candidates))

        # Step 4: k-means clustering on feature embeddings
        if K >= n_candidates:
            # No clustering needed, just rank by BALD
            top_indices = np.argsort(cand_bald)[::-1][:budget]
            return [candidate_indices[i] for i in top_indices]

        try:
            kmeans = KMeans(
                n_clusters=K,
                random_state=self.seed,
                n_init=10,
                max_iter=100
            )
            cluster_labels = kmeans.fit_predict(cand_features)
        except Exception:
            # Fallback to BALD-only if clustering fails
            top_indices = np.argsort(cand_bald)[::-1][:budget]
            return [candidate_indices[i] for i in top_indices]

        # Step 5: Proportional selection from each cluster
        # b_t^(j) = floor(b_t * |G_j| / |C_t|)
        selected = []
        cluster_ids = np.unique(cluster_labels)

        # Compute proportional budget per cluster
        cluster_budgets = {}
        remaining_budget = budget
        for j in cluster_ids:
            cluster_size = (cluster_labels == j).sum()
            b_j = int(np.floor(budget * cluster_size / n_candidates))
            b_j = max(b_j, 0)
            cluster_budgets[j] = b_j
            remaining_budget -= b_j

        # Distribute remaining budget greedily to clusters with highest max BALD
        if remaining_budget > 0:
            cluster_max_bald = {}
            for j in cluster_ids:
                mask = cluster_labels == j
                if mask.any():
                    cluster_max_bald[j] = cand_bald[mask].max()
                else:
                    cluster_max_bald[j] = 0.0

            sorted_clusters = sorted(
                cluster_max_bald.keys(),
                key=lambda j: cluster_max_bald[j],
                reverse=True
            )
            for j in sorted_clusters:
                if remaining_budget <= 0:
                    break
                cluster_budgets[j] += 1
                remaining_budget -= 1

        # Step 6: Select top-scoring from each cluster
        for j in cluster_ids:
            cluster_mask = cluster_labels == j
            cluster_positions = np.where(cluster_mask)[0]
            cluster_bald_scores = cand_bald[cluster_mask]

            b_j = min(cluster_budgets.get(j, 0), len(cluster_positions))
            if b_j <= 0:
                continue

            top_in_cluster = np.argsort(cluster_bald_scores)[::-1][:b_j]
            for ti in top_in_cluster:
                global_idx = candidate_indices[cluster_positions[ti]]
                selected.append(global_idx)

        # Ensure we don't exceed budget
        selected = selected[:budget]

        # Record provenance for the diversity diagnostics.
        position_of = {idx: pos for pos, idx in enumerate(candidate_indices)}
        self.last_selection_info = {
            "cluster_labels": cluster_labels.tolist(),
            "candidate_indices": list(candidate_indices),
            "selected_positions": [
                position_of[i] for i in selected if i in position_of
            ],
            "n_clusters": int(K),
            "cluster_budgets": {int(k): int(v) for k, v in cluster_budgets.items()},
        }

        return selected


# get_strategy() defined below with extended support


# ---------------------------------------------------------------------------
# Extended strategy registry (BSB / PSB / BSB+Diversity)
# ---------------------------------------------------------------------------
# Import here to keep get_strategy() as the single entry point.
# The extended module lazy-imports to avoid circular dependencies.

def get_strategy(name: str, config: dict):
    """
    Factory for all acquisition strategies (original + extended).
    Extended strategies (bsb, psb, bsb_diversity) are loaded from
    src.uncertainty_extended to keep this module dependency-free.
    """
    original = {
        "random":       lambda: RandomSampling(seed=config.get("seed", 42)),
        "deterministic":lambda: DeterministicUncertaintySampling(),
        "bald_only":    lambda: BALDOnlySampling(alpha=config.get("alpha", 0.1)),
        "bald_diversity":lambda: BALDDiversitySampling(
            alpha=config.get("alpha", 0.1),
            beta=config.get("beta", 5),
            seed=config.get("seed", 42),
        ),
    }
    if name in original:
        return original[name]()

    # Extended strategies
    try:
        from src.uncertainty_extended import get_extended_strategy
        return get_extended_strategy(name, config)
    except ImportError as e:
        raise ValueError(
            f"Unknown strategy '{name}' and could not load extended strategies: {e}"
        )
