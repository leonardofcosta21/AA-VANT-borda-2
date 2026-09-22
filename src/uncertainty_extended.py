"""
Extended Uncertainty Estimators and Acquisition Strategies.

Adds three new components to the existing pipeline:

  1. BSB/PSB Acquisition Strategy
     Best-and-Second-Best / Past-Second-Best criterion.
     Reference: Ribeiro et al. (2026), JBCS v.32 n.1.
     Selects samples where the margin between the top-1 and top-2
     class confidences is smallest (most ambiguous predictions).
     Compatible with both softmax output and MC Dropout.

  2. Conformal Prediction Uncertainty
     Black-box, distribution-free uncertainty quantification.
     Calibrates a nonconformity threshold on a held-out set and
     produces prediction sets whose coverage is guaranteed at level
     1 - epsilon. Used as an alternative to MC Dropout for H2/C2.
     No additional training required.

  3. SWAG Uncertainty Estimation
     Stochastic Weight Averaging - Gaussian (Maddox et al. 2019).
     Fits a Gaussian over the last K SGD iterates of training to
     approximate the posterior. Cheaper than full ensembles while
     more expressive than MC Dropout.

Integration with existing pipeline:
  - All three produce the same interface as MCDropoutEstimator:
      { 'variance', 'bald', 'mean_conf', 'det_scores' }
  - BSBSampling / PSBSampling integrate into get_strategy() in
    acquisition.py alongside Random/Det/BALD/BALD+Div.
  - Pass strategy name 'bsb', 'psb', or 'bsb_diversity' to
    run_experiment.py / run_multi_seed.py.

Usage:
    from src.uncertainty_extended import (
        ConformalUncertaintyEstimator,
        SWAGEstimator,
    )
    from src.acquisition_extended import get_extended_strategy

    # Conformal
    cp_est = ConformalUncertaintyEstimator(base_estimator, epsilon=0.1)
    cp_est.calibrate(calibration_image_paths)
    scores = cp_est.compute_mc_uncertainty(image_paths)

    # SWAG
    swag = SWAGEstimator(model_path, K=20, device='cuda')
    swag.collect_swag_statistics(yaml_path, epochs=10)
    scores = swag.compute_mc_uncertainty(image_paths)

    # BSB strategy
    strategy = get_extended_strategy('bsb_diversity', config)
    selected = strategy.select(candidates, budget, bsb_scores=scores, features=feats)
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from copy import deepcopy

# Re-export existing components so callers only need one import
# (lazy import to avoid issues if ultralytics/tqdm not installed in test env)
try:
    from src.mc_dropout import MCDropoutEstimator, enable_mc_dropout, disable_mc_dropout
except ImportError:
    MCDropoutEstimator = None
    enable_mc_dropout = None
    disable_mc_dropout = None

from src.acquisition import (
    AcquisitionStrategy, RandomSampling, DeterministicUncertaintySampling,
    BALDOnlySampling, BALDDiversitySampling, get_strategy,
)

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    tqdm = lambda x, **kw: x
    HAS_TQDM = False


# =============================================================================
# 1. BSB / PSB ACQUISITION STRATEGIES
# =============================================================================

class BSBSampling(AcquisitionStrategy):
    """
    Best-and-Second-Best (BSB) margin criterion.

    For each image, computes the margin between the highest and
    second-highest class confidence across all detections:

        bsb(x) = 1 - (conf_1 - conf_2)

    A small margin means the model is ambiguous between two classes
    → high information content. This is equivalent to the margin
    criterion but applied at detection level.

    Compatible with single-pass softmax (no MC Dropout needed),
    making it faster than BALD while often competitive in practice.

    Reference: Ribeiro et al. (2026). Portfolio-based Active Learning
    with Gaussian Processes for Vulnerabilities Risk Classification.
    Journal of the Brazilian Computer Society, v.32, n.1.
    """

    def __init__(self, alpha: float = 0.1):
        super().__init__("bsb")
        self.alpha = alpha

    def select(
        self,
        candidate_indices: List[int],
        budget: int,
        bsb_scores: np.ndarray = None,
        det_scores: np.ndarray = None,
        **kwargs
    ) -> List[int]:
        """
        Args:
            candidate_indices: pool indices
            budget: annotation budget
            bsb_scores: BSB margin scores (higher = more ambiguous)
                        Computed by BSBScorer.compute_bsb_scores()
            det_scores: fallback deterministic scores if bsb_scores absent
        """
        scores = bsb_scores if bsb_scores is not None else det_scores
        if scores is None:
            raise ValueError("bsb_scores (or det_scores) required for BSB sampling")

        budget = min(budget, len(candidate_indices))

        # Optional quantile threshold
        if len(candidate_indices) > budget * 3:
            cand_s = scores[candidate_indices]
            threshold = np.quantile(cand_s, 1 - self.alpha)
            filtered = [i for i in candidate_indices if scores[i] >= threshold]
            if len(filtered) >= budget:
                candidate_indices = filtered

        cand_scores = scores[candidate_indices]
        top_idx = np.argsort(cand_scores)[::-1][:budget]
        return [candidate_indices[i] for i in top_idx]


class PSBSampling(AcquisitionStrategy):
    """
    Past-Second-Best (PSB) criterion.

    Extends BSB by tracking how the second-best confidence evolves
    across cycles. Samples are prioritised if their second-best
    confidence has *increased* since the last cycle (suggesting the
    model is becoming more uncertain about them as it learns).

        psb(x, t) = conf_2(x, t) - conf_2(x, t-1)

    On cycle 0 (no history), falls back to BSB.

    This criterion is particularly suited to streaming/continual
    settings where the model changes across cycles — directly relevant
    to the UAV-edge adaptive pipeline.
    """

    def __init__(self, alpha: float = 0.1):
        super().__init__("psb")
        self.alpha = alpha
        self._prev_second_best: Optional[np.ndarray] = None

    def update_history(self, second_best_scores: np.ndarray):
        """Call at the end of each cycle with the current second-best scores."""
        self._prev_second_best = second_best_scores.copy()

    def select(
        self,
        candidate_indices: List[int],
        budget: int,
        bsb_scores: np.ndarray = None,
        second_best_scores: np.ndarray = None,
        **kwargs
    ) -> List[int]:
        """
        Args:
            candidate_indices: pool indices
            budget: annotation budget
            bsb_scores: current BSB scores (used as fallback / cycle 0)
            second_best_scores: current second-best confidence per image
        """
        budget = min(budget, len(candidate_indices))

        if second_best_scores is not None and self._prev_second_best is not None:
            # PSB delta: increase in second-best confidence since last cycle
            n = min(len(second_best_scores), len(self._prev_second_best))
            psb_scores = np.zeros(len(second_best_scores))
            psb_scores[:n] = second_best_scores[:n] - self._prev_second_best[:n]
            scores = psb_scores
        else:
            # Cycle 0 fallback: use BSB
            scores = bsb_scores
            if scores is None:
                raise ValueError("bsb_scores required for PSB on cycle 0")

        # Optional threshold
        if len(candidate_indices) > budget * 3:
            cand_s = scores[candidate_indices]
            threshold = np.quantile(cand_s, 1 - self.alpha)
            filtered = [i for i in candidate_indices if scores[i] >= threshold]
            if len(filtered) >= budget:
                candidate_indices = filtered

        cand_scores = scores[candidate_indices]
        top_idx = np.argsort(cand_scores)[::-1][:budget]
        return [candidate_indices[i] for i in top_idx]


class BSBDiversitySampling(AcquisitionStrategy):
    """
    BSB + k-means cluster-then-select (mirrors BALDDiversitySampling).

    Replaces BALD with BSB as the informativeness criterion while
    keeping the same diversity mechanism. Allows direct comparison:

      bald_diversity  vs  bsb_diversity  (H3 extended comparison)

    Useful for the portfolio experiment (see Section 9.2 of the
    thesis adjustment plan): if BSB+Div ≈ BALD+Div in performance,
    BSB+Div is preferred due to lower compute cost (no MC Dropout).
    """

    def __init__(self, alpha: float = 0.1, beta: int = 5, seed: int = 42):
        super().__init__("bsb_diversity")
        self.alpha = alpha
        self.beta = beta
        self.seed = seed

    def select(
        self,
        candidate_indices: List[int],
        budget: int,
        bsb_scores: np.ndarray = None,
        features: np.ndarray = None,
        **kwargs
    ) -> List[int]:
        if bsb_scores is None:
            raise ValueError("bsb_scores required for BSB+Diversity sampling")
        if features is None or not HAS_SKLEARN:
            # Fallback to BSB-only
            return BSBSampling(self.alpha).select(
                candidate_indices, budget, bsb_scores=bsb_scores
            )

        # Quantile threshold
        if len(candidate_indices) > budget * 3:
            cand_s = bsb_scores[candidate_indices]
            threshold = np.quantile(cand_s, 1 - self.alpha)
            filtered = [i for i in candidate_indices if bsb_scores[i] >= threshold]
            if len(filtered) >= budget:
                candidate_indices = filtered

        n = len(candidate_indices)
        budget = min(budget, n)
        if budget <= 0:
            return []

        cand_bsb  = bsb_scores[candidate_indices]
        cand_feat = features[candidate_indices]
        K = min(budget, max(1, n // self.beta))
        K = max(1, min(K, n))

        if K >= n:
            return [candidate_indices[i] for i in np.argsort(cand_bsb)[::-1][:budget]]

        try:
            km = KMeans(n_clusters=K, random_state=self.seed, n_init=10, max_iter=100)
            labels = km.fit_predict(cand_feat)
        except Exception:
            return [candidate_indices[i] for i in np.argsort(cand_bsb)[::-1][:budget]]

        selected = []
        cluster_ids = np.unique(labels)
        budgets = {}
        rem = budget
        for j in cluster_ids:
            b_j = int(np.floor(budget * (labels == j).sum() / n))
            budgets[j] = b_j
            rem -= b_j
        # Distribute remainder
        if rem > 0:
            max_bsb = {j: cand_bsb[labels == j].max() for j in cluster_ids}
            for j in sorted(max_bsb, key=max_bsb.get, reverse=True):
                if rem <= 0:
                    break
                budgets[j] += 1
                rem -= 1

        for j in cluster_ids:
            pos = np.where(labels == j)[0]
            b_j = min(budgets.get(j, 0), len(pos))
            if b_j <= 0:
                continue
            top = np.argsort(cand_bsb[pos])[::-1][:b_j]
            for t in top:
                selected.append(candidate_indices[pos[t]])

        return selected[:budget]


# =============================================================================
# BSB Score Computation (single-pass, no MC Dropout)
# =============================================================================

class BSBScorer:
    """
    Computes BSB/PSB scores from single deterministic forward passes.

    For each image, runs one inference pass and extracts:
      - best confidence:        conf_1 = max score across all detections
      - second-best confidence: conf_2 = second distinct score
      - BSB margin:             1 - (conf_1 - conf_2)
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.05,   # low threshold to see near-miss detections
        iou_threshold:  float = 0.45,
        img_size:       int   = 640,
        device:         str   = "auto",
    ):
        if not HAS_YOLO:
            raise ImportError("ultralytics required: pip install ultralytics")
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self.img_size       = img_size
        self.yolo           = YOLO(model_path)

    def compute_bsb_scores(
        self,
        image_paths: List[str],
        batch_size: int = 16,
    ) -> Dict[str, np.ndarray]:
        """
        Compute BSB margin and second-best confidence for each image.

        Returns dict:
            'bsb_scores':     1 - (conf_1 - conf_2)  ∈ [0, 1]
            'second_best':    conf_2 per image
            'best':           conf_1 per image
            'det_scores':     1 - conf_1  (for fallback compatibility)
        """
        n = len(image_paths)
        bsb    = np.zeros(n)
        best   = np.zeros(n)
        second = np.zeros(n)

        for i in range(0, n, batch_size):
            batch = image_paths[i:i + batch_size]
            results = self.yolo.predict(
                source=batch,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                verbose=False,
                device=self.device,
            )
            for b_idx, r in enumerate(results):
                g = i + b_idx
                if r.boxes is None or len(r.boxes) == 0:
                    bsb[g]    = 1.0   # max uncertainty: no detection
                    best[g]   = 0.0
                    second[g] = 0.0
                    continue

                confs = r.boxes.conf.cpu().numpy()
                confs_sorted = np.sort(confs)[::-1]

                c1 = float(confs_sorted[0])
                c2 = float(confs_sorted[1]) if len(confs_sorted) > 1 else 0.0

                best[g]   = c1
                second[g] = c2
                bsb[g]    = 1.0 - (c1 - c2)   # small margin → high score

        return {
            "bsb_scores":  bsb,
            "second_best": second,
            "best":        best,
            "det_scores":  1.0 - best,
        }


# =============================================================================
# 2. CONFORMAL PREDICTION UNCERTAINTY ESTIMATOR
# =============================================================================

class ConformalUncertaintyEstimator:
    """
    Conformal Prediction for black-box uncertainty quantification.

    Provides distribution-free coverage guarantees without modifying
    the base model. Uses the nonconformity score:

        alpha(x) = 1 - max_i p_i(x)     (least ambiguous = most conforming)

    After calibration on a held-out set of size n_cal, the prediction
    set for a new image at level 1 - epsilon contains all classes
    whose softmax score exceeds the (1 - epsilon) quantile of the
    calibration nonconformity scores.

    For object detection, we apply conformal prediction at the
    image level: an image is "uncertain" if its nonconformity score
    exceeds the calibrated threshold q_hat.

    Reference: Angelopoulos & Bates (2023). Conformal Prediction:
    A Gentle Introduction. Foundations and Trends in ML.
    Package: https://github.com/leoandeol/cods (as suggested in thesis plan)
    """

    def __init__(
        self,
        base_estimator: "MCDropoutEstimator",
        epsilon: float = 0.1,
    ):
        """
        Args:
            base_estimator: fitted MCDropoutEstimator (or BSBScorer)
            epsilon: miscoverage level (default 0.1 → 90% coverage)
        """
        self.base = base_estimator
        self.epsilon = epsilon
        self.q_hat: Optional[float] = None   # calibrated threshold
        self._calibrated = False

    def calibrate(self, calibration_paths: List[str]):
        """
        Calibrate the conformal threshold on a held-out calibration set.

        Computes nonconformity scores on calibration images, then sets
        q_hat = quantile(scores, ceil((n+1)(1-epsilon)) / n).

        Must be called before compute_mc_uncertainty().
        """
        print(f"  [Conformal] Calibrating on {len(calibration_paths)} images "
              f"(ε={self.epsilon})...")

        # Get nonconformity scores from base estimator
        preds = self.base.compute_deterministic_predictions(calibration_paths)
        scores = np.array([
            1.0 - (p["scores"].max() if len(p["scores"]) > 0 else 0.0)
            for p in preds
        ])

        n = len(scores)
        level = np.ceil((n + 1) * (1 - self.epsilon)) / n
        level = min(level, 1.0)
        self.q_hat = float(np.quantile(scores, level))
        self._calibrated = True

        coverage = (scores <= self.q_hat).mean()
        print(f"  [Conformal] q_hat={self.q_hat:.4f} | "
              f"empirical coverage={coverage:.3f} (target={1-self.epsilon:.2f})")

    def compute_mc_uncertainty(
        self,
        image_paths: List[str],
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Compute conformal nonconformity scores.

        Returns same interface as MCDropoutEstimator.compute_mc_uncertainty():
            'variance':   nonconformity score (replaces epistemic variance)
            'bald':       binary: 1 if score > q_hat (uncertain), else 0
            'mean_conf':  mean confidence
            'det_scores': 1 - mean_conf
            'conformal_uncertain': bool array, True if above threshold
        """
        if not self._calibrated:
            print("  [Conformal] WARNING: not calibrated — running without threshold")

        preds = self.base.compute_deterministic_predictions(image_paths)
        n = len(image_paths)

        nonconf = np.zeros(n)
        mean_conf = np.zeros(n)

        for i, p in enumerate(preds):
            c1 = float(p["scores"].max()) if len(p["scores"]) > 0 else 0.0
            nonconf[i]   = 1.0 - c1
            mean_conf[i] = c1

        # Binary uncertainty flag
        if self._calibrated and self.q_hat is not None:
            uncertain = (nonconf > self.q_hat).astype(float)
        else:
            uncertain = nonconf

        return {
            "variance":             nonconf,
            "bald":                 uncertain,
            "mean_conf":            mean_conf,
            "det_scores":           1.0 - mean_conf,
            "conformal_uncertain":  uncertain.astype(bool),
            "q_hat":                np.full(n, self.q_hat or 0.0),
        }

    def coverage_on_test(self, test_paths: List[str]) -> float:
        """Empirical coverage on a test set (should be ≥ 1 - epsilon)."""
        if not self._calibrated:
            return 0.0
        result = self.compute_mc_uncertainty(test_paths)
        return float((result["variance"] <= self.q_hat).mean())


# =============================================================================
# 3. SWAG UNCERTAINTY ESTIMATOR
# =============================================================================

class SWAGEstimator:
    """
    Stochastic Weight Averaging - Gaussian (SWAG).

    Approximates the posterior over model weights by fitting a
    Gaussian to the trajectory of SGD iterates in the last K epochs
    of training:

        θ ~ N(θ_SWA, Σ_SWAG)

    where θ_SWA is the mean (SWA solution) and Σ_SWAG is the
    diagonal + low-rank covariance estimated from weight deviations.

    At inference time, samples S weight vectors from this Gaussian
    and averages predictions, producing calibrated uncertainty
    estimates without MC Dropout.

    Reference: Maddox et al. (2019). A Simple Baseline for Bayesian
    Deep Learning. NeurIPS 2019.

    Computational cost vs alternatives:
      MC Dropout:  1 model,  T inference passes
      SWAG:        1 model,  K training checkpoints, S inference passes
      Ensembles:   M models, 1 inference pass each
      MCMC/HMC:    1 model,  very long chain (prohibitive on edge)

    For edge deployment, SWAG is more expensive than MC Dropout at
    calibration time but comparable at inference if S is small (S=5-10).
    """

    def __init__(
        self,
        model_path: str,
        K: int   = 20,    # number of SGD iterates to collect
        S: int   = 10,    # number of weight samples at inference
        rank: int = 5,    # low-rank covariance approximation
        conf_threshold: float = 0.25,
        iou_threshold:  float = 0.45,
        img_size:       int   = 640,
        device:         str   = "auto",
    ):
        if not HAS_YOLO:
            raise ImportError("ultralytics required: pip install ultralytics")

        self.K    = K
        self.S    = S
        self.rank = rank
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self.img_size       = img_size

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.base_model_path = model_path
        self.yolo = YOLO(model_path)

        # SWAG statistics (populated by collect_swag_statistics)
        self._theta_swa: Optional[torch.Tensor]       = None  # mean
        self._theta_sq:  Optional[torch.Tensor]       = None  # mean of squares (diagonal var)
        self._deviations: List[torch.Tensor]           = []   # low-rank deviations
        self._fitted = False

    # ------------------------------------------------------------------
    # Calibration (collect weight statistics during fine-tuning)
    # ------------------------------------------------------------------

    def collect_swag_statistics(
        self,
        yaml_path: str,
        epochs: int  = 10,
        lr:     float = 1e-3,
    ):
        """
        Fine-tune the model for `epochs` epochs, collecting weight
        snapshots in the last K epochs to build SWAG statistics.

        In practice, call this once after the initial L0 training
        and again after each major update cycle if compute allows.
        For a lightweight pipeline, calling it once at the end of
        initial training is sufficient.

        Args:
            yaml_path: path to YOLO data.yaml for fine-tuning
            epochs:    total fine-tuning epochs (SWAG collects last K)
            lr:        learning rate for fine-tuning
        """
        print(f"  [SWAG] Collecting weight statistics "
              f"({epochs} epochs, collecting last {self.K})...")

        model = self.yolo.model
        params = [p for p in model.parameters() if p.requires_grad]
        n_params = sum(p.numel() for p in params)

        # Flatten all trainable params into a single vector helper
        def get_flat_params() -> torch.Tensor:
            return torch.cat([p.data.view(-1) for p in params]).cpu()

        # Run fine-tuning via ultralytics trainer
        # We hook into the epoch-end callback to collect snapshots
        snapshots: List[torch.Tensor] = []

        def on_epoch_end(trainer):
            current_epoch = trainer.epoch + 1
            total         = trainer.epochs
            collect_from  = max(1, total - self.K + 1)
            if current_epoch >= collect_from:
                snapshots.append(get_flat_params().clone())
                print(f"    [SWAG] Snapshot {len(snapshots)}/{self.K} "
                      f"(epoch {current_epoch}/{total})")

        self.yolo.add_callback("on_train_epoch_end", on_epoch_end)

        try:
            self.yolo.train(
                data=yaml_path,
                epochs=epochs,
                lr0=lr,
                imgsz=self.img_size,
                batch=8,
                device=self.device,
                verbose=False,
                exist_ok=True,
            )
        finally:
            self.yolo.reset_callbacks()

        if not snapshots:
            print("  [SWAG] No snapshots collected — check epoch count vs K.")
            return

        # Build SWAG statistics from snapshots
        stack = torch.stack(snapshots)          # (K, n_params)
        self._theta_swa = stack.mean(dim=0)     # mean weights
        self._theta_sq  = (stack ** 2).mean(dim=0)  # mean of squares

        # Low-rank deviation matrix (K × n_params, capped at rank)
        deviations = stack - self._theta_swa.unsqueeze(0)
        k_eff = min(self.rank, len(snapshots))
        self._deviations = [deviations[i] for i in range(k_eff)]
        self._fitted = True

        diag_var = torch.clamp(self._theta_sq - self._theta_swa ** 2, min=1e-30)
        print(f"  [SWAG] Fitted. mean param norm={self._theta_swa.norm():.2f} | "
              f"mean diag var={diag_var.mean():.2e} | "
              f"low-rank={k_eff}")

    # ------------------------------------------------------------------
    # Weight sampling
    # ------------------------------------------------------------------

    def _sample_weights(self) -> torch.Tensor:
        """
        Sample one weight vector from the SWAG posterior:

            θ ~ θ_SWA  +  (1/√2) * diag(Σ)^{1/2} * ε_1
                        +  (1/(√(2(K-1)))) * D * ε_2

        where ε_1 ~ N(0, I_d), ε_2 ~ N(0, I_K).
        """
        if not self._fitted:
            raise RuntimeError("Call collect_swag_statistics() first.")

        diag_var = torch.clamp(self._theta_sq - self._theta_swa ** 2, min=1e-30)
        diag_std = diag_var.sqrt()

        # Diagonal component
        eps1 = torch.randn_like(self._theta_swa)
        theta = self._theta_swa + (1.0 / (2 ** 0.5)) * diag_std * eps1

        # Low-rank component
        if self._deviations:
            K_eff = len(self._deviations)
            eps2  = torch.randn(K_eff)
            D     = torch.stack(self._deviations)  # (K, n_params)
            lr_component = (D.T @ eps2) / ((2 * (K_eff - 1)) ** 0.5 + 1e-10)
            theta = theta + lr_component.cpu()

        return theta

    def _set_weights(self, flat_theta: torch.Tensor):
        """Load a flat weight vector into the model."""
        model  = self.yolo.model
        params = [p for p in model.parameters() if p.requires_grad]
        offset = 0
        for p in params:
            n = p.numel()
            p.data.copy_(
                flat_theta[offset:offset + n]
                .view_as(p)
                .to(p.device)
            )
            offset += n

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def compute_mc_uncertainty(
        self,
        image_paths: List[str],
        batch_size: int = 8,
    ) -> Dict[str, np.ndarray]:
        """
        Estimate uncertainty by averaging predictions over S weight samples.

        Returns same interface as MCDropoutEstimator:
            'variance', 'bald', 'mean_conf', 'det_scores'
        """
        if not self._fitted:
            print("  [SWAG] Not fitted — falling back to deterministic uncertainty.")
            base = MCDropoutEstimator(
                self.base_model_path, T=1, device=self.device,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold,
                img_size=self.img_size,
            )
            return base.compute_deterministic_predictions.__self__\
                       .compute_mc_uncertainty(image_paths)

        n = len(image_paths)
        # Save original weights
        params = [p for p in self.yolo.model.parameters() if p.requires_grad]
        original_flat = torch.cat([p.data.view(-1) for p in params]).cpu().clone()

        all_confs: List[np.ndarray] = [np.array([]) for _ in range(n)]
        sample_confs: List[List[np.ndarray]] = [[] for _ in range(n)]

        try:
            for s in range(self.S):
                theta_s = self._sample_weights()
                self._set_weights(theta_s)

                for i in range(0, n, batch_size):
                    batch = image_paths[i:i + batch_size]
                    results = self.yolo.predict(
                        source=batch,
                        conf=0.05,
                        iou=self.iou_threshold,
                        imgsz=self.img_size,
                        verbose=False,
                        device=self.device,
                    )
                    for b_idx, r in enumerate(results):
                        g = i + b_idx
                        c = r.boxes.conf.cpu().numpy() if (r.boxes and len(r.boxes)) else np.array([0.0])
                        sample_confs[g].append(c)
        finally:
            # Restore original weights
            self._set_weights(original_flat)

        # Aggregate
        variance   = np.zeros(n)
        bald       = np.zeros(n)
        mean_conf  = np.zeros(n)
        eps = 1e-10

        for g in range(n):
            max_c = np.array([
                sc.max() if len(sc) > 0 else 0.0
                for sc in sample_confs[g]
            ])
            mu           = max_c.mean()
            mean_conf[g] = mu
            variance[g]  = max_c.var()

            # BALD
            p  = np.clip(mu, eps, 1 - eps)
            h_mean = -p * np.log(p) - (1 - p) * np.log(1 - p)
            h_each = [
                -(np.clip(c, eps, 1-eps) * np.log(np.clip(c, eps, 1-eps))
                  + (1-np.clip(c, eps, 1-eps)) * np.log(1-np.clip(c, eps, 1-eps)))
                for c in max_c
            ]
            bald[g] = h_mean - np.mean(h_each)

        return {
            "variance":   variance,
            "bald":       bald,
            "mean_conf":  mean_conf,
            "det_scores": 1.0 - mean_conf,
        }

    def compute_deterministic_predictions(self, image_paths: List[str]) -> List[Dict]:
        """Passthrough to base YOLO deterministic predictions."""
        results_list = []
        for i in range(0, len(image_paths), 16):
            batch = image_paths[i:i + 16]
            results = self.yolo.predict(
                source=batch,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                verbose=False,
                device=self.device,
            )
            for r in results:
                results_list.append({
                    "boxes":   r.boxes.xyxy.cpu().numpy()  if r.boxes else np.array([]),
                    "scores":  r.boxes.conf.cpu().numpy()  if r.boxes else np.array([]),
                    "classes": r.boxes.cls.cpu().numpy()   if r.boxes else np.array([]),
                })
        return results_list

    def extract_features(self, image_paths: List[str], **kwargs) -> np.ndarray:
        """Feature extraction passthrough (uses SWA mean weights)."""
        if self._fitted:
            self._set_weights(self._theta_swa)
        base = MCDropoutEstimator(
            self.base_model_path, T=1, device=self.device,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            img_size=self.img_size,
        )
        return base.extract_features(image_paths)


# =============================================================================
# Extended get_strategy factory
# =============================================================================

def get_extended_strategy(name: str, config: dict) -> AcquisitionStrategy:
    """
    Extended factory that includes BSB/PSB strategies in addition to
    the original four (random / deterministic / bald_only / bald_diversity).

    Pass to run_experiment.py via --strategy bsb / psb / bsb_diversity.
    """
    base_strategies = {
        "random", "deterministic", "bald_only", "bald_diversity"
    }
    if name in base_strategies:
        return get_strategy(name, config)

    extended = {
        "bsb": lambda: BSBSampling(alpha=config.get("alpha", 0.1)),
        "psb": lambda: PSBSampling(alpha=config.get("alpha", 0.1)),
        "bsb_diversity": lambda: BSBDiversitySampling(
            alpha=config.get("alpha", 0.1),
            beta=config.get("beta", 5),
            seed=config.get("seed", 42),
        ),
    }

    if name not in extended:
        raise ValueError(
            f"Unknown strategy '{name}'. "
            f"Available: {sorted(base_strategies | set(extended.keys()))}"
        )
    return extended[name]()


# =============================================================================
# Uncertainty estimator factory
# =============================================================================

def get_uncertainty_estimator(
    method: str,
    model_path: str,
    config: dict,
    calibration_paths: Optional[List[str]] = None,
) -> "MCDropoutEstimator":
    """
    Factory for uncertainty estimators.

    Args:
        method: 'mc_dropout' | 'conformal' | 'swag'
        model_path: path to YOLO weights
        config: experiment config dict
        calibration_paths: required for 'conformal'

    Returns estimator with .compute_mc_uncertainty() interface.
    """
    device = config.get("device", "auto")
    conf   = config.get("conf_threshold", 0.25)
    iou    = config.get("iou_threshold",  0.45)
    imgsz  = config.get("img_size", 640)
    T      = config.get("T", 8)
    dr     = config.get("dropout_rate", 0.1)

    if method == "mc_dropout":
        return MCDropoutEstimator(
            model_path, T=T, dropout_rate=dr,
            conf_threshold=conf, iou_threshold=iou,
            img_size=imgsz, device=device,
        )

    elif method == "conformal":
        base = MCDropoutEstimator(
            model_path, T=1, dropout_rate=0.0,
            conf_threshold=conf, iou_threshold=iou,
            img_size=imgsz, device=device,
        )
        cp = ConformalUncertaintyEstimator(
            base, epsilon=config.get("epsilon", 0.1)
        )
        if calibration_paths:
            cp.calibrate(calibration_paths)
        else:
            print("  [WARNING] conformal estimator not calibrated (no calibration_paths)")
        return cp

    elif method == "swag":
        return SWAGEstimator(
            model_path,
            K=config.get("swag_K", 20),
            S=config.get("swag_S", 10),
            rank=config.get("swag_rank", 5),
            conf_threshold=conf, iou_threshold=iou,
            img_size=imgsz, device=device,
        )

    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose: mc_dropout | conformal | swag"
        )

# Alias para compatibilidade com a interface MCDropoutEstimator
def compute_mc_uncertainty(self, image_paths, **kwargs):
    result = self.compute_bsb_scores(image_paths)
    n = len(image_paths)
    return {
        "variance":    result["bsb_scores"],
        "bald":        result["bsb_scores"],
        "mean_conf":   result["best"],
        "det_scores":  result["det_scores"],
        "bsb_scores":  result["bsb_scores"],
        "second_best": result["second_best"],
    }

BSBScorer.compute_mc_uncertainty = compute_mc_uncertainty

def extract_features_bsb(self, image_paths, **kwargs):
    from src.mc_dropout import MCDropoutEstimator
    tmp = MCDropoutEstimator(
        model_path=self.yolo.ckpt_path if hasattr(self.yolo, 'ckpt_path') else "yolov8n.pt",
        T=1, dropout_rate=0.0,
        conf_threshold=self.conf_threshold,
        iou_threshold=self.iou_threshold,
        img_size=self.img_size,
        device=self.device,
    )
    return tmp.extract_features(image_paths)

BSBScorer.extract_features = extract_features_bsb
