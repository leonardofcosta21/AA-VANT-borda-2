"""
Deep Ensembles as an uncertainty estimator (board suggestion 9.1).

The board asked whether MC Dropout's cost is actually practical at the
edge and, if not, whether Deep Ensembles are worth their infrastructure
cost instead. Answering that needs the alternative implemented and
measured on the same footing, not argued from the literature.

What this gives the thesis
--------------------------
A drop-in estimator with the same interface as ``MCDropoutEstimator``, so
the comparison runs through the identical pipeline and any difference is
attributable to the estimator alone. The cost side is explicit in
``cost_profile()``: M models means M times the weights in storage, M
times the training time, and M forward passes per candidate, against MC
Dropout's single set of weights and T passes.

The disagreement measure
------------------------
Ensemble members are independent networks, so their epistemic
uncertainty is the disagreement between them, estimated exactly as in the
MC Dropout path (predictive variance and the BALD mutual-information
approximation over per-image confidence summaries). Using the same
estimator for both means the two methods' numbers live on the same
scale, which is the point of the comparison.

Members are trained from different random seeds on the same labelled set.
That is the standard recipe (Lakshminarayanan et al., 2017): random
initialisation and data-order shuffling supply enough diversity without
needing bootstrap resampling, which would shrink each member's training
set and confound the comparison with a data-size effect.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = ["DeepEnsembleEstimator", "train_ensemble_members"]


def train_ensemble_members(
    yaml_path: str,
    n_members: int = 5,
    base_seed: int = 42,
    model_name: str = "yolov8n.pt",
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 0.01,
    img_size: int = 640,
    device: str = "auto",
    output_dir: str = "./runs/ensemble",
) -> Dict:
    """Train ``n_members`` independent detectors and report the cost.

    Returns the member weight paths plus the wall-clock and disk cost, so
    the feasibility argument in the thesis rests on measured numbers
    rather than estimates.
    """
    from ultralytics import YOLO

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    weights: List[str] = []
    times: List[float] = []

    for member in range(n_members):
        seed = base_seed + member * 1000  # widely separated seeds
        start = time.time()
        model = YOLO(model_name)
        model.train(
            data=yaml_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            lr0=lr,
            seed=seed,
            project=str(out_root),
            name=f"member_{member}",
            exist_ok=True,
            verbose=False,
            plots=False,
            device=None if device == "auto" else device,
        )
        elapsed = time.time() - start
        save_dir = Path(model.trainer.save_dir)
        weight = save_dir / "weights" / "best.pt"
        if not weight.exists():
            weight = save_dir / "weights" / "last.pt"
        weights.append(str(weight))
        times.append(elapsed)

    total_bytes = sum(
        Path(w).stat().st_size for w in weights if Path(w).exists()
    )
    return {
        "weights": weights,
        "n_members": n_members,
        "train_time_s_per_member": times,
        "train_time_s_total": float(sum(times)),
        "weights_total_mb": total_bytes / 1e6,
    }


class DeepEnsembleEstimator:
    """Uncertainty from the disagreement of M independently trained models.

    Interface-compatible with ``MCDropoutEstimator``: exposes
    ``compute_mc_uncertainty``, ``compute_deterministic_predictions`` and
    ``extract_features``, so ``run_uncertainty_comparison`` and the
    acquisition strategies use it without modification.
    """

    def __init__(
        self,
        member_paths: Sequence[str],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        img_size: int = 640,
        device: str = "auto",
        scoring_conf: float = 0.1,
    ):
        from ultralytics import YOLO

        if not member_paths:
            raise ValueError("DeepEnsembleEstimator needs at least one member")

        self.member_paths = list(member_paths)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.scoring_conf = scoring_conf

        if device == "auto":
            try:
                import torch

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self.device = "cpu"
        else:
            self.device = device

        self.members = [YOLO(p) for p in self.member_paths]
        # The first member is the deployed model: deployment latency must
        # reflect a single network, since an ensemble is used only during
        # acquisition, off the real-time path.
        self.yolo = self.members[0]
        self.T = len(self.members)

    # -- uncertainty -----------------------------------------------------
    def compute_mc_uncertainty(
        self, image_paths: Sequence[str], batch_size: int = 8
    ) -> Dict[str, np.ndarray]:
        """Per-image variance and BALD across ensemble members."""
        paths = list(image_paths)
        n = len(paths)
        variances = np.zeros(n)
        bald = np.zeros(n)
        mean_conf = np.zeros(n)
        det_scores = np.zeros(n)

        for start in range(0, n, batch_size):
            batch = paths[start : start + batch_size]
            # member_confs[m][b] = max confidence of member m on image b
            member_confs: List[List[float]] = []
            for model in self.members:
                results = model.predict(
                    source=batch,
                    conf=self.scoring_conf,
                    iou=self.iou_threshold,
                    imgsz=self.img_size,
                    verbose=False,
                    device=self.device,
                )
                row = []
                for r in results:
                    if r.boxes is not None and len(r.boxes) > 0:
                        row.append(float(r.boxes.conf.cpu().numpy().max()))
                    else:
                        row.append(0.0)
                member_confs.append(row)

            confs = np.asarray(member_confs, dtype=float)  # (M, B)
            for b in range(confs.shape[1]):
                idx = start + b
                column = confs[:, b]
                variances[idx] = float(column.var())
                mean_conf[idx] = float(column.mean())
                bald[idx] = _bald_from_confidences(column)
                det_scores[idx] = 1.0 - mean_conf[idx]

        return {
            "variance": variances,
            "bald": bald,
            "mean_conf": mean_conf,
            "det_scores": det_scores,
        }

    # -- deployment path -------------------------------------------------
    def compute_deterministic_predictions(
        self, image_paths: Sequence[str]
    ) -> List[Dict]:
        """Single-model predictions, as deployed."""
        out: List[Dict] = []
        paths = list(image_paths)
        for start in range(0, len(paths), 16):
            batch = paths[start : start + 16]
            results = self.yolo.predict(
                source=batch,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                verbose=False,
                device=self.device,
            )
            for r in results:
                out.append(
                    {
                        "boxes": r.boxes.xyxy.cpu().numpy()
                        if r.boxes is not None
                        else np.array([]),
                        "scores": r.boxes.conf.cpu().numpy()
                        if r.boxes is not None
                        else np.array([]),
                        "classes": r.boxes.cls.cpu().numpy()
                        if r.boxes is not None
                        else np.array([]),
                    }
                )
        return out

    def extract_features(
        self, image_paths: Sequence[str], batch_size: int = 16
    ) -> np.ndarray:
        """Embeddings from the first member, for diversity clustering.

        Using one member keeps the feature space consistent across cycles;
        averaging embeddings from independently initialised networks would
        mix incompatible coordinate systems.
        """
        from src.mc_dropout import MCDropoutEstimator

        proxy = MCDropoutEstimator.__new__(MCDropoutEstimator)
        proxy.yolo = self.yolo
        proxy.img_size = self.img_size
        proxy.device = self.device
        proxy.conf_threshold = self.conf_threshold
        proxy.iou_threshold = self.iou_threshold
        return MCDropoutEstimator.extract_features(
            proxy, list(image_paths), batch_size=batch_size
        )

    # -- cost ------------------------------------------------------------
    def cost_profile(self, mc_dropout_T: Optional[int] = None) -> Dict:
        """Storage and compute cost, for the feasibility comparison."""
        sizes = []
        for p in self.member_paths:
            try:
                sizes.append(Path(p).stat().st_size)
            except Exception:
                continue
        total_mb = sum(sizes) / 1e6 if sizes else None
        profile = {
            "n_members": len(self.members),
            "weights_total_mb": total_mb,
            "weights_per_member_mb": (total_mb / len(sizes)) if sizes else None,
            "forward_passes_per_candidate": len(self.members),
            "models_resident_in_memory": len(self.members),
        }
        if mc_dropout_T:
            profile["vs_mc_dropout"] = {
                "mc_dropout_T": mc_dropout_T,
                "storage_ratio": len(self.members),
                "acquisition_pass_ratio": len(self.members) / mc_dropout_T,
                "training_runs_ratio": len(self.members),
            }
        return profile


def _bald_from_confidences(confidences: np.ndarray) -> float:
    """BALD mutual information from per-member confidence summaries.

    Identical to the MC Dropout formulation so the two estimators produce
    comparable scores: entropy of the mean prediction minus the mean of
    the per-member entropies, treating the max confidence as a Bernoulli
    detection probability.
    """
    eps = 1e-10
    p_mean = float(np.clip(confidences.mean(), eps, 1 - eps))
    h_mean = -p_mean * np.log(p_mean) - (1 - p_mean) * np.log(1 - p_mean)
    entropies = []
    for c in confidences:
        p = float(np.clip(c, eps, 1 - eps))
        entropies.append(-p * np.log(p) - (1 - p) * np.log(1 - p))
    return float(h_mean - np.mean(entropies))
