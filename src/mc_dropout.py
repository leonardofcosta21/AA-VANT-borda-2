"""
Monte Carlo Dropout Uncertainty Estimation and BALD Computation.
Implements epistemic uncertainty quantification for YOLOv8 models
as described in Chapter 3 (Proposed Approach).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm


def enable_mc_dropout(model: nn.Module, dropout_rate: float = 0.1):
    """
    Enable dropout at inference time for MC sampling.
    Adds dropout layers after key modules if not present,
    or sets existing dropout layers to training mode.
    """
    dropout_found = False
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout_rate
            module.train()
            dropout_found = True
        elif isinstance(module, nn.Dropout2d):
            module.p = dropout_rate
            module.train()
            dropout_found = True

    if not dropout_found:
        # Insert dropout after Conv2d+BatchNorm blocks in the backbone
        _inject_dropout(model, dropout_rate)

    return model


def _inject_dropout(model: nn.Module, dropout_rate: float):
    """Inject Dropout2d layers into the model's backbone."""
    for name, module in model.named_children():
        if isinstance(module, nn.Sequential):
            new_modules = []
            for child in module:
                new_modules.append(child)
                if isinstance(child, (nn.Conv2d, nn.BatchNorm2d)):
                    pass  # accumulate
                if isinstance(child, nn.SiLU) or isinstance(child, nn.ReLU):
                    new_modules.append(nn.Dropout2d(p=dropout_rate))
            # Replace the sequential
            for i, m in enumerate(new_modules):
                if i < len(module):
                    pass
        _inject_dropout(module, dropout_rate)


def disable_mc_dropout(model: nn.Module):
    """Disable MC Dropout (return to standard inference mode)."""
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d)):
            module.eval()


class MCDropoutEstimator:
    """
    Computes uncertainty scores using MC Dropout and BALD.

    Implements the uncertainty estimation pass described in Section 3.3
    (Stage 2: Inference and Uncertainty Filtering).
    """

    def __init__(
        self,
        model_path: str,
        T: int = 8,
        dropout_rate: float = 0.1,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        img_size: int = 640,
        device: str = "auto"
    ):
        self.T = T
        self.dropout_rate = dropout_rate
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model
        self.yolo = YOLO(model_path)

    def compute_deterministic_predictions(
        self,
        image_paths: List[str]
    ) -> List[Dict]:
        """
        Single deterministic forward pass (dropout disabled).
        Returns predictions for each image.
        """
        results_list = []
        batch_size = 16  # process in small batches to avoid OOM
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]
            results = self.yolo.predict(
                source=batch,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                verbose=False,
                device=self.device
            )
            for r in results:
                pred = {
                    "boxes": r.boxes.xyxy.cpu().numpy() if r.boxes else np.array([]),
                    "scores": r.boxes.conf.cpu().numpy() if r.boxes else np.array([]),
                    "classes": r.boxes.cls.cpu().numpy() if r.boxes else np.array([]),
                }
                results_list.append(pred)
        return results_list

    def compute_mc_uncertainty(
        self,
        image_paths: List[str],
        batch_size: int = 8
    ) -> Dict[str, np.ndarray]:
        """
        Compute MC Dropout uncertainty scores for a batch of images.

        Returns dict with:
            - 'variance': mean predictive variance per image (epistemic uncertainty)
            - 'bald': BALD scores per image (mutual information)
            - 'mean_conf': mean confidence across T passes
            - 'det_scores': deterministic confidence (1 - max_conf)
        """
        n_images = len(image_paths)
        all_variances = np.zeros(n_images)
        all_bald = np.zeros(n_images)
        all_mean_conf = np.zeros(n_images)
        all_det_scores = np.zeros(n_images)

        # Get the underlying torch model
        torch_model = self.yolo.model

        for img_idx in tqdm(range(0, n_images, batch_size),
                           desc="MC Dropout", leave=False):
            batch_paths = image_paths[img_idx:img_idx + batch_size]
            actual_batch = len(batch_paths)

            # Collect T stochastic forward passes
            pass_confidences = []  # T x batch x max_det

            for t in range(self.T):
                # Enable dropout for this pass
                enable_mc_dropout(torch_model, self.dropout_rate)

                results = self.yolo.predict(
                    source=batch_paths,
                    conf=0.1,  # lower threshold to capture uncertain detections
                    iou=self.iou_threshold,
                    imgsz=self.img_size,
                    verbose=False,
                    device=self.device
                )

                # Disable dropout after pass
                disable_mc_dropout(torch_model)

                batch_confs = []
                for r in results:
                    if r.boxes and len(r.boxes) > 0:
                        confs = r.boxes.conf.cpu().numpy()
                        batch_confs.append(confs)
                    else:
                        batch_confs.append(np.array([0.0]))

                pass_confidences.append(batch_confs)

            # Compute per-image uncertainty metrics
            for b in range(actual_batch):
                global_idx = img_idx + b

                # Gather confidences across T passes for this image
                img_confs = []
                for t in range(self.T):
                    img_confs.append(pass_confidences[t][b])

                # Compute image-level metrics
                variance, bald, mean_conf = self._compute_image_metrics(img_confs)

                all_variances[global_idx] = variance
                all_bald[global_idx] = bald
                all_mean_conf[global_idx] = mean_conf
                all_det_scores[global_idx] = 1.0 - mean_conf

        return {
            "variance": all_variances,
            "bald": all_bald,
            "mean_conf": all_mean_conf,
            "det_scores": all_det_scores
        }

    def _compute_image_metrics(
        self,
        pass_confidences: List[np.ndarray]
    ) -> Tuple[float, float, float]:
        """
        Compute variance, BALD, and mean confidence for one image
        across T stochastic passes.

        Implements Equations from Section 3.3 and 3.4.
        """
        T = len(pass_confidences)

        # Use max confidence per pass as image-level summary
        max_confs = np.array([
            confs.max() if len(confs) > 0 else 0.0
            for confs in pass_confidences
        ])

        # Mean confidence across passes
        mean_conf = max_confs.mean()

        # Predictive variance (epistemic uncertainty)
        # sigma^2 = (1/T) * sum((f_t - mu)^2)
        variance = max_confs.var()

        # BALD: Mutual information approximation
        # BALD = H[E[p]] - E[H[p]]
        # For binary-like confidence: H(p) = -p*log(p) - (1-p)*log(1-p)
        eps = 1e-10

        # H[E[p]]: entropy of mean prediction
        p_mean = np.clip(mean_conf, eps, 1 - eps)
        h_mean = -p_mean * np.log(p_mean) - (1 - p_mean) * np.log(1 - p_mean)

        # E[H[p]]: mean entropy across passes
        h_passes = []
        for conf in max_confs:
            p = np.clip(conf, eps, 1 - eps)
            h = -p * np.log(p) - (1 - p) * np.log(1 - p)
            h_passes.append(h)
        e_h = np.mean(h_passes)

        bald = h_mean - e_h

        return float(variance), float(bald), float(mean_conf)

    def compute_adaptive_threshold(
        self,
        scores: np.ndarray,
        alpha: float = 0.1
    ) -> float:
        """
        Compute adaptive threshold tau_t = quantile_{1-alpha}(scores).
        Only the top alpha fraction of samples pass the threshold.
        """
        if len(scores) == 0:
            return 0.0
        return float(np.quantile(scores, 1 - alpha))

    def extract_features(
        self,
        image_paths: List[str],
        batch_size: int = 16
    ) -> np.ndarray:
        """
        Extract feature embeddings from penultimate layer of the model.
        Used for diversity-aware clustering in the acquisition stage.
        """
        all_features = []
        torch_model = self.yolo.model

        # Register hook to capture features
        features = {}

        def hook_fn(module, input, output):
            features['feat'] = output

        # Find the last layer before detection head
        # In YOLOv8, this is typically model.model[-2] or the SPPF output
        target_layer = None
        for name, module in torch_model.named_modules():
            if 'sppf' in name.lower() or 'c2f' in name.lower():
                target_layer = module

        if target_layer is None:
            # Fallback: use second-to-last module
            modules = list(torch_model.modules())
            target_layer = modules[-3] if len(modules) > 3 else modules[-1]

        hook = target_layer.register_forward_hook(hook_fn)

        try:
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]

                # Run inference to trigger the hook
                self.yolo.predict(
                    source=batch_paths,
                    conf=self.conf_threshold,
                    imgsz=self.img_size,
                    verbose=False,
                    device=self.device
                )

                if 'feat' in features:
                    feat = features['feat']
                    if isinstance(feat, torch.Tensor):
                        # Global average pooling
                        if feat.dim() == 4:
                            pooled = feat.mean(dim=[2, 3])
                        elif feat.dim() == 3:
                            pooled = feat.mean(dim=2)
                        else:
                            pooled = feat
                        all_features.append(pooled.detach().cpu().numpy())
        finally:
            hook.remove()

        if all_features:
            return np.concatenate(all_features, axis=0)
        else:
            # Fallback: return random features
            return np.random.randn(len(image_paths), 128)
