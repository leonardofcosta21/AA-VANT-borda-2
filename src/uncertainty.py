"""
Uncertainty estimation via Monte Carlo Dropout and BALD.

Implements:
  - MC Dropout forward passes with dropout active at inference
  - Predictive variance (image-level)
  - BALD: Bayesian Active Learning by Disagreement (mutual information)
  - Deterministic uncertainty (1 - max confidence)
"""

import torch
import numpy as np
from ultralytics import YOLO
from copy import deepcopy


def enable_dropout(model):
    """Enable dropout layers during inference for MC Dropout."""
    for module in model.model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def add_dropout_to_model(model, dropout_rate=0.1):
    """
    Inject dropout layers into YOLOv8 detection head if not present.
    Modifies the model in-place by adding Dropout after Conv layers in the head.
    """
    head = model.model.model[-1]  # Detection head
    for name, module in head.named_modules():
        if isinstance(module, torch.nn.Sequential):
            layers = list(module.children())
            new_layers = []
            for layer in layers:
                new_layers.append(layer)
                if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
                    new_layers.append(torch.nn.Dropout(p=dropout_rate))
            if len(new_layers) > len(layers):
                for i, layer in enumerate(new_layers):
                    module.add_module(str(i), layer)
    return model


def mc_dropout_predict(model, image_path, T=8, conf=0.25, device="cpu"):
    """
    Perform T stochastic forward passes with dropout active.
    
    Returns:
        predictions: list of T prediction results
        confidences_per_pass: list of T arrays, each with per-detection confidences
    """
    enable_dropout(model)
    
    all_predictions = []
    all_confidences = []
    
    for t in range(T):
        with torch.no_grad():
            results = model.predict(
                image_path,
                conf=conf,
                iou=0.45,
                verbose=False,
                device=device
            )
        
        if len(results) > 0 and results[0].boxes is not None:
            confs = results[0].boxes.conf.cpu().numpy()
        else:
            confs = np.array([])
        
        all_predictions.append(results)
        all_confidences.append(confs)
    
    return all_predictions, all_confidences


def compute_predictive_variance(all_confidences):
    """
    Compute image-level predictive variance across T MC passes.
    
    u(x) = (1/N_x) * sum_i sigma_i^2(x)
    
    where sigma_i^2 is the variance of confidence for object i across passes.
    """
    T = len(all_confidences)
    
    if T == 0 or all(len(c) == 0 for c in all_confidences):
        return 0.0
    
    # Use the maximum number of detections across passes
    max_dets = max(len(c) for c in all_confidences)
    if max_dets == 0:
        return 0.0
    
    # Pad confidence arrays to same length
    padded = np.zeros((T, max_dets))
    for t, confs in enumerate(all_confidences):
        if len(confs) > 0:
            padded[t, :len(confs)] = confs[:max_dets]
    
    # Variance per detection across T passes
    variances = np.var(padded, axis=0)
    
    # Image-level: mean variance across detections
    return float(np.mean(variances))


def compute_bald_score(all_confidences):
    """
    Compute BALD score (mutual information) for an image.
    
    BALD = H[y|x] - (1/T) * sum_t H[y|x, theta_t]
    
    For detection: approximate using confidence scores.
    H[y|x] = entropy of mean predictions
    E[H[y|x,theta_t]] = mean entropy of individual predictions
    """
    T = len(all_confidences)
    
    if T == 0 or all(len(c) == 0 for c in all_confidences):
        return 0.0
    
    max_dets = max(len(c) for c in all_confidences)
    if max_dets == 0:
        return 0.0
    
    # Pad to same length
    padded = np.zeros((T, max_dets))
    for t, confs in enumerate(all_confidences):
        if len(confs) > 0:
            padded[t, :len(confs)] = np.clip(confs[:max_dets], 1e-8, 1 - 1e-8)
    
    # Binary entropy for each detection: H(p) = -p*log(p) - (1-p)*log(1-p)
    def binary_entropy(p):
        p = np.clip(p, 1e-8, 1 - 1e-8)
        return -p * np.log(p + 1e-8) - (1 - p) * np.log(1 - p + 1e-8)
    
    # Mean confidence across T passes per detection
    mean_conf = np.mean(padded, axis=0)
    
    # H[y|x]: entropy of mean prediction
    total_entropy = binary_entropy(mean_conf)
    
    # E[H[y|x, theta_t]]: mean of individual entropies
    individual_entropies = np.mean(binary_entropy(padded), axis=0)
    
    # BALD = total_entropy - mean_individual_entropy (per detection)
    bald_per_det = total_entropy - individual_entropies
    
    # Image-level: mean BALD across detections
    return float(np.mean(bald_per_det))


def compute_deterministic_uncertainty(model, image_path, conf=0.25, device="cpu"):
    """
    Deterministic uncertainty: u(x) = 1 - max_confidence.
    Single forward pass, no dropout.
    """
    # Disable dropout
    model.eval()
    
    with torch.no_grad():
        results = model.predict(
            image_path,
            conf=conf,
            iou=0.45,
            verbose=False,
            device=device
        )
    
    if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
        confs = results[0].boxes.conf.cpu().numpy()
        # u = 1 - mean(max_conf_per_detection)
        return float(1.0 - np.mean(confs))
    else:
        return 1.0  # No detections = maximum uncertainty


def compute_uncertainty_scores(model, image_paths, strategy, T=8, 
                                conf=0.25, device="cpu", alpha=0.1):
    """
    Compute uncertainty scores for a batch of images.
    
    Args:
        model: YOLOv8 model
        image_paths: list of image file paths
        strategy: "random", "deterministic", "bald_only", or "bald_diversity"
        T: number of MC Dropout passes
        conf: confidence threshold
        device: compute device
        alpha: fraction for quantile threshold
    
    Returns:
        scores: dict mapping image_path -> uncertainty score
        candidate_set: list of image paths exceeding threshold tau
        all_mc_confidences: dict mapping image_path -> list of T confidence arrays (for BALD)
    """
    scores = {}
    all_mc_data = {}
    
    if strategy == "random":
        # Random: no uncertainty computation needed
        for img in image_paths:
            scores[img] = np.random.random()
        return scores, image_paths, {}
    
    elif strategy == "deterministic":
        for img in image_paths:
            scores[img] = compute_deterministic_uncertainty(model, img, conf, device)
        
    elif strategy in ("bald_only", "bald_diversity"):
        for img in image_paths:
            _, all_confs = mc_dropout_predict(model, img, T=T, conf=conf, device=device)
            
            if strategy == "bald_only":
                scores[img] = compute_bald_score(all_confs)
            else:  # bald_diversity
                scores[img] = compute_bald_score(all_confs)
            
            all_mc_data[img] = all_confs
    
    # Compute adaptive threshold: tau = quantile_{1-alpha}
    score_values = np.array(list(scores.values()))
    if len(score_values) > 0:
        tau = np.quantile(score_values, 1 - alpha)
        candidate_set = [img for img, s in scores.items() if s >= tau]
    else:
        candidate_set = []
    
    return scores, candidate_set, all_mc_data
