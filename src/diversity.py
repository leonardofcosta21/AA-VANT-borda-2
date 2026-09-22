"""
Diversity-aware sample selection via k-means clustering.

Implements the cluster-then-select strategy:
  1. Extract feature embeddings from penultimate layer of M_k
  2. Cluster candidates into K groups via k-means
  3. Select top-scoring samples from each cluster proportionally
"""

import numpy as np
import torch
from sklearn.cluster import KMeans
from pathlib import Path


def extract_feature_embeddings(model, image_paths, device="cpu"):
    """
    Extract feature embeddings from the penultimate layer of YOLOv8.
    
    Uses a forward hook to capture intermediate activations from the
    backbone's last layer before the detection head.
    
    Returns:
        embeddings: dict mapping image_path -> numpy array of shape (d,)
    """
    embeddings = {}
    activation = {}
    
    def hook_fn(module, input, output):
        activation['feat'] = output
    
    # Register hook on backbone's last layer (before detect head)
    # In YOLOv8, model.model.model[-2] is typically the last feature layer
    try:
        target_layer = model.model.model[-2]
    except (IndexError, AttributeError):
        # Fallback: use model.model.model[9] (P5 features in YOLOv8)
        target_layer = model.model.model[9]
    
    handle = target_layer.register_forward_hook(hook_fn)
    
    model.eval()
    
    for img_path in image_paths:
        try:
            with torch.no_grad():
                _ = model.predict(
                    img_path,
                    conf=0.1,
                    verbose=False,
                    device=device
                )
            
            if 'feat' in activation:
                feat = activation['feat']
                if isinstance(feat, torch.Tensor):
                    # Global average pooling to get fixed-size embedding
                    if feat.dim() == 4:  # (B, C, H, W)
                        emb = feat.mean(dim=[2, 3]).squeeze(0).cpu().numpy()
                    elif feat.dim() == 3:  # (B, C, L)
                        emb = feat.mean(dim=2).squeeze(0).cpu().numpy()
                    else:
                        emb = feat.squeeze(0).cpu().numpy()
                    embeddings[img_path] = emb
                elif isinstance(feat, (list, tuple)):
                    # If multiple outputs, use the last one
                    f = feat[-1] if isinstance(feat[-1], torch.Tensor) else feat[0]
                    if f.dim() >= 3:
                        emb = f.mean(dim=list(range(2, f.dim()))).squeeze(0).cpu().numpy()
                    else:
                        emb = f.squeeze(0).cpu().numpy()
                    embeddings[img_path] = emb
        except Exception as e:
            # Fallback: random embedding
            embeddings[img_path] = np.random.randn(256)
    
    handle.remove()
    
    return embeddings


def cluster_then_select(candidate_paths, scores, embeddings, budget, beta=5):
    """
    Diversity-aware selection: cluster candidates then select top-scoring
    samples from each cluster proportionally.
    
    Args:
        candidate_paths: list of image paths in candidate set C_t
        scores: dict mapping image_path -> acquisition score (BALD)
        embeddings: dict mapping image_path -> feature embedding
        budget: b_t, number of samples to select
        beta: minimum cluster size factor
    
    Returns:
        selected: list of selected image paths (|selected| <= budget)
    """
    n = len(candidate_paths)
    
    if n <= budget:
        return candidate_paths
    
    # Determine K adaptively
    K = min(budget, n // beta)
    K = max(K, 1)  # At least 1 cluster
    
    # Build embedding matrix
    valid_paths = [p for p in candidate_paths if p in embeddings]
    if len(valid_paths) < budget:
        # Not enough valid embeddings, fallback to score-only selection
        sorted_paths = sorted(candidate_paths, key=lambda p: scores.get(p, 0), reverse=True)
        return sorted_paths[:budget]
    
    emb_matrix = np.array([embeddings[p] for p in valid_paths])
    
    # Handle NaN/Inf
    emb_matrix = np.nan_to_num(emb_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # k-means clustering
    if K >= len(valid_paths):
        K = max(1, len(valid_paths) // 2)
    
    try:
        kmeans = KMeans(n_clusters=K, random_state=42, n_init=10, max_iter=100)
        labels = kmeans.fit_predict(emb_matrix)
    except Exception:
        # Fallback to score-only selection
        sorted_paths = sorted(candidate_paths, key=lambda p: scores.get(p, 0), reverse=True)
        return sorted_paths[:budget]
    
    # Build cluster groups
    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(valid_paths[idx])
    
    # Proportional budget allocation per cluster
    selected = []
    remaining_budget = budget
    
    cluster_budgets = {}
    for j, members in clusters.items():
        b_j = int(np.floor(budget * len(members) / len(valid_paths)))
        cluster_budgets[j] = max(b_j, 0)
    
    # First pass: proportional selection
    for j, members in clusters.items():
        b_j = cluster_budgets[j]
        if b_j == 0:
            continue
        
        # Sort members by score (descending)
        sorted_members = sorted(members, key=lambda p: scores.get(p, 0), reverse=True)
        selected.extend(sorted_members[:b_j])
    
    remaining_budget = budget - len(selected)
    
    # Second pass: greedy allocation of remaining budget
    if remaining_budget > 0:
        already_selected = set(selected)
        # Get unselected candidates sorted by score
        remaining = [(p, scores.get(p, 0)) for p in valid_paths if p not in already_selected]
        remaining.sort(key=lambda x: x[1], reverse=True)
        
        for p, s in remaining[:remaining_budget]:
            selected.append(p)
    
    return selected[:budget]


def select_without_diversity(candidate_paths, scores, budget):
    """
    Select top-budget samples by score only (no diversity filtering).
    Used for random, deterministic, and BALD-only strategies.
    """
    sorted_paths = sorted(candidate_paths, key=lambda p: scores.get(p, 0), reverse=True)
    return sorted_paths[:budget]
