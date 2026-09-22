"""
Dataset utilities for partitioning, YAML generation, and progressive release.

Implements:
  - Dataset splitting (L0, U, T)
  - YOLO-format data.yaml generation
  - Progressive pool release across cycles
  - Distribution shift simulation
"""

import os
import shutil
import yaml
import random
import numpy as np
from pathlib import Path
from collections import defaultdict


def discover_dataset(dataset_dir):
    """
    Discover images and labels in a YOLO-format dataset.
    
    Expected structure:
      dataset_dir/
        images/
          train/  (or just images/)
          val/
        labels/
          train/
          val/
    
    Returns:
        samples: list of dicts with 'image' and 'label' paths
        classes: list of class names (from data.yaml if present)
    """
    dataset_dir = Path(dataset_dir)
    samples = []
    
    # Try to find images
    image_dirs = []
    for subdir in ['images/train', 'images/val', 'images', 'train/images', 'valid/images']:
        d = dataset_dir / subdir
        if d.exists():
            image_dirs.append(d)
    
    if not image_dirs:
        # Try flat structure
        image_dirs = [dataset_dir]
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    
    for img_dir in image_dirs:
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() in image_extensions:
                # Find corresponding label
                label_path = None
                for labels_base in ['labels', '../labels']:
                    lp = img_path.parent.parent / labels_base / img_path.parent.name / img_path.stem
                    for ext in ['.txt']:
                        candidate = lp.with_suffix(ext)
                        if candidate.exists():
                            label_path = candidate
                            break
                    if label_path:
                        break
                
                if label_path is None:
                    # Try same directory structure with 'labels' replacing 'images'
                    lp = Path(str(img_path).replace('/images/', '/labels/').replace('\\images\\', '\\labels\\'))
                    lp = lp.with_suffix('.txt')
                    if lp.exists():
                        label_path = lp
                
                if label_path and label_path.exists():
                    samples.append({
                        'image': str(img_path),
                        'label': str(label_path),
                    })
    
    # Try to load class names
    classes = []
    yaml_path = dataset_dir / 'data.yaml'
    if yaml_path.exists():
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
            if 'names' in data:
                if isinstance(data['names'], dict):
                    classes = [data['names'][k] for k in sorted(data['names'].keys())]
                elif isinstance(data['names'], list):
                    classes = data['names']
    
    return samples, classes


def partition_dataset(samples, L0_size, test_ratio=0.2, seed=42):
    """
    Partition samples into L0 (initial labelled), U (unlabelled pool), T (test).
    
    Args:
        samples: list of sample dicts
        L0_size: number of initial labelled samples
        test_ratio: fraction for test set
        seed: random seed
    
    Returns:
        L0: initial labelled set
        U: unlabelled pool (to be released progressively)
        T: held-out test set
    """
    rng = random.Random(seed)
    shuffled = samples.copy()
    rng.shuffle(shuffled)
    
    n = len(shuffled)
    n_test = int(n * test_ratio)
    
    T = shuffled[:n_test]
    remaining = shuffled[n_test:]
    
    L0_size = min(L0_size, len(remaining))
    L0 = remaining[:L0_size]
    U = remaining[L0_size:]
    
    return L0, U, T


def create_cycle_pool(U, cycle_idx, num_cycles, shift_mode="progressive"):
    """
    Release a subset of U for the current cycle.
    
    Args:
        U: full unlabelled pool
        cycle_idx: current cycle (0-indexed)
        num_cycles: total number of cycles
        shift_mode: "uniform" or "progressive" (simulate distribution shift)
    
    Returns:
        batch: list of samples available this cycle
    """
    n = len(U)
    if n == 0:
        return []
    
    batch_size = n // num_cycles
    start = cycle_idx * batch_size
    end = start + batch_size if cycle_idx < num_cycles - 1 else n
    
    return U[start:end]


def write_data_yaml(output_dir, train_dir, val_dir, class_names, yaml_name="data.yaml"):
    """
    Write a YOLO-format data.yaml file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data = {
        'path': str(output_dir.resolve()),
        'train': str(Path(train_dir).resolve()),
        'val': str(Path(val_dir).resolve()),
        'names': {i: name for i, name in enumerate(class_names)},
        'nc': len(class_names),
    }
    
    yaml_path = output_dir / yaml_name
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    return yaml_path


def prepare_yolo_split(samples, output_dir, split_name="train"):
    """
    Copy/symlink images and labels into YOLO directory structure.
    
    Creates:
        output_dir/images/split_name/
        output_dir/labels/split_name/
    """
    output_dir = Path(output_dir)
    img_dir = output_dir / "images" / split_name
    lbl_dir = output_dir / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    
    for sample in samples:
        src_img = Path(sample['image'])
        src_lbl = Path(sample['label'])
        
        dst_img = img_dir / src_img.name
        dst_lbl = lbl_dir / src_lbl.name
        
        # Use symlinks to save space, fallback to copy
        try:
            if not dst_img.exists():
                os.symlink(src_img.resolve(), dst_img)
            if not dst_lbl.exists():
                os.symlink(src_lbl.resolve(), dst_lbl)
        except OSError:
            if not dst_img.exists():
                shutil.copy2(str(src_img), str(dst_img))
            if not dst_lbl.exists():
                shutil.copy2(str(src_lbl), str(dst_lbl))
    
    return img_dir, lbl_dir


def update_training_set(current_train_dir, new_samples, split_name="train"):
    """
    Add newly annotated samples to the training set directory.
    """
    return prepare_yolo_split(new_samples, current_train_dir.parent.parent, split_name)
