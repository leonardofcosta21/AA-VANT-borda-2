"""
Dataset Manager for the Adaptive UAV-Edge Architecture.
Handles dataset loading, partitioning into L0/U/T splits,
and progressive release of unlabelled samples across cycles.
"""

import os
import random
import shutil
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict


class DatasetManager:
    """Manages dataset partitioning and progressive release for AL experiments."""

    def __init__(self, data_root: str, img_size: int = 640, seed: int = 42):
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Discover images and labels
        self.image_paths = []
        self.label_paths = []
        self._discover_data()

    def _discover_data(self):
        """Find all image-label pairs in the dataset."""
        img_dirs = [
            self.data_root / "images" / "train",
            self.data_root / "images" / "val",
            self.data_root / "train" / "images",
            self.data_root / "valid" / "images",
        ]

        label_dirs = [
            self.data_root / "labels" / "train",
            self.data_root / "labels" / "val",
            self.data_root / "train" / "labels",
            self.data_root / "valid" / "labels",
        ]

        img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

        for img_dir, lbl_dir in zip(img_dirs, label_dirs):
            if not img_dir.exists():
                continue
            for img_path in sorted(img_dir.iterdir()):
                if img_path.suffix.lower() not in img_extensions:
                    continue
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                if lbl_path.exists():
                    self.image_paths.append(str(img_path))
                    self.label_paths.append(str(lbl_path))

        if not self.image_paths:
            # Try flat structure
            for img_path in sorted(self.data_root.rglob("*")):
                if img_path.suffix.lower() in img_extensions:
                    lbl_path = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
                    if not lbl_path.exists():
                        lbl_path = img_path.with_suffix(".txt")
                    if lbl_path.exists():
                        self.image_paths.append(str(img_path))
                        self.label_paths.append(str(lbl_path))

        print(f"[DatasetManager] Found {len(self.image_paths)} image-label pairs")

    def create_splits(
        self,
        l0_size: int,
        test_fraction: float = 0.2,
        output_dir: str = "./experiment_data"
    ) -> Dict[str, List[int]]:
        """
        Create L0 (initial labelled), U (unlabelled pool), T (test) splits.

        Returns dict with indices into self.image_paths for each split.
        """
        n_total = len(self.image_paths)
        indices = list(range(n_total))
        self.rng.shuffle(indices)

        n_test = int(n_total * test_fraction)
        n_test = max(n_test, 50)  # minimum test set size

        test_indices = indices[:n_test]
        remaining = indices[n_test:]

        # L0: initial labelled set
        l0_size = min(l0_size, len(remaining) - 100)  # keep at least 100 for U
        l0_indices = remaining[:l0_size]
        u_indices = remaining[l0_size:]

        splits = {
            "L0": l0_indices,
            "U": u_indices,
            "T": test_indices,
        }

        print(f"[DatasetManager] Splits: L0={len(l0_indices)}, "
              f"U={len(u_indices)}, T={len(test_indices)}")

        return splits

    def prepare_yolo_dataset(
        self,
        splits: Dict[str, List[int]],
        output_dir: str,
        class_names: Optional[List[str]] = None
    ) -> str:
        """
        Create YOLO-format dataset directory with train/val/test structure.
        Returns path to the generated YAML config.
        """
        output_path = Path(output_dir)

        # Create directory structure
        for split_name, folder_name in [("L0", "train"), ("T", "val")]:
            for sub in ["images", "labels"]:
                (output_path / folder_name / sub).mkdir(parents=True, exist_ok=True)

        # Copy files
        for split_name, folder_name in [("L0", "train"), ("T", "val")]:
            for idx in splits[split_name]:
                img_src = self.image_paths[idx]
                lbl_src = self.label_paths[idx]
                img_dst = output_path / folder_name / "images" / Path(img_src).name
                lbl_dst = output_path / folder_name / "labels" / (Path(img_src).stem + ".txt")
                if not img_dst.exists():
                    shutil.copy2(img_src, str(img_dst))
                lbl_dst_path = output_path / folder_name / "labels" / (Path(img_src).stem + ".txt")
                if not lbl_dst_path.exists():
                    shutil.copy2(lbl_src, str(lbl_dst_path))

        # Detect classes from label files
        if class_names is None:
            class_ids = set()
            for idx in splits["L0"] + splits["T"]:
                with open(self.label_paths[idx], 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_ids.add(int(parts[0]))
            n_classes = max(class_ids) + 1 if class_ids else 1
            class_names = {i: f"class_{i}" for i in range(n_classes)}
        else:
            n_classes = len(class_names)
            class_names = {i: name for i, name in enumerate(class_names)}

        # Write YAML
        yaml_path = output_path / "dataset.yaml"
        yaml_content = {
            "path": str(output_path.resolve()),
            "train": "train/images",
            "val": "val/images",
            "nc": n_classes,
            "names": class_names,
        }
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

        return str(yaml_path)

    def add_samples_to_training(
        self,
        sample_indices: List[int],
        output_dir: str
    ):
        """Add newly annotated samples to the training set."""
        output_path = Path(output_dir)
        train_img_dir = output_path / "train" / "images"
        train_lbl_dir = output_path / "train" / "labels"
        train_img_dir.mkdir(parents=True, exist_ok=True)
        train_lbl_dir.mkdir(parents=True, exist_ok=True)

        for idx in sample_indices:
            img_src = self.image_paths[idx]
            lbl_src = self.label_paths[idx]
            img_name = Path(img_src).name
            lbl_name = Path(img_src).stem + ".txt"

            shutil.copy2(img_src, str(train_img_dir / img_name))
            shutil.copy2(lbl_src, str(train_lbl_dir / lbl_name))

    def get_image_paths_for_indices(self, indices: List[int]) -> List[str]:
        """Return image paths for given indices."""
        return [self.image_paths[i] for i in indices]

    def get_cycle_batch(
        self,
        u_indices: List[int],
        cycle: int,
        num_cycles: int
    ) -> List[int]:
        """
        Get the batch of unlabelled samples for a given cycle.
        Progressive release: divide U evenly across cycles.
        """
        batch_size = len(u_indices) // num_cycles
        start = cycle * batch_size
        end = start + batch_size if cycle < num_cycles - 1 else len(u_indices)
        return u_indices[start:end]
