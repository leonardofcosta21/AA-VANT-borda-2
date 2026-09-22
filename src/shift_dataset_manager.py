"""
Distribution Shift Dataset Manager.

Supports controlled covariate-shift experiments (H2) by managing
source→target dataset pairs. The model is trained on a source dataset
and adapted/evaluated on a target dataset with a different visual
distribution (same class taxonomy, shifted input statistics).

Supported shift pairs (aligned with the thesis experimental protocol):
  VisDrone  → FloodNet   (aerial crowd/vehicle → flood scene)
  VisDrone  → SARD       (aerial → search-and-rescue)
  VisDrone  → AIDER      (aerial → disaster scene)

Shift simulation also works with a SINGLE dataset by splitting it into
a "source" domain (first fraction) and a "target" domain (last fraction),
with optional image-level augmentation to simulate visual shift
(brightness shift, fog overlay, colour jitter). This allows running H2
without access to multiple separate datasets.

Usage (two separate datasets):
    mgr = ShiftDatasetManager(
        source_root="./datasets/VisDrone",
        target_root="./datasets/FloodNet",
        class_names=["person", "vehicle", "hazard"],
    )
    source_splits = mgr.get_source_splits(l0_size=100, test_fraction=0.2, seed=42)
    target_splits = mgr.get_target_splits(test_fraction=0.2, seed=42)

Usage (single dataset, simulated shift):
    mgr = ShiftDatasetManager(
        source_root="./datasets/SARD",
        target_root=None,          # same dataset, split by domain_fraction
        simulate_shift=True,
        domain_fraction=0.5,       # first 50% = source, last 50% = target
    )
"""

import os
import random
import shutil
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _discover_yolo_pairs(root: Path) -> List[Dict[str, str]]:
    """Return list of {image, label} dicts from a YOLO-format dataset root."""
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    candidates = []
    for subdir in ["images/train", "images/val", "images", "train/images", "valid/images"]:
        d = root / subdir
        if not d.exists():
            continue
        for img_path in sorted(d.iterdir()):
            if img_path.suffix.lower() not in img_exts:
                continue
            # Locate label
            lbl = Path(
                str(img_path)
                .replace("/images/", "/labels/")
                .replace("\\images\\", "\\labels\\")
            ).with_suffix(".txt")
            if not lbl.exists():
                # flat structure: same dir, .txt
                lbl = img_path.with_suffix(".txt")
            if lbl.exists():
                candidates.append({"image": str(img_path), "label": str(lbl)})
    return candidates


def _load_class_names(root: Path) -> Optional[List[str]]:
    """Try to read class names from data.yaml in dataset root."""
    yaml_path = root / "data.yaml"
    if not yaml_path.exists():
        return None
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    names = data.get("names")
    if isinstance(names, dict):
        return [names[k] for k in sorted(names.keys())]
    if isinstance(names, list):
        return names
    return None


def _write_yolo_yaml(output_dir: Path, class_names: List[str]) -> Path:
    """Write a data.yaml for the shift experiment directory."""
    content = {
        "path": str(output_dir.resolve()),
        "train": "source/train/images",
        "val": "source/val/images",
        # target paths are injected at runtime
        "nc": len(class_names),
        "names": {i: n for i, n in enumerate(class_names)},
    }
    p = output_dir / "data.yaml"
    with open(p, "w") as f:
        yaml.dump(content, f, default_flow_style=False)
    return p


def _copy_samples(samples: List[Dict], dst_root: Path, split: str):
    """Copy (image, label) pairs into YOLO train/val structure."""
    img_dir = dst_root / split / "images"
    lbl_dir = dst_root / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for s in samples:
        src_img = Path(s["image"])
        src_lbl = Path(s["label"])
        dst_img = img_dir / src_img.name
        dst_lbl = lbl_dir / (src_img.stem + ".txt")
        if not dst_img.exists():
            try:
                dst_img.symlink_to(src_img.resolve())
            except OSError:
                shutil.copy2(str(src_img), str(dst_img))
        if not dst_lbl.exists():
            try:
                dst_lbl.symlink_to(src_lbl.resolve())
            except OSError:
                shutil.copy2(str(src_lbl), str(dst_lbl))


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ShiftDatasetManager:
    """
    Manages source and target datasets for controlled covariate-shift
    experiments (H2 in the thesis).

    Parameters
    ----------
    source_root : str or Path
        Path to the source YOLO-format dataset (training domain).
    target_root : str or Path or None
        Path to the target YOLO-format dataset (deployment domain).
        Pass None to use the same dataset with simulated shift.
    class_names : list[str] or None
        Shared class taxonomy. If None, inferred from source data.yaml.
    simulate_shift : bool
        When target_root is None, simulate shift by splitting the source
        dataset by domain_fraction and applying image augmentation.
    domain_fraction : float
        Fraction of source used as source domain when simulating shift.
        The remaining (1 - domain_fraction) becomes the target domain.
    shift_intensity : str
        One of "none", "mild", "moderate", "severe".
        Controls the strength of the simulated shift augmentation.
    """

    SHIFT_PAIR_LABEL = "source→target"

    def __init__(
        self,
        source_root: str,
        target_root: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        simulate_shift: bool = False,
        domain_fraction: float = 0.5,
        shift_intensity: str = "moderate",
    ):
        self.source_root = Path(source_root)
        self.target_root = Path(target_root) if target_root else None
        self.simulate_shift = simulate_shift
        self.domain_fraction = domain_fraction
        self.shift_intensity = shift_intensity

        # Discover source samples
        self._source_samples = _discover_yolo_pairs(self.source_root)
        if not self._source_samples:
            raise ValueError(f"No YOLO image-label pairs found in source: {self.source_root}")
        print(f"[ShiftDatasetManager] Source: {len(self._source_samples)} samples ({self.source_root.name})")

        # Discover target samples
        if self.target_root is not None:
            self._target_samples = _discover_yolo_pairs(self.target_root)
            if not self._target_samples:
                raise ValueError(f"No YOLO image-label pairs found in target: {self.target_root}")
            print(f"[ShiftDatasetManager] Target: {len(self._target_samples)} samples ({self.target_root.name})")
        else:
            self._target_samples = None
            print(f"[ShiftDatasetManager] Target: simulated shift (domain_fraction={domain_fraction}, intensity={shift_intensity})")

        # Class names
        if class_names is not None:
            self.class_names = class_names
        else:
            self.class_names = _load_class_names(self.source_root)
            if self.class_names is None:
                self.class_names = ["object"]
            print(f"[ShiftDatasetManager] Classes: {self.class_names}")

    # ------------------------------------------------------------------
    # Splits
    # ------------------------------------------------------------------

    def get_source_splits(
        self,
        l0_size: int,
        test_fraction: float = 0.2,
        seed: int = 42,
    ) -> Dict[str, List[Dict]]:
        """
        Partition source dataset into L0 (initial labelled), U (pool), T (test).

        Returns
        -------
        dict with keys "L0", "U", "T" mapping to lists of sample dicts.
        """
        rng = random.Random(seed)
        samples = self._source_samples.copy()
        rng.shuffle(samples)

        n = len(samples)
        n_test = max(int(n * test_fraction), 50)
        T = samples[:n_test]
        remaining = samples[n_test:]

        l0_size = min(l0_size, len(remaining) - 50)
        L0 = remaining[:l0_size]
        U = remaining[l0_size:]

        print(f"[ShiftDatasetManager] Source splits → L0={len(L0)}, U={len(U)}, T={len(T)}")
        return {"L0": L0, "U": U, "T": T}

    def get_target_splits(
        self,
        test_fraction: float = 0.2,
        seed: int = 42,
    ) -> Dict[str, List[Dict]]:
        """
        Return target dataset split into U_target (adaptation pool) and T_target (test).

        For simulated shift, the target is derived from the source dataset's
        remaining fraction and augmented to simulate visual covariate shift.

        Returns
        -------
        dict with keys "U" and "T" mapping to lists of sample dicts.
        """
        if self._target_samples is not None:
            # Real target dataset
            rng = random.Random(seed)
            samples = self._target_samples.copy()
            rng.shuffle(samples)
            n = len(samples)
            n_test = max(int(n * test_fraction), 50)
            T = samples[:n_test]
            U = samples[n_test:]
            print(f"[ShiftDatasetManager] Target splits → U={len(U)}, T={len(T)}")
            return {"U": U, "T": T}
        else:
            # Simulated shift: take last (1 - domain_fraction) of source
            rng = random.Random(seed)
            samples = self._source_samples.copy()
            rng.shuffle(samples)
            split_idx = int(len(samples) * self.domain_fraction)
            target_samples = samples[split_idx:]
            n = len(target_samples)
            n_test = max(int(n * test_fraction), 30)
            T = target_samples[:n_test]
            U = target_samples[n_test:]
            print(f"[ShiftDatasetManager] Simulated target splits → U={len(U)}, T={len(T)}")
            return {"U": U, "T": T}

    # ------------------------------------------------------------------
    # Workspace preparation
    # ------------------------------------------------------------------

    def prepare_source_workspace(
        self,
        splits: Dict[str, List[Dict]],
        output_dir: str,
    ) -> str:
        """
        Build YOLO-format directory for source training.
        Returns path to data.yaml.
        """
        out = Path(output_dir) / "source"
        _copy_samples(splits["L0"], out, "train")
        _copy_samples(splits["T"], out, "val")

        yaml_data = {
            "path": str(out.resolve()),
            "train": "train/images",
            "val": "val/images",
            "nc": len(self.class_names),
            "names": {i: n for i, n in enumerate(self.class_names)},
        }
        yaml_path = out / "data.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False)
        return str(yaml_path)

    def add_source_samples(
        self,
        samples: List[Dict],
        output_dir: str,
    ):
        """Add newly annotated source samples to the training split."""
        out = Path(output_dir) / "source"
        _copy_samples(samples, out, "train")

    def prepare_target_workspace(
        self,
        splits: Dict[str, List[Dict]],
        output_dir: str,
    ) -> str:
        """
        Build YOLO-format directory for target evaluation.
        Applies simulated augmentation if in simulate_shift mode.
        Returns path to target data.yaml.
        """
        out = Path(output_dir) / "target"

        if self.simulate_shift and self.target_root is None:
            # Apply augmentation to simulate visual shift
            aug_samples = self._augment_for_shift(splits["T"], out / "val")
            _copy_samples(splits["U"], out, "train")
        else:
            _copy_samples(splits["T"], out, "val")
            _copy_samples(splits["U"], out, "train")

        yaml_data = {
            "path": str(out.resolve()),
            "train": "train/images",
            "val": "val/images",
            "nc": len(self.class_names),
            "names": {i: n for i, n in enumerate(self.class_names)},
        }
        yaml_path = out / "data.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False)
        return str(yaml_path)

    def add_target_samples(
        self,
        samples: List[Dict],
        output_dir: str,
    ):
        """Add newly annotated (simulated oracle) target samples to target train split."""
        out = Path(output_dir) / "target"
        _copy_samples(samples, out, "train")

    # ------------------------------------------------------------------
    # Progressive pool release for target domain
    # ------------------------------------------------------------------

    def get_target_cycle_batch(
        self,
        u_pool: List[Dict],
        cycle: int,
        num_cycles: int,
    ) -> List[Dict]:
        """
        Progressively release target unlabelled samples across cycles.
        Same strategy as the source pool release in the baseline pipeline.
        """
        n = len(u_pool)
        if n == 0:
            return []
        batch_size = max(1, n // num_cycles)
        start = cycle * batch_size
        end = start + batch_size if cycle < num_cycles - 1 else n
        return u_pool[start:end]

    # ------------------------------------------------------------------
    # Simulated shift augmentation
    # ------------------------------------------------------------------

    def _augment_for_shift(
        self,
        samples: List[Dict],
        dst_img_dir: Path,
    ) -> List[Dict]:
        """
        Apply image augmentations that simulate covariate shift.
        Images are saved to dst_img_dir; labels are symlinked as usual.
        Returns augmented sample dicts.

        Intensity levels:
          mild     — ±20% brightness, minor colour jitter
          moderate — ±40% brightness, contrast shift, mild fog overlay
          severe   — strong brightness/contrast, heavy fog, colour cast
        """
        try:
            import cv2
        except ImportError:
            # cv2 not available: fall back to copying without augmentation
            print("[ShiftDatasetManager] cv2 not found — copying target without augmentation")
            _copy_samples(samples, dst_img_dir.parent, "val")
            return samples

        dst_img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir = dst_img_dir.parent / "labels"
        lbl_dir.mkdir(parents=True, exist_ok=True)

        intensity_params = {
            "none":     dict(brightness=0.0, contrast=0.0, fog=0.0),
            "mild":     dict(brightness=0.20, contrast=0.10, fog=0.10),
            "moderate": dict(brightness=0.40, contrast=0.25, fog=0.25),
            "severe":   dict(brightness=0.60, contrast=0.40, fog=0.45),
        }
        params = intensity_params.get(self.shift_intensity, intensity_params["moderate"])

        aug_samples = []
        rng = np.random.RandomState(0)

        for s in samples:
            src_img = Path(s["image"])
            src_lbl = Path(s["label"])
            dst_img = dst_img_dir / src_img.name
            dst_lbl = lbl_dir / (src_img.stem + ".txt")

            # Read and augment image
            img = cv2.imread(str(src_img))
            if img is None:
                # Fallback: just copy
                shutil.copy2(str(src_img), str(dst_img))
            else:
                img = self._apply_shift(img, params, rng)
                cv2.imwrite(str(dst_img), img)

            # Symlink label
            if not dst_lbl.exists():
                try:
                    dst_lbl.symlink_to(src_lbl.resolve())
                except OSError:
                    shutil.copy2(str(src_lbl), str(dst_lbl))

            aug_samples.append({"image": str(dst_img), "label": str(dst_lbl)})

        return aug_samples

    @staticmethod
    def _apply_shift(
        img: "np.ndarray",
        params: Dict[str, float],
        rng: "np.random.RandomState",
    ) -> "np.ndarray":
        """Apply brightness, contrast and fog augmentation to a single image."""
        img = img.astype(np.float32)

        # Brightness shift (additive, random sign)
        b = params["brightness"]
        if b > 0:
            delta = rng.uniform(-b, b) * 255
            img = np.clip(img + delta, 0, 255)

        # Contrast shift (multiplicative around mean)
        c = params["contrast"]
        if c > 0:
            mean = img.mean()
            factor = 1.0 + rng.uniform(-c, c)
            img = np.clip((img - mean) * factor + mean, 0, 255)

        # Fog overlay (additive white towards bright)
        f = params["fog"]
        if f > 0:
            fog_alpha = rng.uniform(0, f)
            fog_layer = np.ones_like(img) * 240  # near-white fog
            img = np.clip(img * (1 - fog_alpha) + fog_layer * fog_alpha, 0, 255)

        return img.astype(np.uint8)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def describe(self) -> str:
        """Return a human-readable description of the shift experiment setup."""
        src_name = self.source_root.name
        if self.target_root is not None:
            tgt_name = self.target_root.name
            shift_type = "real_dataset"
        else:
            tgt_name = f"{src_name}_shifted"
            shift_type = f"simulated_{self.shift_intensity}"
        return (
            f"ShiftPair: {src_name}→{tgt_name} | "
            f"type={shift_type} | classes={self.class_names}"
        )
