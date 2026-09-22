"""
Configuration for the Adaptive UAV-Edge Learning Architecture.
All parameters aligned with Chapter 4 (Experimental Methodology).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path


@dataclass
class DatasetConfig:
    """Dataset and partitioning configuration."""
    data_root: str = "./datasets/SARD"
    yaml_path: str = "./datasets/sard.yaml"
    img_size: int = 640
    # Initial labelled set sizes to evaluate
    l0_sizes: List[int] = field(default_factory=lambda: [100, 200, 300, 400])
    # Test set fraction (held out, fixed)
    test_fraction: float = 0.2
    # Random seed for reproducible splits
    split_seed: int = 42


@dataclass
class ModelConfig:
    """Perception model configuration."""
    model_name: str = "yolov8n.pt"
    img_size: int = 640
    # Pretrained weights (COCO)
    pretrained: bool = True
    # Confidence threshold for detections
    conf_threshold: float = 0.25
    # IoU threshold for NMS
    iou_threshold: float = 0.5


@dataclass
class UncertaintyConfig:
    """MC Dropout uncertainty estimation configuration."""
    # Number of stochastic forward passes
    T: int = 8
    # Dropout rate (applied during MC inference)
    dropout_rate: float = 0.1
    # Fraction alpha for quantile-based threshold
    alpha: float = 0.1


@dataclass
class ActiveLearningConfig:
    """Active Learning and acquisition configuration."""
    # Annotation budget per cycle
    budget_per_cycle: int = 50
    # Number of adaptation cycles
    num_cycles: int = 8
    # Diversity: number of clusters (adaptive formula)
    # K = min(budget, floor(|C_t| / beta))
    beta: int = 5
    # Strategies to evaluate
    strategies: List[str] = field(default_factory=lambda: [
        "random",
        "deterministic",
        "bald_only",
        "bald_diversity"  # proposed
    ])


@dataclass
class TrainingConfig:
    """Incremental fine-tuning configuration."""
    # Epochs per update cycle
    epochs_per_cycle: int = 5
    # Batch size
    batch_size: int = 16
    # Initial learning rate (for first training on L0)
    initial_lr: float = 0.01
    # Learning rate for incremental updates (1/10 of initial)
    incremental_lr: float = 0.001
    # Optimizer
    optimizer: str = "SGD"
    # Dual-condition trigger
    n_min: int = 50  # Condition A: minimum new samples
    tau_shift_percentile: float = 90  # Condition B: calibrated from L0
    shift_window: int = 3  # W: number of recent batches for moving average


@dataclass
class ShiftConfig:
    """Distribution shift experiment configuration (H2)."""
    # Source dataset path (training domain)
    source_root: str = "./datasets/SARD"
    # Target dataset path (deployment domain); None = simulate
    target_root: Optional[str] = None
    # Simulate shift via image augmentation when target_root is None
    simulate_shift: bool = True
    # Augmentation intensity: "none", "mild", "moderate", "severe"
    shift_intensity: str = "moderate"
    # Fraction of source used as source domain for simulated shift
    domain_fraction: float = 0.5
    # Number of independent seeds for H2 validation (≥5 recommended)
    n_seeds: int = 5
    # Base seed (seed_k = base_seed + k)
    base_seed: int = 42
    # Known shift pair label for logging
    pair_label: str = "SARD→SARD_shifted"


@dataclass
class EvaluationConfig:
    """Evaluation metrics configuration."""
    # IoU threshold for mAP
    iou_threshold: float = 0.5
    # Measure inference latency
    measure_latency: bool = True
    # Number of warmup images for latency measurement
    latency_warmup: int = 10
    # Number of images for latency measurement
    latency_samples: int = 100


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    active_learning: ActiveLearningConfig = field(default_factory=ActiveLearningConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    shift: ShiftConfig = field(default_factory=ShiftConfig)
    # Output directory
    output_dir: str = "./results"
    # Global random seed
    seed: int = 42
    # Device
    device: str = "auto"  # "auto", "cuda", "cpu"
    # Verbose logging
    verbose: bool = True
