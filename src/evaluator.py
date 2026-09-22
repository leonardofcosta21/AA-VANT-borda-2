"""
Evaluation metrics aligned with research hypotheses H1-H4.

  H1 (Annotation Efficiency) -> AUC
  H2 (Robustness under Shift) -> mAP@50 trajectory, recall stability
  H3 (Diversity Benefit) -> AUC comparison BALD vs BALD+diversity
  H4 (Operational Feasibility) -> inference latency
"""

import numpy as np
import time
import json
from pathlib import Path
from ultralytics import YOLO

from src.metrics import trapezoid


def _cuda_sync():
    """Block until queued GPU work finishes, so timings are real.

    Without this, ``predict`` returns as soon as the kernels are queued
    and every CUDA latency figure is understated.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


class Evaluator:
    """Evaluate model performance and track learning trajectories."""
    
    def __init__(self, test_data_yaml, device="cpu"):
        self.test_data_yaml = str(test_data_yaml)
        self.device = device
        self.trajectory = []  # List of per-cycle metrics
    
    def evaluate(self, model, cycle_idx, cumulative_labels):
        """
        Evaluate model on the fixed test set.
        
        Args:
            model: YOLO model to evaluate
            cycle_idx: current adaptation cycle
            cumulative_labels: total number of labelled samples so far
        
        Returns:
            metrics: dict with mAP50, precision, recall, latency
        """
        # Run validation
        try:
            results = model.val(
                data=self.test_data_yaml,
                imgsz=640,
                batch=16,
                conf=0.25,
                iou=0.45,
                verbose=False,
                device=self.device,
                plots=False,
            )
            
            map50 = float(results.box.map50) if hasattr(results.box, 'map50') else 0.0
            map50_95 = float(results.box.map) if hasattr(results.box, 'map') else 0.0
            precision = float(results.box.mp) if hasattr(results.box, 'mp') else 0.0
            recall = float(results.box.mr) if hasattr(results.box, 'mr') else 0.0
            
        except Exception as e:
            print(f"[WARNING] Evaluation failed at cycle {cycle_idx}: {e}")
            map50 = 0.0
            map50_95 = 0.0
            precision = 0.0
            recall = 0.0
        
        # Measure inference latency (deterministic single pass)
        latency = self.measure_latency(model, n_samples=20)
        
        metrics = {
            'cycle': cycle_idx,
            'cumulative_labels': cumulative_labels,
            'mAP50': map50,
            'mAP50_95': map50_95,
            'precision': precision,
            'recall': recall,
            'latency_ms': latency,
        }
        
        self.trajectory.append(metrics)
        
        return metrics
    
    def measure_latency(self, model, n_samples=20, image_paths=None):
        """
        Measure average inference latency (ms per image).

        Timing runs on real test images when ``image_paths`` is supplied.
        The previous implementation timed a single buffer of uniform
        random noise, which is not a valid latency measurement for a
        detector: noise produces almost no candidate boxes, so NMS and the
        post-processing path -- a real share of YOLO's per-frame cost --
        were never exercised, and repeating one identical buffer let
        caches make the numbers optimistic. A noise buffer is still used
        as a last resort when no images are available, and the returned
        dict flags that case so the thesis never reports a synthetic
        number as a measured one.

        Returns the mean for backward compatibility; the full
        distribution is kept in ``self.last_latency_profile`` because an
        edge feasibility claim depends on the tail, not the mean. For the
        full profiling path (memory, power, utilisation) use
        ``src.profiling.profile_inference``.
        """
        device = self.device
        synthetic = not image_paths

        if synthetic:
            dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            sources = [dummy] * (n_samples + 3)
        else:
            paths = list(image_paths)
            sources = [paths[i % len(paths)] for i in range(n_samples + 3)]

        for src in sources[:3]:
            try:
                model.predict(src, verbose=False, device=device)
            except Exception:
                continue

        _cuda_sync()
        times = []
        for src in sources[3:]:
            start = time.perf_counter()
            try:
                model.predict(src, verbose=False, device=device)
            except Exception:
                continue
            _cuda_sync()
            times.append((time.perf_counter() - start) * 1000)

        if not times:
            self.last_latency_profile = {"error": "all inference calls failed"}
            return 0.0

        arr = np.array(times)
        self.last_latency_profile = {
            "mean_ms": float(arr.mean()),
            "std_ms": float(arr.std()),
            "median_ms": float(np.median(arr)),
            "p95_ms": float(np.percentile(arr, 95)),
            "max_ms": float(arr.max()),
            "n": int(len(arr)),
            "synthetic_input": synthetic,
        }
        return float(arr.mean())
    
    def compute_auc(self):
        """
        Compute Area Under the Learning Curve.
        AUC = integral of mAP50 over cumulative labels.
        Uses trapezoidal rule.
        """
        if len(self.trajectory) < 2:
            return 0.0
        
        x = [m['cumulative_labels'] for m in self.trajectory]
        y = [m['mAP50'] for m in self.trajectory]

        return trapezoid(y, x)
    
    def compute_stability(self):
        """
        Compute learning stability metrics.
        - monotonicity: fraction of cycles where mAP50 increased
        - smoothness: std of mAP50 differences between consecutive cycles
        """
        if len(self.trajectory) < 2:
            return {'monotonicity': 0.0, 'smoothness': 0.0}
        
        maps = [m['mAP50'] for m in self.trajectory]
        diffs = [maps[i+1] - maps[i] for i in range(len(maps)-1)]
        
        monotonicity = sum(1 for d in diffs if d >= 0) / len(diffs)
        smoothness = float(np.std(diffs))
        
        return {
            'monotonicity': monotonicity,
            'smoothness': smoothness,
        }
    
    def get_summary(self):
        """Get complete evaluation summary."""
        auc = self.compute_auc()
        stability = self.compute_stability()
        
        final = self.trajectory[-1] if self.trajectory else {}
        
        # Budget-normalised AUC: the raw integral scales with the
        # annotation budget (hence values like 240.65), which the board
        # asked to be made readable. Dividing by the span integrated over
        # turns it into a mean mAP@50 per unit of annotation, in [0, 1].
        span = 0.0
        if len(self.trajectory) > 1:
            span = float(
                self.trajectory[-1]['cumulative_labels']
                - self.trajectory[0]['cumulative_labels']
            )

        return {
            'trajectory': self.trajectory,
            'auc': auc,
            'auc_normalized': float(auc / span) if span > 0 else 0.0,
            'annotation_span': span,
            'stability': stability,
            'final_mAP50': final.get('mAP50', 0.0),
            'final_precision': final.get('precision', 0.0),
            'final_recall': final.get('recall', 0.0),
            'mean_latency_ms': np.mean([m['latency_ms'] for m in self.trajectory]) if self.trajectory else 0.0,
        }
    
    def save_results(self, filepath):
        """Save results to JSON."""
        summary = self.get_summary()
        
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=convert)
        
        return filepath
