import os, time, shutil
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from ultralytics import YOLO


class ModelTrainer:
    def __init__(self, model_name="yolov8n.pt", img_size=640, device="auto"):
        self.model_name = model_name
        self.img_size = img_size
        self.device = device
        self.model = None
        self.current_weights_path = None
        self.training_history = []

    def initial_training(self, yaml_path, epochs=50, batch_size=16, lr=0.01, output_dir="./runs/initial"):
        print(f"\n[Trainer] Initial training: epochs={epochs}, bs={batch_size}, lr={lr}")
        self.model = YOLO(self.model_name)
        self.model.train(data=yaml_path, epochs=epochs, batch=batch_size, imgsz=self.img_size,
                         lr0=lr, lrf=0.01, project=output_dir, name="initial", exist_ok=True,
                         verbose=True, plots=True, save=True, patience=10,
                         device=self.device if self.device != "auto" else None)
        save_dir = Path(self.model.trainer.save_dir)
        wp = save_dir / "weights" / "best.pt"
        if not wp.exists():
            wp = save_dir / "weights" / "last.pt"
        self.current_weights_path = str(wp)
        self.model = YOLO(self.current_weights_path)
        return self.current_weights_path

    def incremental_update(self, yaml_path, cycle, epochs=5, batch_size=16, lr=0.001, output_dir="./runs/updates"):
        print(f"\n[Trainer] Incremental update cycle {cycle}: epochs={epochs}, lr={lr}")
        t0 = time.time()
        self.model = YOLO(self.current_weights_path)
        self.model.train(data=yaml_path, epochs=epochs, batch=batch_size, imgsz=self.img_size,
                         lr0=lr, lrf=0.1, project=output_dir, name=f"cycle_{cycle}", exist_ok=True,
                         verbose=False, plots=False, save=True, patience=epochs,
                         device=self.device if self.device != "auto" else None)
        dt = time.time() - t0
        save_dir = Path(self.model.trainer.save_dir)
        wp = save_dir / "weights" / "best.pt"
        if not wp.exists():
            wp = save_dir / "weights" / "last.pt"
        self.current_weights_path = str(wp)
        self.model = YOLO(self.current_weights_path)
        self.training_history.append({"cycle": cycle, "weights": self.current_weights_path, "time": dt})
        return self.current_weights_path, dt

    def evaluate(self, yaml_path, split="val"):
        if self.model is None:
            self.model = YOLO(self.current_weights_path)
        r = self.model.val(data=yaml_path, split=split, imgsz=self.img_size, conf=0.25, iou=0.5,
                           verbose=False, plots=False, device=self.device if self.device != "auto" else None)
        return {"mAP50": float(r.box.map50), "mAP50_95": float(r.box.map),
                "precision": float(r.box.mp), "recall": float(r.box.mr)}

    def measure_latency(self, image_paths, warmup=10, n_samples=100):
        if self.model is None:
            self.model = YOLO(self.current_weights_path)
        imgs = image_paths[:min(n_samples+warmup, len(image_paths))]
        for img in imgs[:warmup]:
            self.model.predict(source=img, imgsz=self.img_size, verbose=False,
                               device=self.device if self.device != "auto" else None)
        measure = imgs[warmup:warmup+n_samples]
        t0 = time.time()
        for img in measure:
            self.model.predict(source=img, imgsz=self.img_size, verbose=False,
                               device=self.device if self.device != "auto" else None)
        return ((time.time()-t0)/len(measure))*1000

    def get_weights_path(self):
        return self.current_weights_path


class UpdateTrigger:
    """Dual-condition trigger: accumulation OR shift detection."""
    def __init__(self, n_min=50, tau_shift=None, window=3):
        self.n_min, self.tau_shift, self.window = n_min, tau_shift, window
        self.uncertainty_history, self.n_accumulated = [], 0

    def calibrate_threshold(self, uncertainties, percentile=90):
        self.tau_shift = float(np.percentile(uncertainties, percentile))
        print(f"[Trigger] tau_shift={self.tau_shift:.4f} (p{percentile})")

    def record_batch_uncertainty(self, mean_u):
        self.uncertainty_history.append(mean_u)

    def add_annotations(self, n):
        self.n_accumulated += n

    def should_update(self):
        if self.n_accumulated >= self.n_min:
            return True, f"accumulation(n={self.n_accumulated})"
        if self.tau_shift and len(self.uncertainty_history) >= self.window:
            u = np.mean(self.uncertainty_history[-self.window:])
            if u > self.tau_shift:
                return True, f"shift(u={u:.4f})"
        return False, "none"

    def reset_after_update(self):
        self.n_accumulated = 0