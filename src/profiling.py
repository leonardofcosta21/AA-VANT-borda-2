"""
Operational resource profiling for hypothesis H4 and contribution C3.

The examination board's objection to H4 was precise: "medir apenas
latencia e insuficiente -- ampliar para incluir consumo de memoria,
trafego de rede, utilizacao de processamento e consumo de energia". This
module measures all four, plus the thermal and duty-cycle signals that
make an edge-feasibility claim defensible.

Design
------
Two entry points cover the two questions the thesis asks.

``ResourceMonitor`` samples the host in a background thread while an
arbitrary block of work runs, and reports peak and mean utilisation,
energy integrated from instantaneous power, and I/O deltas. Use it around
a whole adaptation cycle to answer "what does continuous adaptation cost
while the mission is running".

``profile_inference`` runs a controlled latency benchmark with warmup,
per-image timing and percentile reporting, separately for the
deterministic forward pass (deployment cost) and the T-pass MC Dropout
sweep (acquisition cost). Use it to answer "does the deployed detector
hold the real-time budget".

Portability
-----------
Every backend is optional and probed at construction time. On the
workstation, NVML (through pynvml or nvidia-smi) supplies GPU power,
memory and utilisation. On a Jetson, ``tegrastats`` supplies the same
plus the rail-level power that NVML does not expose on Tegra, and the
sysfs INA3221 rails are read directly when present. When nothing is
available the corresponding fields are ``None`` rather than zero, so a
missing measurement can never be mistaken for a measurement of zero --
the distinction matters when the thesis has to declare a limitation.

Network accounting
------------------
"Trafego de rede" in this architecture is not generic interface traffic;
it is the payload the closed loop is obliged to move: candidate frames
uplinked for annotation, labels returned, and model weights redistributed
after each update. :class:`NetworkAccountant` tallies those logical
payloads from the pipeline's own events, and the monitor additionally
records real interface counters so the two can be cross-checked.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None

__all__ = [
    "ResourceSample",
    "ResourceReport",
    "ResourceMonitor",
    "NetworkAccountant",
    "profile_inference",
    "describe_platform",
    "is_jetson",
]


# ---------------------------------------------------------------------------
# Platform probing
# ---------------------------------------------------------------------------

def is_jetson() -> bool:
    """True when running on NVIDIA Tegra (Jetson) hardware."""
    model = Path("/proc/device-tree/model")
    try:
        if model.exists():
            text = model.read_text(errors="ignore").lower()
            if "jetson" in text or "tegra" in text or "orin" in text:
                return True
    except Exception:
        pass
    return Path("/etc/nv_tegra_release").exists()


def describe_platform() -> Dict:
    """Machine fingerprint stored with every result, for reproducibility.

    The thesis has to state where each number was measured; embedding the
    fingerprint in the result JSON means a table can never drift from the
    platform it was produced on.
    """
    info: Dict = {
        "hostname": platform.node(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "is_jetson": is_jetson(),
        "cpu_count_logical": os.cpu_count(),
    }
    if psutil is not None:
        try:
            info["cpu_count_physical"] = psutil.cpu_count(logical=False)
            info["total_ram_gb"] = round(psutil.virtual_memory().total / 1e9, 2)
        except Exception:
            pass
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            info["gpu_total_memory_gb"] = round(props.total_memory / 1e9, 2)
    except Exception:
        info["torch"] = None
    try:
        import ultralytics

        info["ultralytics"] = ultralytics.__version__
    except Exception:
        info["ultralytics"] = None
    if is_jetson():
        info["jetson_model"] = _read_text("/proc/device-tree/model")
        info["jetpack"] = _read_text("/etc/nv_tegra_release")
        info["power_mode"] = _nvpmodel_query()
    return info


def _read_text(path: str) -> Optional[str]:
    try:
        return Path(path).read_text(errors="ignore").strip("\x00\n ")
    except Exception:
        return None


def _nvpmodel_query() -> Optional[str]:
    if not shutil.which("nvpmodel"):
        return None
    try:
        out = subprocess.run(
            ["nvpmodel", "-q"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# GPU backends
# ---------------------------------------------------------------------------

class _NvmlBackend:
    """GPU telemetry via pynvml, with an nvidia-smi fallback."""

    def __init__(self, device_index: int = 0):
        self.index = device_index
        self.handle = None
        self.mode = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self.pynvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self.mode = "pynvml"
        except Exception:
            self.pynvml = None
            if shutil.which("nvidia-smi"):
                self.mode = "nvidia-smi"

    @property
    def available(self) -> bool:
        return self.mode is not None

    def sample(self) -> Dict[str, Optional[float]]:
        if self.mode == "pynvml":
            return self._sample_pynvml()
        if self.mode == "nvidia-smi":
            return self._sample_smi()
        return {}

    def _sample_pynvml(self) -> Dict[str, Optional[float]]:
        p = self.pynvml
        out: Dict[str, Optional[float]] = {}
        try:
            util = p.nvmlDeviceGetUtilizationRates(self.handle)
            out["gpu_util_pct"] = float(util.gpu)
            out["gpu_mem_util_pct"] = float(util.memory)
        except Exception:
            pass
        try:
            mem = p.nvmlDeviceGetMemoryInfo(self.handle)
            out["gpu_mem_used_mb"] = float(mem.used) / 1e6
            out["gpu_mem_total_mb"] = float(mem.total) / 1e6
        except Exception:
            pass
        try:
            out["gpu_power_w"] = float(p.nvmlDeviceGetPowerUsage(self.handle)) / 1000.0
        except Exception:
            pass
        try:
            out["gpu_temp_c"] = float(
                p.nvmlDeviceGetTemperature(self.handle, p.NVML_TEMPERATURE_GPU)
            )
        except Exception:
            pass
        return out

    def _sample_smi(self) -> Dict[str, Optional[float]]:
        query = (
            "utilization.gpu,utilization.memory,memory.used,memory.total,"
            "power.draw,temperature.gpu"
        )
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    f"--query-gpu={query}",
                    "--format=csv,noheader,nounits",
                    f"--id={self.index}",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            parts = [p.strip() for p in proc.stdout.strip().split(",")]
            keys = [
                "gpu_util_pct",
                "gpu_mem_util_pct",
                "gpu_mem_used_mb",
                "gpu_mem_total_mb",
                "gpu_power_w",
                "gpu_temp_c",
            ]
            out: Dict[str, Optional[float]] = {}
            for key, value in zip(keys, parts):
                try:
                    out[key] = float(value)
                except (TypeError, ValueError):
                    out[key] = None
            return out
        except Exception:
            return {}


class _TegrastatsBackend:
    """Jetson telemetry by parsing a background ``tegrastats`` stream.

    tegrastats is the only source of module-level power on Tegra: NVML
    reports nothing useful there, so a Jetson profiling run without this
    backend cannot report energy at all. The parser tolerates the
    formatting differences across JetPack releases by matching on rail
    names rather than field positions.
    """

    def __init__(self, interval_ms: int = 200):
        self.interval_ms = interval_ms
        self.proc: Optional[subprocess.Popen] = None
        self.samples: List[Dict[str, Optional[float]]] = []
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.available = bool(shutil.which("tegrastats")) and is_jetson()

    def start(self) -> None:
        if not self.available:
            return
        try:
            self.proc = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except Exception:
            self.available = False
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            if self._stop.is_set():
                break
            parsed = self._parse(line)
            if parsed:
                self.samples.append(parsed)

    def stop(self) -> None:
        self._stop.set()
        if self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=3)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        if self._thread:
            self._thread.join(timeout=2)

    @staticmethod
    def _parse(line: str) -> Dict[str, Optional[float]]:
        """Extract RAM, GPU load, temperature and power rails from one line."""
        out: Dict[str, Optional[float]] = {}
        tokens = line.split()
        for i, token in enumerate(tokens):
            if token == "RAM" and i + 1 < len(tokens):
                # e.g. "RAM 4096/31918MB"
                try:
                    used, total = tokens[i + 1].replace("MB", "").split("/")
                    out["ram_used_mb"] = float(used)
                    out["ram_total_mb"] = float(total)
                except Exception:
                    pass
            if token.startswith("GR3D_FREQ") or token == "GR3D_FREQ":
                candidate = token if "%" in token else (
                    tokens[i + 1] if i + 1 < len(tokens) else ""
                )
                pct = candidate.split("%")[0].split("@")[0].split("_")[-1]
                try:
                    out["gpu_util_pct"] = float(pct)
                except Exception:
                    pass
            if "@" in token and token.split("@")[0] in {
                "CPU", "GPU", "AUX", "AO", "thermal", "tj", "tboard", "SOC0",
                "SOC1", "SOC2", "CV0", "CV1", "CV2", "tdiode",
            }:
                name, _, value = token.partition("@")
                try:
                    out[f"temp_{name.lower()}_c"] = float(value.rstrip("C"))
                except Exception:
                    pass
            # Power rails: "VDD_GPU_SOC 3200mW/3100mW" or "POM_5V_GPU 1234/1100"
            if ("VDD" in token or "POM" in token or "VIN" in token) and i + 1 < len(tokens):
                rail = token.lower()
                raw = tokens[i + 1].replace("mW", "")
                try:
                    instant = float(raw.split("/")[0])
                    out[f"power_{rail}_w"] = instant / 1000.0
                except Exception:
                    pass
        if out:
            out["timestamp"] = time.time()
        return out

    def summarize(self) -> Dict[str, Optional[float]]:
        if not self.samples:
            return {}
        keys = {k for s in self.samples for k in s if k != "timestamp"}
        summary: Dict[str, Optional[float]] = {}
        for key in sorted(keys):
            values = [s[key] for s in self.samples if s.get(key) is not None]
            if not values:
                continue
            summary[f"tegra_{key}_mean"] = float(np.mean(values))
            summary[f"tegra_{key}_max"] = float(np.max(values))
        power_keys = [k for k in keys if k.startswith("power_")]
        if power_keys and len(self.samples) > 1:
            total = []
            for sample in self.samples:
                vals = [sample.get(k) for k in power_keys]
                vals = [v for v in vals if v is not None]
                if vals:
                    total.append(sum(vals))
            if total:
                duration = (
                    self.samples[-1].get("timestamp", 0)
                    - self.samples[0].get("timestamp", 0)
                )
                summary["tegra_total_power_w_mean"] = float(np.mean(total))
                summary["tegra_energy_j"] = float(np.mean(total) * max(duration, 0.0))
        return summary


# ---------------------------------------------------------------------------
# Sampling monitor
# ---------------------------------------------------------------------------

@dataclass
class ResourceSample:
    t: float
    cpu_pct: Optional[float] = None
    rss_mb: Optional[float] = None
    sys_mem_used_mb: Optional[float] = None
    gpu_util_pct: Optional[float] = None
    gpu_mem_used_mb: Optional[float] = None
    gpu_power_w: Optional[float] = None
    gpu_temp_c: Optional[float] = None


@dataclass
class ResourceReport:
    label: str
    duration_s: float
    platform: Dict = field(default_factory=dict)
    n_samples: int = 0
    interval_s: float = 0.0
    cpu_pct_mean: Optional[float] = None
    cpu_pct_peak: Optional[float] = None
    rss_mb_mean: Optional[float] = None
    rss_mb_peak: Optional[float] = None
    rss_mb_delta: Optional[float] = None
    sys_mem_used_mb_peak: Optional[float] = None
    gpu_util_pct_mean: Optional[float] = None
    gpu_util_pct_peak: Optional[float] = None
    gpu_mem_used_mb_peak: Optional[float] = None
    torch_gpu_peak_mb: Optional[float] = None
    gpu_power_w_mean: Optional[float] = None
    gpu_power_w_peak: Optional[float] = None
    gpu_energy_j: Optional[float] = None
    gpu_temp_c_peak: Optional[float] = None
    net_sent_mb: Optional[float] = None
    net_recv_mb: Optional[float] = None
    disk_read_mb: Optional[float] = None
    disk_write_mb: Optional[float] = None
    tegra: Dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict:
        return asdict(self)


class ResourceMonitor:
    """Sample host resources in the background while a block of work runs.

    Usage::

        with ResourceMonitor("cycle_3_finetune") as monitor:
            trainer.incremental_update(...)
        report = monitor.report

    The monitor is deliberately cheap: one sampling thread at 5 Hz costs
    well under 1% of a core, so profiling can stay enabled for the whole
    campaign rather than being a separate run whose numbers might not
    correspond to the run that produced the accuracy figures.
    """

    def __init__(
        self,
        label: str = "block",
        interval_s: float = 0.2,
        device_index: int = 0,
        track_gpu: bool = True,
        track_tegra: bool = True,
    ):
        self.label = label
        self.interval_s = interval_s
        self.samples: List[ResourceSample] = []
        self.report: Optional[ResourceReport] = None
        self.warnings: List[str] = []

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0 = 0.0
        self._net0 = None
        self._disk0 = None
        self._rss0: Optional[float] = None

        self._proc = None
        if psutil is not None:
            try:
                self._proc = psutil.Process(os.getpid())
            except Exception:
                self.warnings.append("psutil process handle unavailable")
        else:
            self.warnings.append(
                "psutil not installed: CPU, RSS, network and disk are unmeasured"
            )

        self._gpu = _NvmlBackend(device_index) if track_gpu else None
        if track_gpu and (self._gpu is None or not self._gpu.available):
            self.warnings.append(
                "no NVML or nvidia-smi: GPU utilisation, memory and power are unmeasured"
            )

        self._tegra = _TegrastatsBackend() if track_tegra else None
        if self._tegra is not None and not self._tegra.available and is_jetson():
            self.warnings.append("on Jetson but tegrastats is unavailable")

    # -- context manager -------------------------------------------------
    def __enter__(self) -> "ResourceMonitor":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> None:
        self._t0 = time.time()
        self._stop.clear()
        if self._proc is not None:
            try:
                self._proc.cpu_percent(interval=None)  # prime the counter
                self._rss0 = self._proc.memory_info().rss / 1e6
            except Exception:
                pass
        if psutil is not None:
            try:
                self._net0 = psutil.net_io_counters()
            except Exception:
                self._net0 = None
            try:
                self._disk0 = psutil.disk_io_counters()
            except Exception:
                self._disk0 = None
        _reset_torch_peak_memory()
        if self._tegra is not None and self._tegra.available:
            self._tegra.start()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.samples.append(self._sample())
            self._stop.wait(self.interval_s)

    def _sample(self) -> ResourceSample:
        sample = ResourceSample(t=time.time())
        if self._proc is not None:
            try:
                sample.cpu_pct = float(self._proc.cpu_percent(interval=None))
                sample.rss_mb = float(self._proc.memory_info().rss) / 1e6
            except Exception:
                pass
        if psutil is not None:
            try:
                sample.sys_mem_used_mb = float(psutil.virtual_memory().used) / 1e6
            except Exception:
                pass
        if self._gpu is not None and self._gpu.available:
            gpu = self._gpu.sample()
            sample.gpu_util_pct = gpu.get("gpu_util_pct")
            sample.gpu_mem_used_mb = gpu.get("gpu_mem_used_mb")
            sample.gpu_power_w = gpu.get("gpu_power_w")
            sample.gpu_temp_c = gpu.get("gpu_temp_c")
        return sample

    def stop(self) -> ResourceReport:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=max(2.0, self.interval_s * 5))
        if self._tegra is not None and self._tegra.available:
            self._tegra.stop()
        self.report = self._build_report()
        return self.report

    def _build_report(self) -> ResourceReport:
        duration = max(time.time() - self._t0, 1e-9)
        report = ResourceReport(
            label=self.label,
            duration_s=duration,
            platform=describe_platform(),
            n_samples=len(self.samples),
            interval_s=self.interval_s,
            warnings=list(self.warnings),
        )

        def series(attr: str) -> List[float]:
            return [
                getattr(s, attr)
                for s in self.samples
                if getattr(s, attr) is not None
            ]

        cpu = series("cpu_pct")
        if cpu:
            report.cpu_pct_mean = float(np.mean(cpu))
            report.cpu_pct_peak = float(np.max(cpu))

        rss = series("rss_mb")
        if rss:
            report.rss_mb_mean = float(np.mean(rss))
            report.rss_mb_peak = float(np.max(rss))
            if self._rss0 is not None:
                report.rss_mb_delta = float(np.max(rss) - self._rss0)

        sys_mem = series("sys_mem_used_mb")
        if sys_mem:
            report.sys_mem_used_mb_peak = float(np.max(sys_mem))

        gpu_util = series("gpu_util_pct")
        if gpu_util:
            report.gpu_util_pct_mean = float(np.mean(gpu_util))
            report.gpu_util_pct_peak = float(np.max(gpu_util))

        gpu_mem = series("gpu_mem_used_mb")
        if gpu_mem:
            report.gpu_mem_used_mb_peak = float(np.max(gpu_mem))

        report.torch_gpu_peak_mb = _torch_peak_memory_mb()

        power = [
            (s.t, s.gpu_power_w) for s in self.samples if s.gpu_power_w is not None
        ]
        if power:
            watts = [w for _, w in power]
            report.gpu_power_w_mean = float(np.mean(watts))
            report.gpu_power_w_peak = float(np.max(watts))
            if len(power) > 1:
                times = np.array([t for t, _ in power])
                report.gpu_energy_j = float(
                    np.trapezoid(np.array(watts), times)
                    if hasattr(np, "trapezoid")
                    else np.trapz(np.array(watts), times)
                )

        temps = series("gpu_temp_c")
        if temps:
            report.gpu_temp_c_peak = float(np.max(temps))

        if psutil is not None and self._net0 is not None:
            try:
                net1 = psutil.net_io_counters()
                report.net_sent_mb = (net1.bytes_sent - self._net0.bytes_sent) / 1e6
                report.net_recv_mb = (net1.bytes_recv - self._net0.bytes_recv) / 1e6
            except Exception:
                pass
        if psutil is not None and self._disk0 is not None:
            try:
                disk1 = psutil.disk_io_counters()
                report.disk_read_mb = (
                    disk1.read_bytes - self._disk0.read_bytes
                ) / 1e6
                report.disk_write_mb = (
                    disk1.write_bytes - self._disk0.write_bytes
                ) / 1e6
            except Exception:
                pass

        if self._tegra is not None and self._tegra.available:
            report.tegra = self._tegra.summarize()

        return report


def _reset_torch_peak_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _torch_peak_memory_mb() -> Optional[float]:
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated()) / 1e6
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Network accounting
# ---------------------------------------------------------------------------

class NetworkAccountant:
    """Tally the logical payloads the closed loop moves over the link.

    Raw interface counters cannot answer the board's question on their
    own, because in the offline experiments no bytes actually cross a
    network. What the thesis needs is the traffic the architecture *would*
    generate per cycle, derived from the artefacts the pipeline really
    produces: the candidate frames uplinked for human annotation, the
    label payload returned, and the updated weights redistributed.

    Sizes come from the files on disk, so the figures are measured rather
    than assumed. ``link_mbps`` converts them into transfer times, which
    is what determines whether a cycle fits inside the mission's duty
    cycle.
    """

    def __init__(self, link_mbps: float = 10.0):
        self.link_mbps = link_mbps
        self.events: List[Dict] = []

    def record(
        self,
        cycle: int,
        kind: str,
        paths: Optional[Sequence[str]] = None,
        n_items: Optional[int] = None,
        bytes_override: Optional[int] = None,
    ) -> Dict:
        """Record one transfer event.

        ``kind`` is one of ``candidate_uplink``, ``label_downlink``,
        ``weights_broadcast`` or a caller-defined label.
        """
        total_bytes = 0
        if bytes_override is not None:
            total_bytes = int(bytes_override)
        elif paths:
            for p in paths:
                try:
                    total_bytes += Path(p).stat().st_size
                except Exception:
                    continue
        event = {
            "cycle": cycle,
            "kind": kind,
            "n_items": n_items if n_items is not None else (len(paths) if paths else 0),
            "bytes": total_bytes,
            "mb": total_bytes / 1e6,
            "transfer_s": (total_bytes * 8) / (self.link_mbps * 1e6)
            if self.link_mbps > 0
            else None,
        }
        self.events.append(event)
        return event

    def summary(self) -> Dict:
        if not self.events:
            return {"total_mb": 0.0, "by_kind": {}, "events": []}
        by_kind: Dict[str, Dict[str, float]] = {}
        for event in self.events:
            bucket = by_kind.setdefault(
                event["kind"], {"mb": 0.0, "n_items": 0, "transfer_s": 0.0}
            )
            bucket["mb"] += event["mb"]
            bucket["n_items"] += event["n_items"]
            bucket["transfer_s"] += event.get("transfer_s") or 0.0
        total_mb = sum(b["mb"] for b in by_kind.values())
        cycles = {e["cycle"] for e in self.events}
        return {
            "link_mbps": self.link_mbps,
            "total_mb": total_mb,
            "total_transfer_s": sum(b["transfer_s"] for b in by_kind.values()),
            "mb_per_cycle": total_mb / max(len(cycles), 1),
            "by_kind": by_kind,
            "events": self.events,
        }


# ---------------------------------------------------------------------------
# Inference latency benchmark
# ---------------------------------------------------------------------------

def profile_inference(
    predict_fn: Callable[[str], object],
    image_paths: Sequence[str],
    warmup: int = 10,
    n_samples: int = 100,
    label: str = "deterministic",
    realtime_budget_ms: float = 33.0,
) -> Dict:
    """Benchmark a single-image inference callable.

    Reports the full distribution rather than only the mean, because an
    edge feasibility claim rests on the tail: a 5 ms mean with a 60 ms p99
    misses the real-time budget on roughly one frame in a hundred, and
    only the percentile reveals it.

    ``predict_fn`` takes one image path and performs exactly one
    inference. Pass the deterministic forward pass to measure deployment
    latency, and a closure that runs the T-pass MC Dropout sweep to
    measure acquisition overhead.
    """
    paths = list(image_paths)
    if not paths:
        return {"label": label, "error": "no images supplied"}

    # Cycle the pool when it is smaller than the requested sample count so
    # the benchmark size is the one requested rather than silently reduced.
    warm = [paths[i % len(paths)] for i in range(warmup)]
    measured = [paths[(warmup + i) % len(paths)] for i in range(n_samples)]

    for path in warm:
        try:
            predict_fn(path)
        except Exception:
            continue

    _synchronize_cuda()
    timings: List[float] = []
    with ResourceMonitor(label=f"inference_{label}", interval_s=0.1) as monitor:
        for path in measured:
            start = time.perf_counter()
            try:
                predict_fn(path)
            except Exception:
                continue
            _synchronize_cuda()
            timings.append((time.perf_counter() - start) * 1000.0)

    if not timings:
        return {"label": label, "error": "every inference call failed"}

    arr = np.array(timings)
    report = monitor.report
    energy = report.gpu_energy_j if report else None
    result = {
        "label": label,
        "n_measured": len(arr),
        "warmup": warmup,
        "latency_ms_mean": float(arr.mean()),
        "latency_ms_std": float(arr.std()),
        "latency_ms_median": float(np.median(arr)),
        "latency_ms_p90": float(np.percentile(arr, 90)),
        "latency_ms_p95": float(np.percentile(arr, 95)),
        "latency_ms_p99": float(np.percentile(arr, 99)),
        "latency_ms_min": float(arr.min()),
        "latency_ms_max": float(arr.max()),
        "throughput_fps": float(1000.0 / arr.mean()) if arr.mean() > 0 else None,
        "realtime_budget_ms": realtime_budget_ms,
        "frames_over_budget_pct": float((arr > realtime_budget_ms).mean() * 100.0),
        "meets_budget_at_p95": bool(np.percentile(arr, 95) <= realtime_budget_ms),
        "resources": report.as_dict() if report else {},
    }
    if energy is not None and len(arr) > 0:
        result["energy_per_frame_mj"] = float(energy / len(arr) * 1000.0)
    return result


def _synchronize_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def save_report(report, path: str) -> str:
    """Persist any profiling structure as JSON, creating parent dirs."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = report.as_dict() if hasattr(report, "as_dict") else report
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    return str(out)
