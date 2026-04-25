import datetime as dt
import json
import os
import platform
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch


@dataclass
class TimingStats:
    mean_latency_ms_per_sample: float
    std_latency_ms_per_sample: float
    median_latency_ms_per_sample: float
    p95_latency_ms_per_sample: float
    throughput_samples_per_sec: float
    mean_total_elapsed_sec: float
    repeat_total_seconds: List[float]
    sample_latencies_ms: List[float]


def _run_command(cmd: List[str]) -> str:
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    except Exception:
        return ""
    return (proc.stdout or "").strip()


def _read_meminfo_gb() -> Optional[float]:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) >= 2:
                kb = float(parts[1])
                return kb / (1024.0 * 1024.0)
    return None


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def _gpu_info() -> Dict:
    if not torch.cuda.is_available():
        return {"available": False, "count": 0, "gpus": []}

    gpus = []
    driver_version = ""
    nvidia_smi = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if nvidia_smi:
        for row in nvidia_smi.splitlines():
            parts = [p.strip() for p in row.split(",")]
            if len(parts) >= 3:
                name, mem_mb, drv = parts[:3]
                driver_version = drv
                gpus.append(
                    {
                        "name": name,
                        "memory_total_mb": float(mem_mb),
                    }
                )
    else:
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            gpus.append(
                {
                    "name": props.name,
                    "memory_total_mb": float(props.total_memory / (1024.0 * 1024.0)),
                }
            )

    return {"available": True, "count": len(gpus), "driver_version": driver_version, "gpus": gpus}


def collect_machine_info(used_gpu_benchmark: bool) -> Dict:
    cudnn_version = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    try:
        import importlib.metadata as importlib_metadata
    except Exception:
        importlib_metadata = None

    packages = {}
    for pkg in ["numpy", "pandas", "torch", "scikit-learn", "xgboost", "Py6S", "matplotlib", "joblib"]:
        if importlib_metadata is None:
            continue
        try:
            packages[pkg] = importlib_metadata.version(pkg)
        except Exception:
            continue

    info = {
        "timestamp_utc": dt.datetime.utcnow().isoformat() + "Z",
        "hostname": socket.gethostname(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "platform": platform.platform(),
            "kernel": platform.uname().release,
        },
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": cudnn_version,
        "cpu": {
            "model_name": _cpu_model(),
            "physical_cores": os.cpu_count(),
            "logical_threads": os.cpu_count(),
        },
        "memory": {
            "total_ram_gb": _read_meminfo_gb(),
        },
        "gpu": _gpu_info(),
        "gpu_benchmark_used": bool(used_gpu_benchmark),
        "packages": packages,
    }
    return info


def choose_torch_device(device_name: str) -> torch.device:
    name = str(device_name).lower().strip()
    if name == "cpu":
        return torch.device("cpu")
    if name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device(name)
    raise ValueError(f"Unsupported device: {device_name}")


def synchronize_if_needed(device: Optional[torch.device]) -> None:
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_callable(
    fn: Callable[[], None],
    n_samples: int,
    repeats: int,
    warmup_runs: int,
    sync_device: Optional[torch.device] = None,
) -> TimingStats:
    for _ in range(max(0, warmup_runs)):
        fn()
        synchronize_if_needed(sync_device)

    total_times = []
    sample_latencies_ms = []
    for _ in range(max(1, repeats)):
        synchronize_if_needed(sync_device)
        t0 = time.perf_counter()
        fn()
        synchronize_if_needed(sync_device)
        t1 = time.perf_counter()
        elapsed = max(0.0, t1 - t0)
        total_times.append(elapsed)
        sample_latencies_ms.append((elapsed / max(1, n_samples)) * 1000.0)

    arr = np.asarray(sample_latencies_ms, dtype=float)
    mean_ms = float(arr.mean())
    std_ms = float(arr.std(ddof=0))
    median_ms = float(np.median(arr))
    p95_ms = float(np.percentile(arr, 95))
    throughput = float(1000.0 / max(mean_ms, 1e-12))
    return TimingStats(
        mean_latency_ms_per_sample=mean_ms,
        std_latency_ms_per_sample=std_ms,
        median_latency_ms_per_sample=median_ms,
        p95_latency_ms_per_sample=p95_ms,
        throughput_samples_per_sec=throughput,
        mean_total_elapsed_sec=float(np.mean(total_times)),
        repeat_total_seconds=[float(x) for x in total_times],
        sample_latencies_ms=[float(x) for x in sample_latencies_ms],
    )


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def markdown_table(headers: List[str], rows: List[List[object]]) -> str:
    cols = len(headers)
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * cols) + " |")
    for row in rows:
        vals = [str(v) if v is not None else "" for v in row]
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)
