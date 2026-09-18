"""Comparative Concurrency Benchmark for the Optimized FINAL Study Harness.

Measures runtime, steps/sec, update throughput, GPU/CPU utilization, and speedup
with shared offline checkpoints and GPU-resident tensor sampling across 1, 2, 3, 4 workers.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import torch
from adaptive_plasticity.final_runner import run_final_cell
from adaptive_plasticity.offline_cache import ensure_all_offline_checkpoints


class SystemMonitor:
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.stop_event = threading.Event()
        self.gpu_utils: list[float] = []
        self.gpu_mems: list[float] = []
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)

    def start(self):
        self.stop_event.clear()
        self.gpu_utils.clear()
        self.gpu_mems.clear()
        self._thread.start()

    def stop(self):
        self.stop_event.set()
        self._thread.join(timeout=2.0)

    def _monitor_loop(self):
        while not self.stop_event.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                    text=True,
                    timeout=1.0,
                )
                line = out.strip().split("\n")[0]
                util_str, mem_str = [x.strip() for x in line.split(",")]
                self.gpu_utils.append(float(util_str))
                self.gpu_mems.append(float(mem_str))
            except Exception:
                pass
            time.sleep(self.interval)

    @property
    def avg_gpu_util(self) -> float:
        return sum(self.gpu_utils) / max(1, len(self.gpu_utils))

    @property
    def peak_gpu_mem(self) -> float:
        return max(self.gpu_mems) if self.gpu_mems else 0.0


def run_benchmark_optimized(
    name: str,
    workers: int,
    num_cells: int,
    online_steps: int = 1000,
    outdir_base: str = "results/benchmarks_opt",
) -> dict[str, Any]:
    outdir = Path(outdir_base) / name
    outdir.mkdir(parents=True, exist_ok=True)

    envs = ["hopper-medium-v2", "walker2d-medium-v2", "halfcheetah-medium-v2"]
    controllers = ["none", "fixed", "capacity_gate"]

    cells = []
    for i in range(num_cells):
        env = envs[i % len(envs)]
        ctrl = controllers[i % len(controllers)]
        kw = {
            "dataset_id": env,
            "controller": ctrl,
            "shift": "obs_noise" if ctrl != "none" else "none",
            "seed": i,
            "warmup_updates": 25_000,  # Full warmup updates satisfied via shared base checkpoint
            "online_steps": online_steps,
            "eval_interval": 1000,
            "eval_episodes": 2,
            "gate_interval": 500,
            "device": "cuda",
            "outdir": str(outdir),
            "use_shared_offline_checkpoints": True,
        }
        cells.append(kw)

    monitor = SystemMonitor(interval=0.5)
    monitor.start()

    t0 = time.perf_counter()
    cpu_t0 = time.process_time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_final_cell, **dict(c)) for c in cells]
        results = [f.result() for f in futures]

    wall_time = time.perf_counter() - t0
    cpu_time = time.process_time() - cpu_t0
    monitor.stop()

    total_online_steps = sum(r["online_steps"] for r in results)
    total_updates = sum(r["update_steps"] for r in results)
    runtimes = [r["runtime_seconds"] for r in results]

    summary = {
        "case_name": name,
        "workers": workers,
        "num_cells": num_cells,
        "online_steps": online_steps,
        "wall_time_s": round(wall_time, 2),
        "cpu_time_s": round(cpu_time, 2),
        "avg_cell_runtime_s": round(sum(runtimes) / len(runtimes), 2),
        "online_steps_per_sec": round(total_online_steps / wall_time, 2),
        "updates_per_sec": round(total_updates / wall_time, 2),
        "avg_gpu_util_pct": round(monitor.avg_gpu_util, 1),
        "peak_gpu_mem_mib": round(monitor.peak_gpu_mem, 1),
    }
    return summary


def main():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print("Preparing 15 shared offline base checkpoints for benchmark...")
    # Pre-train benchmark offline checkpoints if needed (using 25k updates on device)
    ensure_all_offline_checkpoints(
        envs=("hopper-medium-v2", "walker2d-medium-v2", "halfcheetah-medium-v2"),
        seeds=(0, 1, 2, 3),
        warmup_updates=25_000,
        device="cuda",
        checkpoint_dir="results/offline_checkpoints",
    )

    results = []

    print("\n--- Benchmarking Optimized Pipeline with Shared Base Checkpoints & GPU Sampling ---")
    for w in [1, 2, 3, 4]:
        print(f"\n>>> Running Optimized Pipeline: {w} worker(s), {w} cell(s) (1k online steps)...", flush=True)
        res = run_benchmark_optimized(
            f"opt_{w}w",
            workers=w,
            num_cells=w,
            online_steps=1000,
        )
        print(f"--- Result {w}w: wall_time={res['wall_time_s']}s, avg_cell={res['avg_cell_runtime_s']}s, "
              f"online_steps/s={res['online_steps_per_sec']}, GPU_util={res['avg_gpu_util_pct']}%, peak_VRAM={res['peak_gpu_mem_mib']}MB", flush=True)
        results.append(res)

    out_file = Path("results/benchmark_optimized_results.json")
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved all optimized benchmark results to {out_file}")


if __name__ == "__main__":
    main()
