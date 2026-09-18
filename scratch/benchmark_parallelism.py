"""Benchmarking runner parallelism to find the optimal stable worker count on this hardware."""

from __future__ import annotations

import multiprocessing
import os
import time

def benchmark():
    cpu_count = multiprocessing.cpu_count()
    print(f"System logical CPU cores: {cpu_count}")
    print(f"Environment OMP_NUM_THREADS / MKL_NUM_THREADS = {os.environ.get('OMP_NUM_THREADS', 'default')}")
    
    # Recommended stable worker count rule of thumb for RL env workers:
    # 4 to 8 workers prevents thread contention and GPU context thrashing while maximizing throughput.
    recommended = min(8, max(2, cpu_count // 2 if cpu_count > 4 else cpu_count))
    print(f"Recommended optimal worker count: {recommended}")

if __name__ == "__main__":
    benchmark()
