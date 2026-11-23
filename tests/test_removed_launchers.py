"""Unguarded training scripts stay out of experiments/."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

REMOVED = (
    "experiments/run_overnight.py",
    "experiments/benchmark_concurrency.py",
    "experiments/mechanistic_stress_test.py",
)


def test_unguarded_training_launchers_are_removed():
    for relative in REMOVED:
        assert not (REPO_ROOT / relative).exists()
    readme = (REPO_ROOT / "experiments" / "README.md").read_text(encoding="utf-8")
    for name in ("run_overnight.py", "benchmark_concurrency.py", "mechanistic_stress_test.py"):
        assert name not in readme
