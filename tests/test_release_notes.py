"""Release notes name #14-#19, and the pins and MuJoCo note stay in place."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_changelog_names_pull_requests_14_through_19():
    text = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    section = text.split("### Merged pull requests #14–#19", 1)[1].split("### ", 1)[0]
    for number in range(14, 20):
        assert f"#{number}." in section
    assert "No manuscript number was added or replaced." in section


def test_changelog_names_pull_requests_20_through_26_and_their_replacements():
    text = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    section = text.split("### Merged pull requests #20–#26", 1)[1].split("### ", 1)[0]
    for number in range(20, 27):
        assert f"#{number}." in section
    assert "experiments/run_overnight.py" in section
    assert "135-cell primary matrix" in section
    assert "15 reference cells" in section
    assert "experiments/final_factorial.py --execute --confirm-matrix CONFIRM-FINAL-150" in section
    assert "experiments/benchmark_concurrency.py" in section
    assert "nothing replaces those measurements" in section
    assert "--workers" in section
    assert "does not sweep worker counts" in section
    assert "experiments/mechanistic_stress_test.py" in section
    assert "35% neuron dormancy" in section
    assert "has no successor" in section
    assert "experiments/stress_factorial.py --execute --confirm-matrix CONFIRM-STRESS-24" in section
    assert "different experiment" in section
    assert "No manuscript number was added or replaced." in section


def test_requirements_stay_pinned_and_glfw_stays_documented():
    for line in (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        assert "==" in stripped
        assert ">=" not in stripped
    repro = (REPO_ROOT / "REPRODUCIBILITY.md").read_text(encoding="utf-8")
    assert "export MUJOCO_GL=glfw" in repro
    assert "MUJOCO_GL=egl" in repro
