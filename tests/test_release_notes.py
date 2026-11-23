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


def test_changelog_names_pull_requests_27_through_33():
    text = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    section = text.split("### Merged pull requests #27–#33", 1)[1].split("### ", 1)[0]
    for number in range(27, 34):
        assert f"#{number}." in section
    assert "experiments/run_overnight.py" in section
    assert "15 reference cells" in section
    assert "experiments/benchmark_concurrency.py" in section
    assert "has no successor" in section
    assert "experiments/mechanistic_stress_test.py" in section
    assert "not the 24-cell actuator-cripple study" in section
    assert "CONFIRM-FIXTURE-SMOKE" in section
    assert "best_normalized" in section
    assert "SYNTHETIC FIXTURE" in section
    assert "not a result" in section
    assert "docs/run_guide.md" in section
    assert "docs/blockers.md" in section
    assert "No printed return changed." in section
    assert "No behavior changed." in section
    assert "1.4" in section and "1.44" in section
    assert "No manuscript number was added or replaced." in section


def test_changelog_names_pull_requests_34_through_37():
    text = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    section = text.split("### Merged pull requests #34–#37", 1)[1].split("### ", 1)[0]
    for number in range(34, 38):
        assert f"#{number}." in section
    assert "experiments/run_overnight.py" in section
    assert "15 reference cells" in section
    assert "experiments/benchmark_concurrency.py" in section
    assert "has no successor" in section
    assert "experiments/mechanistic_stress_test.py" in section
    assert "not the 24-cell actuator-cripple study" in section
    assert "addressable-now" in section
    assert "needs-data" in section
    assert "No printed return changed." in section
    assert "TODO: cite" in section
    assert "No bibliography entry was added." in section
    assert "ghcr.io/xu-cheng/texlive-small:20261001" in section
    assert "No score is produced." in section
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
