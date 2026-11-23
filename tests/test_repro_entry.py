"""The single reproducibility entry point does not train or rewrite manifests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import check_manifests
from adaptive_plasticity.datasets import raw_path, sha256_of_file


def test_requirements_pin_every_direct_dependency():
    """No requirement line is left as a range. Pins that are already present stay."""
    for line in (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        assert "==" in stripped
        assert ">=" not in stripped
        assert "~=" not in stripped
        assert not stripped.endswith("*")


def test_make_check_runs_tests_plans_and_hashes_with_glfw():
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "export MUJOCO_GL ?= glfw" in makefile
    assert "experiments/dry_run_plans.py --plans-only" in makefile
    assert "experiments/check_manifests.py" in makefile
    assert "experiments/check_public_claims.py" in makefile
    assert "experiments/check_claim_coverage.py" in makefile
    assert '-m "not needs_offline_data"' in makefile
    # The default target chains the reproducibility checks.
    assert "check: test dry-run hash-check claims claim-coverage" in makefile


def test_hash_check_is_read_only_and_distinguishes_mismatch(tmp_path):
    target = raw_path("hopper-medium-v2", tmp_path)
    target.write_bytes(b"synthetic-fixture-not-a-dataset")
    manifest = tmp_path / "data" / "manifests" / "hopper-medium-v2.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text('{"sha256": "' + sha256_of_file(target) + '"}', encoding="utf-8")
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    matched = check_manifests.summarize_hashes(tmp_path)
    assert matched["status"] == 0
    assert "hopper-medium-v2" not in matched["mismatches"]
    assert "walker2d-medium-v2" in matched["absent"]
    assert "Read-only" in check_manifests.format_report(matched)
    assert "38.78" not in check_manifests.format_report(matched)

    manifest.write_text('{"sha256": "' + ("ab" * 32) + '"}', encoding="utf-8")
    mismatched = check_manifests.summarize_hashes(tmp_path)
    assert mismatched["status"] == 1
    assert mismatched["mismatches"] == ["hopper-medium-v2"]

    required = check_manifests.summarize_hashes(tmp_path, require_present=True)
    assert required["status"] == 1

    manifest.write_text('{"sha256": "' + sha256_of_file(target) + '"}', encoding="utf-8")
    missing_required = check_manifests.summarize_hashes(tmp_path, require_present=True)
    assert missing_required["status"] == 2

    with pytest.raises(SystemExit) as raised:
        check_manifests.main(["--root", str(tmp_path), "--require-present"])
    assert raised.value.code == 2

    after = {path for path in tmp_path.rglob("*") if path.is_file()}
    assert after == set(before)
    assert manifest.read_text(encoding="utf-8") == '{"sha256": "' + sha256_of_file(target) + '"}'


def test_present_file_without_a_sha256_is_not_treated_as_absent(tmp_path):
    target = raw_path("halfcheetah-medium-v2", tmp_path)
    target.write_bytes(b"present-but-unverified")
    missing_manifest = check_manifests.summarize_hashes(tmp_path)
    assert missing_manifest["status"] == 1
    assert missing_manifest["unverified"] == ["halfcheetah-medium-v2"]
    assert "halfcheetah-medium-v2" not in missing_manifest["absent"]
    assert "halfcheetah-medium-v2: unverified" in check_manifests.format_report(missing_manifest)
    assert "walker2d-medium-v2: absent" in check_manifests.format_report(missing_manifest)

    manifest = tmp_path / "data" / "manifests" / "halfcheetah-medium-v2.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text('{"sha256": ""}', encoding="utf-8")
    empty_digest = check_manifests.summarize_hashes(tmp_path)
    assert empty_digest["status"] == 1
    assert empty_digest["unverified"] == ["halfcheetah-medium-v2"]
    assert manifest.read_text(encoding="utf-8") == '{"sha256": ""}'
