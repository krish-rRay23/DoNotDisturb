# One reproducibility entry point. Does not train and does not download.
#
# Headless MuJoCo in this environment needs MUJOCO_GL=glfw. MUJOCO_GL=egl
# failed to import (eglQueryString missing). Override the variable when EGL
# is actually available.

export MUJOCO_GL ?= glfw
PYTHON ?= python3

.PHONY: check test dry-run hash-check claims

check: test dry-run hash-check claims

test:
	$(PYTHON) -m pytest tests/ -q -m "not needs_offline_data"

dry-run:
	$(PYTHON) experiments/dry_run_plans.py --plans-only

hash-check:
	$(PYTHON) experiments/check_manifests.py

claims:
	$(PYTHON) experiments/check_public_claims.py
