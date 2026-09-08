"""M3 offline dataset pipeline: registry, acquisition, loading, and validation.

Scope is strictly limited to reproducible handling of the six canonical D4RL
MuJoCo v2 datasets (medium and medium-replay for HalfCheetah, Hopper,
Walker2d). This module contains no RL algorithm, no training, no reward or
observation normalization, and no hidden preprocessing.

Provenance: the authoritative source is the official D4RL registry
(``Farama-Foundation/D4RL``, ``d4rl/infos.py``), which maps each ``*-v2``
dataset to an ``http://rail.eecs.berkeley.edu/...`` HDF5 file. That origin
server is currently unreachable, so retrieval uses Internet Archive Wayback
Machine snapshots of the exact official URLs (timestamps and archived byte
sizes recorded in :data:`REGISTRY`). Minari ``mujoco/*-v0`` conversions are
never substituted: they are different dataset versions.

Acquisition and loading are strictly separate. :func:`acquire_dataset` is the
only function that touches the network and must be invoked explicitly; :func:`load_local_dataset`
reads only from ``data/raw/`` and raises if the file is absent instead of
downloading anything.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import requests

CHUNK_BYTES = 1 << 20  # 1 MiB streaming chunks for downloads.

# Canonical D4RL MuJoCo v2 registry. ``official_url`` is the exact entry from
# the D4RL registry (d4rl/infos.py). ``wayback_timestamp`` identifies the
# verified Wayback snapshot (HTTP 200) of that official URL, and
# ``archive_record_bytes`` is the snapshot's CDX record length (compressed
# WARC length, informational only -- it is expected to differ slightly from
# the decompressed file size). obs_dim/act_dim are the known MuJoCo
# observation/action dimensions.
REGISTRY: dict[str, dict[str, Any]] = {
    "halfcheetah-medium-v2": {
        "filename": "halfcheetah_medium-v2.hdf5",
        "official_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_medium-v2.hdf5",
        "wayback_timestamp": "20250102175609",
        "archive_record_bytes": 237673062,
        "obs_dim": 17,
        "act_dim": 6,
    },
    "hopper-medium-v2": {
        "filename": "hopper_medium-v2.hdf5",
        "official_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium-v2.hdf5",
        "wayback_timestamp": "20250102180047",
        "archive_record_bytes": 152717255,
        "obs_dim": 11,
        "act_dim": 3,
    },
    "walker2d-medium-v2": {
        "filename": "walker2d_medium-v2.hdf5",
        "official_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_medium-v2.hdf5",
        "wayback_timestamp": "20250102175749",
        "archive_record_bytes": 232211479,
        "obs_dim": 17,
        "act_dim": 6,
    },
    "halfcheetah-medium-replay-v2": {
        "filename": "halfcheetah_medium_replay-v2.hdf5",
        "official_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/halfcheetah_medium_replay-v2.hdf5",
        "wayback_timestamp": "20250102180132",
        "archive_record_bytes": 59433929,
        "obs_dim": 17,
        "act_dim": 6,
    },
    "hopper-medium-replay-v2": {
        "filename": "hopper_medium_replay-v2.hdf5",
        "official_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_medium_replay-v2.hdf5",
        "wayback_timestamp": "20250102184654",
        "archive_record_bytes": 75843507,
        "obs_dim": 11,
        "act_dim": 3,
    },
    "walker2d-medium-replay-v2": {
        "filename": "walker2d_medium_replay-v2.hdf5",
        "official_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_medium_replay-v2.hdf5",
        "wayback_timestamp": "20250102174320",
        "archive_record_bytes": 86314700,
        "obs_dim": 17,
        "act_dim": 6,
    },
}

REQUIRED_ARRAYS = ("observations", "actions", "rewards", "terminals")
OPTIONAL_ARRAYS = ("timeouts",)
MAX_EPISODE_STEPS = 1000  # MuJoCo locomotion episode horizon.


def registry() -> dict[str, dict[str, Any]]:
    """Return a copy of the canonical dataset registry."""
    return copy.deepcopy(REGISTRY)


def raw_dir(project_root: str | Path = ".") -> Path:
    """Return the immutable raw-data directory (created on demand)."""
    path = Path(project_root) / "data" / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


def manifest_dir(project_root: str | Path = ".") -> Path:
    """Return the manifest directory (created on demand)."""
    path = Path(project_root) / "data" / "manifests"
    path.mkdir(parents=True, exist_ok=True)
    return path


def raw_path(canonical_id: str, project_root: str | Path = ".") -> Path:
    """Return the expected local path of a raw dataset file."""
    if canonical_id not in REGISTRY:
        raise KeyError(f"Unknown dataset {canonical_id!r}. Known: {sorted(REGISTRY)}")
    return raw_dir(project_root) / REGISTRY[canonical_id]["filename"]


def manifest_path(canonical_id: str, project_root: str | Path = ".") -> Path:
    """Return the expected manifest path for a dataset."""
    return manifest_dir(project_root) / f"{canonical_id}.json"


def retrieval_url(canonical_id: str) -> str:
    """Return the exact Wayback snapshot URL for a registry entry.

    The ``id_`` suffix requests the archived bytes verbatim.
    """
    entry = REGISTRY[canonical_id]
    official_https = entry["official_url"].replace("http://", "https://", 1)
    return f"https://web.archive.org/web/{entry['wayback_timestamp']}id_/{official_https}"


def sha256_of_file(path: str | Path) -> str:
    """Compute the hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def acquire_dataset(canonical_id: str, project_root: str | Path = ".") -> dict[str, Any]:
    """Explicitly download a raw dataset and write its manifest.

    This is the only network-touching function in the module. It never runs
    implicitly: loading functions raise instead of downloading. A manifest is
    written even before validation, with ``validation_status`` set to
    ``acquired-unvalidated``.
    """
    if canonical_id not in REGISTRY:
        raise KeyError(f"Unknown dataset {canonical_id!r}. Known: {sorted(REGISTRY)}")
    entry = REGISTRY[canonical_id]
    target = raw_path(canonical_id, project_root)
    url = retrieval_url(canonical_id)

    if target.is_file() and target.stat().st_size > 0:
        print(f"Raw file already present ({target}, {target.stat().st_size} bytes); skipping download.")
    else:
        print(f"Downloading {canonical_id} from {url}")
        print(f"Archive record length (reference only): {entry['archive_record_bytes']} bytes.")
        tmp_target = target.with_suffix(".hdf5.part")
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            downloaded = 0
            with open(tmp_target, "wb") as handle:
                for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    downloaded += len(chunk)
            print(f"Downloaded {downloaded} bytes to {tmp_target}")
        if downloaded == 0:
            tmp_target.unlink(missing_ok=True)
            raise ValueError(f"Download of {canonical_id} produced an empty file. Partial file removed.")
        if downloaded != entry["archive_record_bytes"]:
            print(
                f"Note: decompressed size {downloaded} differs from the archive record "
                f"length {entry['archive_record_bytes']} (record lengths are compressed; "
                "integrity is established by schema/statistics validation instead)."
            )
        tmp_target.replace(target)

    manifest = {
        "canonical_id": canonical_id,
        "filename": entry["filename"],
        "official_url": entry["official_url"],
        "retrieval_method": "wayback_snapshot_of_official_url",
        "retrieval_url": url,
        "wayback_timestamp": entry["wayback_timestamp"],
        "archive_record_bytes": entry["archive_record_bytes"],
        "local_path": str(target),
        "local_bytes": target.stat().st_size,
        "sha256": sha256_of_file(target),
        "validation_status": "acquired-unvalidated",
        "created_at": datetime.now(UTC).isoformat(),
    }
    destination = manifest_path(canonical_id, project_root)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Manifest written to {destination}")
    return manifest


def _read_arrays(h5file: h5py.File) -> dict[str, np.ndarray]:
    """Read required/optional arrays from an open HDF5 file."""
    data: dict[str, np.ndarray] = {}
    for name in REQUIRED_ARRAYS + OPTIONAL_ARRAYS:
        if name in h5file:
            data[name] = np.asarray(h5file[name][()])
    return data


def load_local_dataset(canonical_id: str, project_root: str | Path = ".") -> dict[str, np.ndarray]:
    """Load a previously acquired raw dataset into memory.

    Local-only: this function never accesses the network. It raises
    :class:`FileNotFoundError` with acquisition instructions when the raw
    file is absent.
    """
    target = raw_path(canonical_id, project_root)
    if not target.is_file():
        raise FileNotFoundError(
            f"Raw dataset {canonical_id!r} not found at {target}. "
            f"Acquire it explicitly with: python -m adaptive_plasticity.datasets --acquire {canonical_id}"
        )
    with h5py.File(target, "r") as h5file:
        missing = [name for name in REQUIRED_ARRAYS if name not in h5file]
        if missing:
            raise ValueError(f"Raw file {target} is missing required arrays: {missing}")
        return _read_arrays(h5file)


def validate_schema(
    data: dict[str, np.ndarray],
    *,
    expected_obs_dim: int | None = None,
    expected_act_dim: int | None = None,
) -> dict[str, Any]:
    """Validate the HDF5 array schema of an offline RL dataset.

    Checks key presence, two-dimensional observations/actions, one-dimensional
    rewards/terminals(/timeouts), a shared transition count, expected
    dimensions, floating/boolean dtypes, and finiteness of observations,
    actions, and rewards. No data is modified.
    """
    issues: list[str] = []
    arrays: dict[str, dict[str, Any]] = {}
    for name, array in data.items():
        arrays[name] = {"shape": list(array.shape), "dtype": str(array.dtype)}

    for name in REQUIRED_ARRAYS:
        if name not in data:
            issues.append(f"missing required array: {name}")
    if issues:
        return {"ok": False, "issues": issues, "arrays": arrays, "transitions": 0}

    observations, actions = data["observations"], data["actions"]
    rewards, terminals = data["rewards"], data["terminals"]
    count = int(observations.shape[0]) if observations.ndim else 0

    if observations.ndim != 2:
        issues.append(f"observations must be 2-D, got shape {observations.shape}")
    if actions.ndim != 2:
        issues.append(f"actions must be 2-D, got shape {actions.shape}")
    if rewards.ndim != 1 or rewards.shape[0] != count:
        issues.append(f"rewards must be 1-D with {count} entries, got shape {rewards.shape}")
    if terminals.ndim != 1 or terminals.shape[0] != count:
        issues.append(f"terminals must be 1-D with {count} entries, got shape {terminals.shape}")
    if actions.shape[0] != count:
        issues.append(f"actions first dim {actions.shape[0]} != observations first dim {count}")
    if count == 0:
        issues.append("dataset contains zero transitions")
    if "timeouts" in data and (data["timeouts"].ndim != 1 or data["timeouts"].shape[0] != count):
        issues.append(f"timeouts must be 1-D with {count} entries, got shape {data['timeouts'].shape}")

    if expected_obs_dim is not None and observations.ndim == 2 and observations.shape[1] != expected_obs_dim:
        issues.append(f"observations dim {observations.shape[1]} != expected {expected_obs_dim}")
    if expected_act_dim is not None and actions.ndim == 2 and actions.shape[1] != expected_act_dim:
        issues.append(f"actions dim {actions.shape[1]} != expected {expected_act_dim}")

    for name in ("observations", "actions", "rewards"):
        array = data[name]
        if not np.issubdtype(array.dtype, np.floating):
            issues.append(f"{name} dtype {array.dtype} is not floating")
        elif not np.all(np.isfinite(array)):
            issues.append(f"{name} contains NaN or infinite values")
    if set(np.unique(terminals)).difference({0, 1, False, True}):
        issues.append("terminals contains values other than 0/1")
    if "timeouts" in data and set(np.unique(data["timeouts"])).difference({0, 1, False, True}):
        issues.append("timeouts contains values other than 0/1")

    return {"ok": not issues, "issues": issues, "arrays": arrays, "transitions": count}


def analyze_episodes(data: dict[str, np.ndarray]) -> dict[str, Any]:
    """Analyze episode boundaries from terminal/timeout flags.

    A transition ends an episode when ``terminals`` or (when present)
    ``timeouts`` is set. Reports episode count, length statistics, and the
    termination/truncation split. A trailing run of transitions without a
    final boundary flag is a known property of the original D4RL files (the
    recording ends mid-episode); it is reported as ``trailing_partial_length``
    with an explanatory note and does not fail validation. No data is
    modified.
    """
    terminals = np.asarray(data["terminals"]).astype(bool)
    timeouts = np.asarray(data.get("timeouts", np.zeros_like(terminals))).astype(bool)
    ends = np.flatnonzero(terminals | timeouts)

    lengths: list[int] = []
    previous = -1
    for end in ends.tolist():
        lengths.append(int(end - previous))
        previous = int(end)
    trailing = int(terminals.shape[0] - 1 - previous) if previous < terminals.shape[0] - 1 else 0

    issues: list[str] = []
    notes: list[str] = []
    if trailing:
        notes.append(
            f"{trailing} trailing transitions form a final partial episode without a "
            "boundary flag (known property of original D4RL files)"
        )
    if lengths and max(lengths) > MAX_EPISODE_STEPS:
        issues.append(f"episode length {max(lengths)} exceeds horizon {MAX_EPISODE_STEPS}")
    if lengths and min(lengths) <= 0:
        issues.append("non-positive episode length detected")

    length_array = np.asarray(lengths, dtype=np.int64) if lengths else np.zeros(0, dtype=np.int64)
    return {
        "num_episodes": len(lengths),
        "num_transitions": int(terminals.shape[0]),
        "num_terminations": int(np.count_nonzero(terminals)),
        "num_truncations": int(np.count_nonzero(timeouts & ~terminals)),
        "min_length": int(length_array.min()) if lengths else 0,
        "max_length": int(length_array.max()) if lengths else 0,
        "mean_length": float(length_array.mean()) if lengths else 0.0,
        "std_length": float(length_array.std()) if lengths else 0.0,
        "has_timeouts": "timeouts" in data,
        "trailing_partial_length": trailing,
        "dangling_transitions": trailing,
        "ok": not issues,
        "issues": issues,
        "notes": notes,
    }


def sample_batch(data: dict[str, np.ndarray], batch_size: int, seed: int) -> dict[str, np.ndarray]:
    """Draw a deterministic random batch of transitions.

    Uses ``numpy.random.default_rng(seed)``; identical seeds always yield
    identical index sets. Sampling is without replacement and read-only.
    """
    count = int(data["observations"].shape[0])
    if not 0 < batch_size <= count:
        raise ValueError(f"batch_size must satisfy 0 < batch_size <= {count}, got {batch_size}")
    generator = np.random.default_rng(seed)
    indices = generator.choice(count, size=batch_size, replace=False)
    return {name: np.asarray(array)[indices] for name, array in data.items()}


def inspect_dataset(canonical_id: str, project_root: str | Path = ".") -> dict[str, Any]:
    """Build a local-only inspection report for an acquired dataset.

    Loads the raw file, validates its schema against the registry dimensions,
    analyzes episode boundaries, and summarizes rewards. Never downloads.
    """
    entry = REGISTRY[canonical_id]
    data = load_local_dataset(canonical_id, project_root)
    schema = validate_schema(data, expected_obs_dim=entry["obs_dim"], expected_act_dim=entry["act_dim"])
    episodes = analyze_episodes(data)
    rewards = np.asarray(data["rewards"], dtype=np.float64)
    manifest_file = manifest_path(canonical_id, project_root)
    report: dict[str, Any] = {
        "canonical_id": canonical_id,
        "local_path": str(raw_path(canonical_id, project_root)),
        "manifest_present": manifest_file.is_file(),
        "schema": schema,
        "episodes": episodes,
        "rewards": {
            "mean": float(rewards.mean()),
            "std": float(rewards.std()),
            "min": float(rewards.min()),
            "max": float(rewards.max()),
        },
        "ok": bool(schema["ok"] and episodes["ok"]),
    }
    return report


def mark_manifest_validated(canonical_id: str, report: dict[str, Any], project_root: str | Path = ".") -> dict[str, Any]:
    """Record inspection results in the dataset manifest."""
    destination = manifest_path(canonical_id, project_root)
    if destination.is_file():
        manifest = json.loads(destination.read_text(encoding="utf-8"))
    else:
        entry = REGISTRY[canonical_id]
        target = raw_path(canonical_id, project_root)
        manifest = {
            "canonical_id": canonical_id,
            "filename": entry["filename"],
            "official_url": entry["official_url"],
            "retrieval_method": "wayback_snapshot_of_official_url",
            "retrieval_url": retrieval_url(canonical_id),
            "wayback_timestamp": entry["wayback_timestamp"],
            "archive_record_bytes": entry["archive_record_bytes"],
            "local_path": str(target),
            "local_bytes": target.stat().st_size if target.is_file() else None,
            "sha256": sha256_of_file(target) if target.is_file() else None,
            "created_at": datetime.now(UTC).isoformat(),
        }
    manifest["observed"] = {
        "transitions": report["schema"]["transitions"],
        "obs_dim": report["schema"]["arrays"]["observations"]["shape"][1],
        "act_dim": report["schema"]["arrays"]["actions"]["shape"][1],
        "episodes": report["episodes"]["num_episodes"],
    }
    manifest["schema_ok"] = bool(report["schema"]["ok"])
    manifest["episodes_ok"] = bool(report["episodes"]["ok"])
    manifest["validation_status"] = "validated" if report["ok"] else "validation-failed"
    manifest["validated_at"] = datetime.now(UTC).isoformat()
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    """Command-line entry point: list, acquire, or inspect datasets."""
    parser = argparse.ArgumentParser(description="M3 offline dataset pipeline (explicit acquisition, local-only loading).")
    parser.add_argument("--root", type=Path, default=Path("."), help="Project root containing data/.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="List registry entries.")
    group.add_argument("--acquire", metavar="DATASET_ID", help="Download one raw dataset (explicit network use).")
    group.add_argument("--inspect", metavar="DATASET_ID", help="Validate and summarize a local dataset (no network).")
    arguments = parser.parse_args()

    if arguments.list:
        for canonical_id, entry in REGISTRY.items():
            present = raw_path(canonical_id, arguments.root).is_file()
            print(f"{canonical_id}: {entry['filename']} (archive record {entry['archive_record_bytes']} bytes) local={'yes' if present else 'no'}")
        return

    if arguments.acquire:
        acquire_dataset(arguments.acquire, arguments.root)
        return

    report = inspect_dataset(arguments.inspect, arguments.root)
    mark_manifest_validated(arguments.inspect, report, arguments.root)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
