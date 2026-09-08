"""M3 unit tests: registry, schema validation, episodes, sampling, manifests.

All fixtures are small synthetic HDF5 files generated at runtime. These tests
never touch the network and never require real D4RL datasets.
"""

import h5py
import numpy as np
import pytest

from adaptive_plasticity import datasets

OBS_DIM = 17
ACT_DIM = 6


def write_fixture_h5(path, count=257, boundaries=((99, "terminal"), (199, "timeout"), (256, "terminal"))):
    """Write a small synthetic D4RL-shaped HDF5 file with known boundaries."""
    rng = np.random.default_rng(7)
    terminals = np.zeros(count, dtype=np.bool_)
    timeouts = np.zeros(count, dtype=np.bool_)
    for index, kind in boundaries:
        if kind == "terminal":
            terminals[index] = True
        else:
            timeouts[index] = True
    with h5py.File(path, "w") as handle:
        handle.create_dataset("observations", data=rng.normal(size=(count, OBS_DIM)))
        handle.create_dataset("actions", data=rng.normal(size=(count, ACT_DIM)))
        handle.create_dataset("rewards", data=rng.normal(size=(count,)))
        handle.create_dataset("terminals", data=terminals)
        handle.create_dataset("timeouts", data=timeouts)


def read_fixture(path):
    with h5py.File(path, "r") as handle:
        return {name: np.asarray(handle[name][()]) for name in handle.keys()}


def test_registry_covers_all_six_canonical_datasets():
    registry = datasets.registry()
    assert sorted(registry) == [
        "halfcheetah-medium-replay-v2",
        "halfcheetah-medium-v2",
        "hopper-medium-replay-v2",
        "hopper-medium-v2",
        "walker2d-medium-replay-v2",
        "walker2d-medium-v2",
    ]
    for canonical_id, entry in registry.items():
        assert entry["official_url"].startswith("http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/")
        assert entry["official_url"].endswith(entry["filename"])
        assert entry["wayback_timestamp"].isdigit()
        assert entry["archive_record_bytes"] > 0
        url = datasets.retrieval_url(canonical_id)
        assert entry["wayback_timestamp"] in url
        assert entry["filename"] in url
        assert "Minari" not in url and "mujoco/" not in url  # no Minari substitution


def test_registry_dimensions_match_mujoco_environments():
    registry = datasets.registry()
    assert (registry["halfcheetah-medium-v2"]["obs_dim"], registry["halfcheetah-medium-v2"]["act_dim"]) == (17, 6)
    assert (registry["hopper-medium-v2"]["obs_dim"], registry["hopper-medium-v2"]["act_dim"]) == (11, 3)
    assert (registry["walker2d-medium-v2"]["obs_dim"], registry["walker2d-medium-v2"]["act_dim"]) == (17, 6)


def test_schema_validation_accepts_fixture(tmp_path):
    path = tmp_path / "fixture.hdf5"
    write_fixture_h5(path)
    report = datasets.validate_schema(read_fixture(path), expected_obs_dim=OBS_DIM, expected_act_dim=ACT_DIM)
    assert report["ok"], report["issues"]
    assert report["transitions"] == 257


def test_schema_validation_rejects_bad_inputs(tmp_path):
    path = tmp_path / "fixture.hdf5"
    write_fixture_h5(path)
    data = read_fixture(path)

    without_terminals = {k: v for k, v in data.items() if k != "terminals"}
    assert datasets.validate_schema(without_terminals)["ok"] is False

    ragged = dict(data)
    ragged["actions"] = ragged["actions"][:-1]
    assert datasets.validate_schema(ragged)["ok"] is False

    assert datasets.validate_schema(data, expected_obs_dim=999)["ok"] is False

    nan_data = dict(data)
    nan_data["rewards"] = nan_data["rewards"].copy()
    nan_data["rewards"][0] = np.nan
    assert datasets.validate_schema(nan_data)["ok"] is False


def test_episode_analysis_counts_boundaries(tmp_path):
    path = tmp_path / "fixture.hdf5"
    write_fixture_h5(path)
    report = datasets.analyze_episodes(read_fixture(path))
    assert report["ok"], report["issues"]
    assert report["num_episodes"] == 3
    assert report["num_transitions"] == 257
    assert report["num_terminations"] == 2
    assert report["num_truncations"] == 1
    assert report["min_length"] == 57
    assert report["max_length"] == 100
    assert report["dangling_transitions"] == 0


def test_episode_analysis_reports_dangling_tail_as_note(tmp_path):
    path = tmp_path / "fixture.hdf5"
    write_fixture_h5(path, count=260)  # 3 flagged boundaries, 3 unflagged trailing steps
    report = datasets.analyze_episodes(read_fixture(path))
    assert report["ok"] is True  # trailing partial episode is expected D4RL behavior
    assert report["trailing_partial_length"] == 3
    assert report["dangling_transitions"] == 3
    assert any("partial episode" in note for note in report["notes"])


def test_deterministic_sampling(tmp_path):
    path = tmp_path / "fixture.hdf5"
    write_fixture_h5(path)
    data = read_fixture(path)
    first = datasets.sample_batch(data, batch_size=64, seed=123)
    second = datasets.sample_batch(data, batch_size=64, seed=123)
    third = datasets.sample_batch(data, batch_size=64, seed=999)
    for name in data:
        assert np.array_equal(first[name], second[name])
    assert not np.array_equal(first["observations"], third["observations"])
    assert first["observations"].shape == (64, OBS_DIM)
    with pytest.raises(ValueError):
        datasets.sample_batch(data, batch_size=1000, seed=1)


def test_loader_refuses_to_download_and_points_to_acquire(tmp_path):
    with pytest.raises(FileNotFoundError, match="--acquire halfcheetah-medium-v2"):
        datasets.load_local_dataset("halfcheetah-medium-v2", project_root=tmp_path)
    assert not (tmp_path / "data" / "raw" / "halfcheetah_medium-v2.hdf5").exists()


def test_loader_and_inspect_round_trip_on_synthetic_copy(tmp_path):
    target = datasets.raw_path("halfcheetah-medium-v2", project_root=tmp_path)
    write_fixture_h5(target)
    report = datasets.inspect_dataset("halfcheetah-medium-v2", project_root=tmp_path)
    assert report["ok"] is True
    assert report["schema"]["transitions"] == 257
    assert report["episodes"]["num_episodes"] == 3
    manifest = datasets.mark_manifest_validated("halfcheetah-medium-v2", report, project_root=tmp_path)
    assert manifest["validation_status"] == "validated"
    assert manifest["observed"]["transitions"] == 257
    assert manifest["observed"]["episodes"] == 3
    assert datasets.manifest_path("halfcheetah-medium-v2", project_root=tmp_path).is_file()
