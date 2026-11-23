"""Broader-domain configs stay out of the reported plans."""

from __future__ import annotations

import pytest

from adaptive_plasticity.ablations import load_all_studies
from adaptive_plasticity.datasets import REGISTRY
from adaptive_plasticity.placeholders import (
    PlaceholderNotRunnable,
    load_placeholders,
    refuse_placeholder,
)


def test_placeholders_are_not_in_the_ablation_plan():
    placeholders = load_placeholders()
    assert set(placeholders) == {"halfcheetah_medium_replay", "antmaze_umaze"}
    planned = {cell["dataset_id"] for cell in load_all_studies()}
    assert "halfcheetah-medium-replay-v2" not in planned
    assert "antmaze-umaze-v2" not in planned
    assert len(load_all_studies()) == 125


def test_replay_dataset_is_registered_and_antmaze_is_not():
    placeholders = load_placeholders()
    assert placeholders["halfcheetah_medium_replay"]["dataset_id"] in REGISTRY
    assert placeholders["antmaze_umaze"]["dataset_id"] not in REGISTRY
    for document in placeholders.values():
        assert document["status"] == "not_yet_run"
        assert document["executable"] is False


def test_refuse_placeholder_does_not_launch():
    with pytest.raises(PlaceholderNotRunnable, match="not_yet_run"):
        refuse_placeholder("antmaze_umaze")
    with pytest.raises(PlaceholderNotRunnable, match="halfcheetah-medium-replay-v2"):
        refuse_placeholder("halfcheetah_medium_replay")
