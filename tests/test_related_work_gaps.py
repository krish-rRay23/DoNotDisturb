"""Related work uses bibliography keys that already exist and marks the rest TODO."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEX_PATH = REPO_ROOT / "research_paper" / "iclr2027" / "iclr2027_conference.tex"
BIB_PATH = REPO_ROOT / "research_paper" / "iclr2027" / "iclr2027_conference.bib"

EXISTING_KEYS = {
    "kostrikov2021iql",
    "fu2020d4rl",
    "kumar2020cql",
    "fujimoto2021td3bc",
    "dohare2024loss",
    "sokar2023dormant",
    "kumar2020implicit",
    "lyle2023understanding",
    "lyle2024plasticity",
    "ash2020warmstarting",
    "nikishin2022primacy",
    "agarwal2021rliable",
    "haarnoja2018sac",
    "fujimoto2018td3",
    "lee2022offline2online",
    "doro2023sample",
    "kumar2022dr3",
    "ball2023efficient",
    "schwarzer2024bigger",
    "nair2020awac",
    "agarwal2021deep",
    "roy2007effective",
}


def test_related_work_cites_unused_keys_and_marks_the_gaps():
    tex = TEX_PATH.read_text(encoding="utf-8")
    related = tex.split(r"\section{Related Work", 1)[1].split(r"\section{", 1)[0]
    assert "lyle2024plasticity" in related
    assert "kumar2020cql" in related
    assert "schwarzer2024bigger" in related
    assert "sokar2023dormant, schwarzer2024bigger" not in tex
    for marker in (
        "TODO: cite Cal-QL",
        "TODO: cite PEX",
        "TODO: cite RO2O",
        "TODO: cite other dormancy criteria",
    ):
        assert marker in related
    keys = set(re.findall(r"@\w+\{([^,\s]+)", BIB_PATH.read_text(encoding="utf-8")))
    assert keys == EXISTING_KEYS
