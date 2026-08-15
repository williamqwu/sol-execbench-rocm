# SPDX-License-Identifier: Apache-2.0
"""What an artifact says about the part it belongs to.

`stamp()` collected the device names and never stated a part: 2 of 14,379 JSON
artifacts under `artifacts/` carry `_provenance.part`. Everything downstream --
the gates' `ArtifactTree`, `leaderboard/ingest.py`, `score_solutions.py` -- has
been *inferring* it from the torch device list instead.

The inference is not the bug and must not be removed; measured, deleting it
takes task 03's check D on MI355X from 2078 comparisons to zero, which the gate
reports as "no submissions on disk -- untested". What these tests pin is the
statement that makes the inference checkable:

* a **declaration** (a GPU-free derivation naming its own `--part`) and a
  **detection** (a measurement naming the cards it ran on) are different claims
  and both are recorded, so neither can quietly replace the other;
* a declaration the visible cards contradict RAISES, because both explanations
  -- wrong flag, wrong node -- invalidate the artifact;
* the raise carries the block it would have written, so a caller that has
  already paid for the measurement can record the conflict instead of losing it.

Also pinned here: the wrapper bug in the manifest rebuilders. `stamp()` returns
`{"_provenance": {...}}`, and assigning onto that wrapper before wrapping it
again is why `artifacts/09/manifest-v1.1.json` and `v1.2.json` have no `utc`, no
`git_sha` and no `host` at the level every consumer reads. Those two files are
frozen release artifacts and are NOT regenerated; the scripts are fixed so the
next run stamps correctly, and these tests run them into a tmp path.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import provenance as prov  # noqa: E402


@pytest.fixture
def no_cards(monkeypatch):
    """A process that can see no GPU -- the host python every deriver runs on."""
    monkeypatch.setattr(prov, "torch_info", lambda: {"available": False})


@pytest.fixture
def mi355x(monkeypatch):
    monkeypatch.setattr(prov, "torch_info", lambda: {
        "available": True, "device_count": 2,
        "devices": ["AMD Instinct MI355X", "AMD Instinct MI355X"]})


# -- detected_part ----------------------------------------------------------

@pytest.mark.parametrize("devices,expected", [
    (["AMD Instinct MI355X"], "MI355X"),
    (["AMD Instinct MI350X", "AMD Instinct MI350X"], "MI350X"),
    (["AMD Instinct MI300X"], "MI300X"),
    ([], None),
    (["NVIDIA B200"], None),
    # Two parts in one process is not an answer. `ingest.run_part()` refuses a
    # run that spans two parts for the same reason -- returning the first would
    # attribute half the measurements to the wrong silicon.
    (["AMD Instinct MI350X", "AMD Instinct MI355X"], None),
])
def test_detected_part_reads_device_names(devices, expected):
    assert prov.detected_part(devices) == expected


def test_detected_part_uses_the_names_torch_info_already_collected(mi355x):
    """No new hardware call: the device list is already in every stamp."""
    assert prov.detected_part() == "MI355X"


def test_detected_part_is_none_with_no_torch(no_cards):
    assert prov.detected_part() is None


# -- the three keys ---------------------------------------------------------

def test_stamp_detects_when_nothing_is_declared(mi355x):
    p = prov.stamp("t")["_provenance"]
    assert (p["part"], p["part_source"], p["part_detected"]) == (
        "MI355X", "detected", "MI355X")


def test_stamp_declares_where_the_derivation_knows(no_cards):
    """`device="meta"` sees no card and the part is still a fact about the run."""
    p = prov.stamp("t", part="MI355X")["_provenance"]
    assert (p["part"], p["part_source"], p["part_detected"]) == (
        "MI355X", "declared", None)


def test_stamp_always_emits_the_keys_even_when_it_knows_nothing(no_cards):
    p = prov.stamp("t")["_provenance"]
    assert (p["part"], p["part_source"], p["part_detected"]) == (None, None, None)


def test_declaration_and_detection_are_both_kept(mi355x):
    """The statement never erases its own evidence."""
    p = prov.stamp("t", part="MI355X")["_provenance"]
    assert p["part_source"] == "declared" and p["part_detected"] == "MI355X"


def test_extra_part_is_honoured_as_a_declaration(mi355x):
    """`artifacts/01/unlocked-clock.json` set it through `extra` before the
    keyword existed; that is a caller statement like any other."""
    p = prov.stamp("t", {"part": "MI355X"})["_provenance"]
    assert (p["part"], p["part_source"]) == ("MI355X", "declared")


# -- the cross-check --------------------------------------------------------

def test_a_declaration_the_cards_contradict_raises(mi355x):
    with pytest.raises(prov.PartConflict) as e:
        prov.stamp("09-manifest", part="MI350X")
    assert "MI350X" in str(e.value) and "MI355X" in str(e.value)


def test_the_conflict_carries_the_block_it_would_have_written(mi355x):
    """So an expensive result can be recorded, marked, rather than lost."""
    with pytest.raises(prov.PartConflict) as e:
        prov.stamp("09-manifest", part="MI350X")
    block = e.value.block["_provenance"]
    assert block["part_conflict"] == {"declared": "MI350X", "detected": "MI355X"}
    assert block["part"] == "MI350X" and block["part_detected"] == "MI355X"


def test_cross_derivation_is_opt_in_and_visible(mi355x):
    """Deriving MI350X bounds on the MI355X node is legitimate -- and is a fact
    about the artifact, so it is written down rather than assumed."""
    p = prov.stamp("t", part="MI350X", allow_cross_part=True)["_provenance"]
    assert p["part"] == "MI350X" and p["part_cross_derived"] is True
    assert p["part_detected"] == "MI355X"


def test_no_conflict_when_the_declaration_agrees(mi355x):
    p = prov.stamp("t", part="MI355X")["_provenance"]
    assert "part_conflict" not in p and "part_cross_derived" not in p


def test_the_part_is_never_sourced_from_the_environment(mi355x, monkeypatch):
    """`score_solutions.py` asks `stamp()` "what part is this NODE" by reading a
    fresh stamp back. An env-sourced declaration would leak into that answer and
    make its `--part`-versus-detected comparison a tautology."""
    for var in ("SOLEXBENCH_PART", "PART", "SOLEXBENCH_TARGET_PART"):
        monkeypatch.setenv(var, "MI350X")
    p = prov.stamp("t")["_provenance"]
    assert (p["part"], p["part_source"]) == ("MI355X", "detected")


def test_write_artifact_declares(tmp_path, no_cards):
    out = prov.write_artifact(tmp_path / "a.json", "03-t-sol", {"n": 1},
                              part="MI355X")
    doc = json.loads(out.read_text())
    assert doc["_provenance"]["part"] == "MI355X"
    assert doc["_provenance"]["part_source"] == "declared"
    assert doc["n"] == 1


def test_detect_part_cli_prints_one_line():
    """A driver on a python without torch asks one that has one; empty output
    means unresolvable and must not read as an answer.

    Whatever this interpreter can see is a valid answer -- the contract under
    test is the shape (one line, one part name or nothing), because
    `agent_score._container_detected_part` parses stdout.
    """
    script = ROOT / "scripts" / "provenance.py"
    out = subprocess.run([sys.executable, str(script), "--detect-part"],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0
    assert out.stdout.strip() in ("", "MI350X", "MI355X", "MI300X")


# -- the wrapper bug in the manifest rebuilders -----------------------------

@pytest.mark.parametrize("script,source", [
    ("rebuild_manifest_v11.py", "artifacts/09/manifest-v1.json"),
    ("rebuild_manifest_v12.py", "artifacts/09/manifest-v1.1.json"),
])
def test_a_rebuilt_manifest_stamps_at_the_level_consumers_read(
        tmp_path, script, source):
    """`prov = stamp(...)` then `prov["part"] = ...` writes onto the WRAPPER.

    The shipped v1.1 and v1.2 read `_provenance._provenance`, so `utc`,
    `git_sha`, `host` and `python` are all None to `ingest.py` and to every
    gate, and `part` is None because v1 never declared one either. Written to a
    tmp path: the release artifacts are frozen and regenerating them is a
    release decision, not a test.
    """
    if not (ROOT / source).exists():
        pytest.skip(f"{source} not in this tree")
    out = tmp_path / "manifest.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / script),
                        "--out", str(out)],
                       capture_output=True, text=True, cwd=ROOT, timeout=1800)
    assert r.returncode == 0, r.stderr[-2000:]
    p = json.loads(out.read_text())["_provenance"]
    assert "_provenance" not in p, "stamped onto the wrapper again"
    for key in ("utc", "git_sha", "host", "python", "task"):
        assert p.get(key), f"{key} lost at the level consumers read"
    # The part comes from the source manifest, which states it in one of the two
    # conventions or in its device names -- never from this host, which is
    # irrelevant to arithmetic over another part's numbers.
    assert p["part"] == "MI350X"
    assert p["part_source"] == "declared"
