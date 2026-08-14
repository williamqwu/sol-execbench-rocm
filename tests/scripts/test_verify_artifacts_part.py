# SPDX-License-Identifier: Apache-2.0
"""The acceptance gates, pointed at a part.

`verify_artifacts.py` spelled every artifact location as `ART / "04" / ...` --
task-keyed and part-blind. On a tree holding both parts that is not a missing
feature, it is a false pass: `--task 04` during MI355X bring-up read
`artifacts/04` and reported "5 checks, 0 failed" about MI350X's divergence
figure, and `--task 07` validated MI350X's spike while an MI355X one sat
unread beside it.

So what these tests pin is not "the resolver returns a path". It is the two
properties that make the gate mean something:

* the requested part's artifacts are the ones read, under both of the tree's
  layout conventions (`NN-<part>/`, and host-suffixed files inside the shared
  `artifacts/00` and `artifacts/01`); and
* a **missing** artifact for the requested part fails, naming what it looked
  for, rather than resolving to the other part's file. The fallback is the bug;
  a gate that cannot fail is worse than no gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import verify_artifacts as va  # noqa: E402


def _prov(part: str | None = None, host: str = "a-host", utc: str = "2026-01-01T00:00:00+00:00"):
    prov: dict = {"utc": utc, "git_sha": "deadbeef", "host": host}
    if part is not None:
        # Two ways an artifact names its part, and both are in the tree: the
        # explicit field, and the torch device list the older files only have.
        prov["part"] = part
        prov["torch"] = {"devices": [f"AMD Instinct {part}"]}
    return prov


def _write(p: Path, doc: dict) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc))
    return p


@pytest.fixture
def art(tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    root.mkdir()
    monkeypatch.setattr(va, "ART", root)
    return root


def _use(monkeypatch, part: str, **kw) -> va.ArtifactTree:
    tree = va.ArtifactTree(part=part, **kw)
    monkeypatch.setattr(va, "TREE", tree)
    return tree


# -- the mapping -----------------------------------------------------------

@pytest.mark.parametrize("task", ["02", "03", "04", "05", "06", "07", "08", "09", "10"])
def test_part_suffixed_directory_per_task(art, task):
    """Every task but 00/01 lives in `artifacts/NN-<part>` for a non-default part."""
    assert va.ArtifactTree("MI350X").dir(task) == art / task
    assert va.ArtifactTree("MI355X").dir(task) == art / f"{task}-MI355X"


@pytest.mark.parametrize("task", ["00", "01"])
def test_tasks_00_and_01_share_one_directory(art, task):
    """The exception: both parts' task-00/01 files sit in the same directory.

    Not normalised away by moving files -- the MI350X manifest cites
    `artifacts/00` and `artifacts/01` by path.
    """
    assert va.ArtifactTree("MI350X").dir(task) == art / task
    assert va.ArtifactTree("MI355X").dir(task) == art / task


def test_default_part_paths_are_exactly_the_old_ones(art):
    """The default resolves with no globbing and no provenance reading."""
    t = va.ArtifactTree()
    assert t.part == va.DEFAULT_PART
    assert t.path("04", "compare") == art / "04" / "compare"
    assert t.path("01", "unlocked-clock.json") == art / "01" / "unlocked-clock.json"
    assert t.shared("deferred.json") == art / "deferred.json"


def test_host_suffixed_file_resolves_for_the_other_part(art):
    """`artifacts/01/unlocked-clock-<host>.json` is task 01's MI355X artifact."""
    _write(art / "01" / "unlocked-clock.json", {"_provenance": _prov("MI350X")})
    mi355 = _write(art / "01" / "unlocked-clock-mia1-p02-g46.json",
                   {"_provenance": _prov("MI355X")})

    assert va.ArtifactTree("MI355X").path("01", "unlocked-clock.json") == mi355
    assert (va.ArtifactTree("MI350X").path("01", "unlocked-clock.json")
            == art / "01" / "unlocked-clock.json")


def test_newest_matching_host_wins_and_host_pins_it(art):
    """Two MI355X nodes in one directory: the recent one, unless --host says which.

    `artifacts/01/unlocked-clock.json` really is MI355X data from an earlier
    node, so the unsuffixed name is a candidate for MI355X too -- and must not
    win over a measurement taken on the node in front of you.
    """
    old = _write(art / "01" / "unlocked-clock.json",
                 {"_provenance": _prov("MI355X", host="g10", utc="2026-08-05T00:00:00+00:00")})
    new = _write(art / "01" / "unlocked-clock-g46.json",
                 {"_provenance": _prov("MI355X", host="g46", utc="2026-08-14T00:00:00+00:00")})

    assert va.ArtifactTree("MI355X").path("01", "unlocked-clock.json") == new
    assert va.ArtifactTree("MI355X", host="g10").path("01", "unlocked-clock.json") == old


def test_foreign_part_file_is_not_a_fallback(art):
    """The bug, pinned: MI355X missing must not resolve to the MI350X file."""
    _write(art / "01" / "stability-gpu0.json", {"_provenance": _prov("MI350X"), "cv": 0.003})

    p = va.ArtifactTree("MI355X").path("01", "stability-gpu0.json")
    assert not p.exists()
    assert va.load_json(p) is None
    assert p != art / "01" / "stability-gpu0.json"
    assert "MI355X" in va.ArtifactTree("MI355X").searched("01", "stability-gpu0.json")


def test_unattributable_file_is_kept_for_the_default_part_only(art):
    """Most MI350X artifacts predate `_provenance.part`; MI355X's do not.

    So "does not say" counts as the default part and never as the other one --
    accepting an unlabelled file for MI355X is how a fallback comes back.
    """
    _write(art / "01" / "interference.json", {"_provenance": {"utc": "x", "git_sha": "y"}})

    assert va.ArtifactTree("MI350X").path("01", "interference.json").exists()
    assert not va.ArtifactTree("MI355X").path("01", "interference.json").exists()


def test_floor_glob_cannot_mix_parts(art):
    """The `floor-gpu*.json` glob that was one MI355X run from corrupting task 01.

    Unfiltered, an unlocked MI355X floor near 1700 MHz joins MI350X's floors and
    the `F_LOCK <= min(p5)` comparison is made against the wrong node.
    """
    for gpu in range(3):
        _write(art / "01" / f"floor-gpu{gpu}.json",
               {"_provenance": _prov("MI350X"), "steady_state": {"p5_mhz": 1335}})
    _write(art / "01" / "floor-gpu0-mia1-p02-g46.json",
           {"_provenance": _prov("MI355X"), "steady_state": {"p5_mhz": 1724}})

    mi350 = va.ArtifactTree("MI350X").glob("01", "floor-gpu*.json")
    mi355 = va.ArtifactTree("MI355X").glob("01", "floor-gpu*.json")
    assert [p.name for p in mi350] == ["floor-gpu0.json", "floor-gpu1.json",
                                       "floor-gpu2.json"]
    assert [p.name for p in mi355] == ["floor-gpu0-mia1-p02-g46.json"]


def test_glob_in_a_part_scoped_directory_reads_everything(art):
    """No provenance filter where the directory already names the part."""
    _write(art / "06-MI355X" / "authoritative" / "L1__001.json", {"winner_by_workload": {}})
    got = va.ArtifactTree("MI355X").glob("06", "authoritative/*.json")
    assert [p.name for p in got] == ["L1__001.json"]


def test_artifacts_root_override(tmp_path):
    """`--artifacts-root` relocates the whole tree and still applies the part."""
    other = tmp_path / "elsewhere"
    t = va.ArtifactTree("MI355X", root=other)
    assert t.path("03", "t_sol.json") == other / "03-MI355X" / "t_sol.json"


# -- the gates themselves --------------------------------------------------

def test_check_04_fails_for_a_part_with_no_artifacts(art, monkeypatch):
    """The reported defect, end to end.

    With only MI350X's task-04 tree on disk, `--part MI355X` used to report
    5 checks / 0 failed and quote MI350X's -0.61%. It must now fail, and the
    failure must name the path it looked for.
    """
    (art / "04" / "compare").mkdir(parents=True)
    (art / "04" / "clock-domain-verification.log").write_text("ok")
    (art / "04" / "methodology-comparison.md").write_text(
        "median divergence on kernels >= 100 us: -0.61%\n")

    _use(monkeypatch, "MI350X")
    ok = va.Checks()
    va.check_04(ok)
    assert [s for s, _, _ in ok.results].count(va.FAIL) == 0
    assert any("-0.61" in d for _, _, d in ok.results)

    _use(monkeypatch, "MI355X")
    bad = va.Checks()
    va.check_04(bad)
    assert [s for s, _, _ in bad.results].count(va.FAIL) == 3
    assert not any("-0.61" in d for _, _, d in bad.results)
    assert all("04-MI355X" in d for s, _, d in bad.results if s == va.FAIL)


def test_check_07_reads_its_own_parts_spike(art, monkeypatch):
    """Task 07 validated `artifacts/07/spike.json` while the MI355X one sat unread."""
    _write(art / "07" / "spike.json", {"_provenance": _prov("MI350X"), "verdict": "no-go"})
    _write(art / "07-MI355X" / "spike.json", {"_provenance": _prov("MI355X"), "verdict": "go"})
    monkeypatch.setattr(va, "ROOT", art.parent)   # no dataset -> FP8 arm is a JUDGE

    _use(monkeypatch, "MI355X")
    c = va.Checks()
    va.check_07(c)
    verdicts = [d for _, n, d in c.results if n == "spike has an explicit verdict"]
    assert verdicts == ["go"]


def test_check_02_does_not_borrow_the_other_parts_references(art, monkeypatch):
    """A part with no reference sweep fails; it does not inherit one."""
    (art / "02" / "references-amd").mkdir(parents=True)
    _write(art / "02" / "references-amd" / "L1__001_x.json",
           {"problem": "L1__001_x", "per_workload": [{"status": "PASSED",
                                                      "methodology": "hip_events"}]})

    _use(monkeypatch, "MI355X")
    c = va.Checks()
    va.check_02(c)
    assert [(s, n) for s, n, _ in c.results] == [
        (va.FAIL, "reference sweep ran (AMD tolerances)")]
    assert "02-MI355X" in c.results[0][2]


def test_deferred_json_is_shared_by_both_parts(art, monkeypatch):
    """`deferred.json` is a decision about the dataset, not a measurement.

    It stays at `artifacts/deferred.json` for every part. Part-keying it would
    change what the coverage checks assert, which is a methodology decision and
    not this resolver's to take.
    """
    assert (va.ArtifactTree("MI355X").shared("deferred.json")
            == va.ArtifactTree("MI350X").shared("deferred.json")
            == art / "deferred.json")
