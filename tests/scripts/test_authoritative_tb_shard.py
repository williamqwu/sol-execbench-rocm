# SPDX-License-Identifier: Apache-2.0
"""Card-pinned sharding of the authoritative T_b pass.

The pass may run 8-way only because STATE.md 4.4 re-times T_b and T_k back to
back on the same card, so a problem's T_b must share a card with ITS OWN T_k --
not with every other problem's T_b. That makes two things load-bearing, and
both are tested here:

  * the partition is a pure function of the problem name, so every replicate of
    a problem lands on the same card; and
  * the union of the shards is exactly the plan, with nothing duplicated and
    nothing dropped -- coverage lives in the merged output directory.

CPU-only: nothing here touches a GPU.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

_spec = importlib.util.spec_from_file_location(
    "_authoritative_tb", ROOT / "scripts" / "authoritative_tb.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

parse_shard = _mod.parse_shard
shard_of = _mod.shard_of
build_plan = _mod.build_plan
card_identity = _mod.card_identity
annotate = _mod.annotate

SCRIPT = ROOT / "scripts" / "authoritative_tb.py"


# ---------------------------------------------------------------- shard specs

@pytest.mark.parametrize("spec,want", [("0/8", (0, 8)), ("7/8", (7, 8)),
                                       ("0/1", (0, 1)), ("3/4", (3, 4))])
def test_well_formed_shard_specs_parse(spec, want):
    assert parse_shard(spec) == want


@pytest.mark.parametrize("spec", [
    "8/8",        # out of range: this shard would silently run nothing
    "-1/8",       # ditto
    "0/0",        # a zero-shard partition covers nothing
    "1/-2",
    "3",          # missing the count
    "3/",
    "/8",
    "3/8/2",
    "a/8",
    "3/b",
    "3.0/8",
    "",
])
def test_malformed_or_out_of_range_shard_specs_are_refused(spec):
    with pytest.raises(ValueError):
        parse_shard(spec)


# ----------------------------------------------------------- the partition

def _plan(n: int) -> list[str]:
    return sorted(f"L{i % 2 + 1}__{i:03d}_problem" for i in range(n))


@pytest.mark.parametrize("n_items,n_shards", [(168, 8), (235, 8), (7, 8),
                                              (8, 8), (1, 1), (100, 3)])
def test_the_partition_is_exact_and_disjoint(n_items, n_shards):
    plan = _plan(n_items)
    shards = [shard_of(plan, (i, n_shards)) for i in range(n_shards)]
    union = [k for s in shards for k in s]
    assert sorted(union) == plan            # nothing dropped
    assert len(union) == len(set(union))    # nothing duplicated
    assert sum(len(s) for s in shards) == len(plan)


def test_a_problem_lands_on_the_same_shard_every_time():
    """A pure function of the name -- so a re-run, a resume, or the later T_k
    re-timing puts the same problem back on the same card."""
    plan = _plan(168)
    where = {k: i for i in range(8) for k in shard_of(plan, (i, 8))}
    for _ in range(5):
        again = {k: i for i in range(8) for k in shard_of(plan, (i, 8))}
        assert again == where
    # And it does not depend on the order the caller happened to build it in,
    # only on the sort -- build_plan sorts, so a shuffled input is the same.
    reshuffled = sorted(reversed(plan))
    assert {k: i for i in range(8)
            for k in shard_of(reshuffled, (i, 8))} == where


def test_no_shard_means_the_whole_plan_unchanged():
    plan = _plan(20)
    assert shard_of(plan, None) == plan
    assert shard_of(plan, None) is not plan     # a copy, not the caller's list


def test_shard_of_does_not_mutate_its_input():
    plan = _plan(20)
    before = list(plan)
    shard_of(plan, (2, 8))
    assert plan == before


# --------------------------------------------------------- plan construction

def _candidate(tmp: Path, key: str, *, winner: bool = True) -> None:
    variants = {"v_baseline": {"ok": True, "all_passed": winner,
                               "latency_ms_by_workload": {"w0": 1.0}}}
    (tmp / f"{key}.json").write_text(json.dumps({"variants": variants}))


def test_build_plan_is_sorted_and_separates_the_no_winner_problems(tmp_path):
    cand = tmp_path / "candidates"
    cand.mkdir()
    for key in ["L2__020_b", "L1__001_a", "Quant__005_c"]:
        _candidate(cand, key)
    _candidate(cand, "L1__002_nowin", winner=False)

    plan, no_winner = build_plan(cand, Path("data"), top_k=2, within=0.25)
    assert [f"{p.parent.name}__{p.name}" for p, _ in plan] == [
        "L1__001_a", "L2__020_b", "Quant__005_c"]
    assert no_winner == ["L1__002_nowin"]


# ---------------------------------------------------------------- the guards

def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          capture_output=True, text=True, timeout=300)


@pytest.fixture()
def tree(tmp_path):
    cand = tmp_path / "candidates"
    cand.mkdir()
    for i in range(16):
        _candidate(cand, f"L1__{i:03d}_p")
    return cand, tmp_path / "out"


def test_a_shard_whose_card_disagrees_with_its_index_is_refused(tree):
    cand, out = tree
    r = _run("--candidates", str(cand), "--out", str(out),
             "--shard", "3/8", "--gpu", "5", "--dry-run")
    assert r.returncode != 0
    assert "--allow-gpu-shard-mismatch" in r.stderr
    assert "one card" in r.stderr        # says WHY, not just that it refused


def test_the_mismatch_override_is_honoured(tree):
    cand, out = tree
    r = _run("--candidates", str(cand), "--out", str(out), "--shard", "3/8",
             "--gpu", "5", "--allow-gpu-shard-mismatch", "--dry-run")
    assert r.returncode == 0, r.stderr
    assert "shard        3/8" in r.stdout


@pytest.mark.parametrize("spec", ["8/8", "0/0", "banana", "2"])
def test_the_cli_rejects_a_malformed_shard(tree, spec):
    cand, out = tree
    r = _run("--candidates", str(cand), "--out", str(out),
             "--shard", spec, "--gpu", "0", "--dry-run")
    assert r.returncode != 0
    assert "--shard" in r.stderr


# ------------------------------------------------------- dry run / coverage

def _dry_run_keys(cand: Path, out: Path, *extra: str) -> list[str]:
    r = _run("--candidates", str(cand), "--out", str(out), "--dry-run", *extra)
    assert r.returncode == 0, r.stderr
    return [ln.strip().split(":")[0] for ln in r.stdout.splitlines()
            if ln.startswith("  ") and ":" in ln and "/" in ln.split(":")[0]]


def test_eight_dry_runs_reconstruct_the_plan_exactly(tree):
    cand, out = tree
    shards = [_dry_run_keys(cand, out, "--shard", f"{i}/8", "--gpu", str(i))
              for i in range(8)]
    union = [k for s in shards for k in s]
    assert len(union) == 16
    assert len(set(union)) == 16
    assert sorted(union) == sorted(f"L1/{i:03d}_p" for i in range(16))


def test_the_dry_run_lists_the_whole_shard_not_a_preview(tmp_path):
    """An operator verifies the partition here, before spending GPU hours, so
    a truncated listing would be worse than none."""
    cand = tmp_path / "candidates"
    cand.mkdir()
    for i in range(80):
        _candidate(cand, f"L1__{i:03d}_p")
    keys = _dry_run_keys(cand, tmp_path / "out", "--shard", "0/8", "--gpu", "0")
    assert len(keys) == 10       # 80 / 8, and it happens to exceed no cap
    keys = _dry_run_keys(cand, tmp_path / "out", "--shard", "0/2", "--gpu", "0")
    assert len(keys) == 40       # the unsharded listing caps at 10; this must not


def test_gpu_alone_is_unchanged(tree):
    """The frozen MI350X pass used `--gpu 0` with no shard. Its plan, its
    order, and its ten-line dry-run preview must not have moved."""
    cand, out = tree
    r = _run("--candidates", str(cand), "--out", str(out), "--gpu", "0",
             "--dry-run")
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    assert lines[0] == "candidates   16 problems"
    assert lines[1] == "re-time      16 problems, 16 pending, top-2 variants each"
    assert lines[2] == "no winner    0"
    assert lines[3] == "gpu          0  (authoritative, exclusive)"
    # No shard line, and the preview still stops at ten.
    assert not any(ln.startswith("shard") for ln in lines)
    assert len([ln for ln in lines if ln.startswith("  ")]) == 10


# ------------------------------------------------------------ card identity

def test_an_unidentifiable_card_is_recorded_as_such_never_guessed(monkeypatch):
    """Prime directive 1: no fabricated identity. Prime directive 5: no silent
    omission either -- the field is present and says it failed."""
    def boom(*_a, **_k):
        raise OSError("no /dev/kfd here")
    monkeypatch.setattr(_mod.subprocess, "run", boom)
    got = card_identity("3")
    assert got["identified"] is False
    assert got["hip_visible_devices"] == "3"
    assert "no /dev/kfd here" in got["error"]
    assert "uuid" not in got and "bdf" not in got


def test_card_identity_reads_the_probe_output(monkeypatch):
    payload = {"identified": True, "uuid": "u-1", "bdf": "0000:75:00.0",
               "hostname": "node-a", "drm_card": "/sys/class/drm/card1"}

    class R:
        returncode = 0
        stdout = "some torch warning\nCARD_IDENTITY " + json.dumps(payload) + "\n"
        stderr = ""

    seen = {}

    def fake_run(cmd, **kw):
        seen["env"] = kw["env"]
        return R()

    monkeypatch.setattr(_mod.subprocess, "run", fake_run)
    got = card_identity("6")
    # Resolved for the card the children will use, not for torch device 0.
    assert seen["env"]["HIP_VISIBLE_DEVICES"] == "6"
    assert got["uuid"] == "u-1" and got["hip_visible_devices"] == "6"


def test_annotate_is_additive_and_survives_a_missing_artifact(tmp_path):
    f = tmp_path / "a.json"
    f.write_text(json.dumps({"ok": True, "gpu": "3", "variants": {"v": 1}}))
    annotate(f, {"card_identity": {"identified": True, "uuid": "u-1"},
                 "shard": {"index": 3, "count": 8}})
    doc = json.loads(f.read_text())
    assert doc["ok"] is True and doc["variants"] == {"v": 1}
    assert doc["card_identity"]["uuid"] == "u-1"
    assert doc["shard"] == {"index": 3, "count": 8}

    bad = tmp_path / "b.json"
    bad.write_text("{truncated")
    annotate(bad, {"card_identity": {}})          # must not raise
    assert bad.read_text() == "{truncated"        # and must not overwrite


def test_the_shard_note_says_coverage_is_a_merged_property():
    assert "MERGED" in _mod.SHARD_NOTE
    assert "check_coverage" in _mod.SHARD_NOTE
