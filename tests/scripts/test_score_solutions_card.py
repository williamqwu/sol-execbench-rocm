# SPDX-License-Identifier: Apache-2.0
"""Card-matched scoring: the partition, the refusal, the basis, the part.

`STATE.md` 4.4 re-times `T_b` and `T_k` back to back on ONE card, and
`scripts/authoritative_tb.py` therefore runs 8-way with per-problem card
pinning. Scoring has to land each solution on the card holding its own
problem's anchor, and -- because a cross-card score is invisible in the output
-- has to *refuse* the ones that do not, rather than merely making the right
layout available.

Four properties, one per section:

  * the partition is the one `authoritative_tb.py` actually used, for the same
    N, and is still an exact disjoint cover;
  * a card mismatch is refused and counted, never scored;
  * with `T_SOL` present and `T_b` absent every record says it was published on
    the `sol_headroom` basis, and `backfill_scores.py` raises it to
    `sol_score_v1` only when the anchor's card checks out;
  * part resolution never falls back to the MI350X tree on an MI355X node.

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
sys.path.insert(0, str(ROOT / "src"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ss = _load("_score_solutions", ROOT / "scripts" / "score_solutions.py")
bf = _load("_backfill_scores", ROOT / "scripts" / "backfill_scores.py")
atb = _load("_authoritative_tb", ROOT / "scripts" / "authoritative_tb.py")

SCRIPT = ROOT / "scripts" / "score_solutions.py"

CARD_A = {"identified": True, "hostname": "mia1-p02-g46", "bdf": "0000:15:00.0",
          "uuid": "aaaa-1", "drm_card": "/sys/class/drm/card9"}
CARD_B = {"identified": True, "hostname": "mia1-p02-g46", "bdf": "0000:65:00.0",
          "uuid": "bbbb-2", "drm_card": "/sys/class/drm/card17"}
CARD_A_OTHER_HOST = {**CARD_A, "hostname": "mia1-p02-g45"}


def _plan(n: int) -> list[str]:
    return sorted(f"L{i % 2 + 1}__{i:03d}_problem" for i in range(n))


def _anchor(tb_dir: Path, key: str, *, card: dict | None = CARD_A,
            shard: tuple[int, int] | None = None) -> None:
    tb_dir.mkdir(parents=True, exist_ok=True)
    doc: dict = {"problem": key, "ok": True}
    if card is not None:
        doc["card_identity"] = card
    if shard is not None:
        doc["shard"] = {"index": shard[0], "count": shard[1],
                        "gpu": str(shard[0])}
    (tb_dir / f"{key}.json").write_text(json.dumps(doc))


# ------------------------------------------------------------- the partition

def test_the_partition_is_the_one_authoritative_tb_used(tmp_path):
    """Not a second implementation of the same idea -- the same assignment.

    Every problem goes to the shard its own artifact records, which is where
    `authoritative_tb.py` put it, so the T_k lands on the card holding the T_b.
    """
    plan = _plan(64)
    tb = tmp_path / "authoritative"
    want = {}
    for i in range(8):
        for key in atb.shard_of(plan, (i, 8)):
            _anchor(tb, key, shard=(i, 8))
            want.setdefault(i, []).append(key)

    for i in range(8):
        assert ss.partition_problems(plan, (i, 8), tb) == sorted(want[i])


@pytest.mark.parametrize("n_anchored", [0, 1, 30, 64])
def test_the_partition_is_exact_and_disjoint(tmp_path, n_anchored):
    """Whatever mix of anchored and not-yet-anchored problems exists mid-pass,
    the union over the shards is exactly the input. Coverage lives in the
    merged output; a partition that loses a problem loses it silently."""
    plan = _plan(64)
    tb = tmp_path / "authoritative"
    tb.mkdir()
    for i in range(8):
        for key in atb.shard_of(plan, (i, 8)):
            if plan.index(key) < n_anchored:
                _anchor(tb, key, shard=(i, 8))

    shards = [ss.partition_problems(plan, (i, 8), tb) for i in range(8)]
    union = [k for s in shards for k in s]
    assert sorted(union) == plan
    assert len(union) == len(set(union))


def test_an_unanchored_problem_falls_back_to_authoritative_tbs_stride(tmp_path):
    """No anchor means no card to match, so the fallback only has to be the
    same convention -- and it is literally `authoritative_tb.shard_of`."""
    plan = _plan(24)
    tb = tmp_path / "authoritative"
    tb.mkdir()
    for i in range(8):
        assert ss.partition_problems(plan, (i, 8), tb) == atb.shard_of(plan, (i, 8))


def test_no_shard_means_every_problem(tmp_path):
    plan = _plan(24)
    assert ss.partition_problems(plan, None, tmp_path) == plan


def test_a_shard_recorded_under_a_different_count_is_not_reused(tmp_path):
    """A 4-way pass says nothing about where an 8-way pass would have put a
    problem. Reusing the index would put T_k on a card chosen by arithmetic
    rather than by the anchor."""
    tb = tmp_path / "authoritative"
    _anchor(tb, "L1__001_p", shard=(2, 4))
    assert ss.anchor_shard(tb, "L1__001_p", 8) is None
    assert ss.anchor_shard(tb, "L1__001_p", 4) == 2


def test_anchor_shard_survives_a_missing_or_broken_artifact(tmp_path):
    tb = tmp_path / "authoritative"
    tb.mkdir()
    assert ss.anchor_shard(tb, "nope", 8) is None
    (tb / "broken.json").write_text("{truncated")
    assert ss.anchor_shard(tb, "broken", 8) is None
    assert ss.anchor_shard(None, "anything", 8) is None


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          capture_output=True, text=True, timeout=300)


def test_the_cli_refuses_a_shard_whose_card_disagrees_with_its_index():
    r = _run("--run-id", "nope", "--shard", "3/8", "--gpu", "5", "--dry-run")
    assert r.returncode != 0
    assert "--allow-gpu-shard-mismatch" in r.stderr
    assert "does not hold its own anchor" in r.stderr


@pytest.mark.parametrize("spec", ["8/8", "0/0", "banana", "2", "3.0/8"])
def test_the_cli_refuses_a_malformed_shard(spec):
    r = _run("--run-id", "nope", "--shard", spec, "--gpu", "0", "--dry-run")
    assert r.returncode != 0
    assert "--shard" in r.stderr


# ---------------------------------------------------------- the card refusal

def test_the_matching_card_is_accepted_and_says_so(tmp_path):
    tb = tmp_path / "authoritative"
    _anchor(tb, "L1__001_p", card=CARD_A)
    anchor, note = ss.anchor_card(tb, "L1__001_p")
    v = ss.card_verdict(anchor, CARD_A, t_b_in_scope=True, anchor_note=note)
    assert v["ok"] and v["state"] == "matched"


@pytest.mark.parametrize("other,label", [
    (CARD_B, "a different card on the same node"),
    (CARD_A_OTHER_HOST, "the same BDF on a different node"),
    ({**CARD_A, "uuid": "cccc-3"}, "a re-enumerated BDF"),
])
def test_a_card_mismatch_is_refused(tmp_path, other, label):
    tb = tmp_path / "authoritative"
    _anchor(tb, "L1__001_p", card=CARD_A)
    anchor, note = ss.anchor_card(tb, "L1__001_p")
    v = ss.card_verdict(anchor, other, t_b_in_scope=True, anchor_note=note)
    assert not v["ok"], label
    assert v["state"] == "mismatch"
    assert "4.4" in v["reason"]


@pytest.mark.parametrize("anchor_card_doc,actual,why", [
    (None, CARD_A, "no anchor artifact at all"),
    ({"identified": False, "error": "no /dev/kfd"}, CARD_A,
     "the anchor's card could not identify itself"),
    (CARD_A, {"identified": False, "error": "boom"},
     "this process's card could not identify itself"),
    (CARD_A, None, "no identity for this process"),
    ({"identified": True, "hostname": "h", "bdf": "", "uuid": "u"},
     {"identified": True, "hostname": "h", "bdf": "", "uuid": "u"},
     "a partial identity is not a match even when it is equal"),
])
def test_an_unprovable_card_is_refused_not_assumed(tmp_path, anchor_card_doc,
                                                   actual, why):
    """Prime directive 1, in the negative: an identity that cannot be
    established is never filled in with the plausible one."""
    tb = tmp_path / "authoritative"
    if anchor_card_doc is not None:
        _anchor(tb, "L1__001_p", card=anchor_card_doc)
    anchor, note = ss.anchor_card(tb, "L1__001_p")
    v = ss.card_verdict(anchor, actual, t_b_in_scope=True, anchor_note=note)
    assert not v["ok"], why
    assert v["state"] == "unverifiable"


def test_with_no_t_b_in_scope_the_check_is_not_applicable_not_passed(tmp_path):
    """`0 refused` must not be readable as `all checked`. A record with no T_b
    makes no cross-card claim, and that is a third state, not a pass."""
    v = ss.card_verdict(None, None, t_b_in_scope=False, anchor_note="")
    assert v["ok"] and v["state"] == "not_applicable"


def test_anchor_card_reports_how_it_failed(tmp_path):
    tb = tmp_path / "authoritative"
    tb.mkdir()
    assert ss.anchor_card(None, "k")[0] is None
    assert "no authoritative T_b directory" in ss.anchor_card(None, "k")[1]
    assert "no authoritative T_b artifact" in ss.anchor_card(tb, "k")[1]
    (tb / "k.json").write_text("{truncated")
    assert "unreadable" in ss.anchor_card(tb, "k")[1]
    (tb / "k.json").write_text(json.dumps({"ok": True}))
    assert "no card_identity block" in ss.anchor_card(tb, "k")[1]


# ------------------------------------------------- the refusal, end to end

def _fake_prov(part_device: str = "AMD Instinct MI355X") -> dict:
    return {"_provenance": {"task": "10-scoring", "utc": "2026-08-14T00:00:00Z",
                            "git_sha": "deadbeef", "host": "mia1-p02-g46",
                            "f_lock_mhz": None,
                            "torch": {"devices": [part_device]}}}


def _run_tree(tmp_path, problems: list[str]) -> Path:
    run_root = tmp_path / "artifacts" / "10" / "runs" / "t1"
    for key in problems:
        d = run_root / "claude-code" / key
        (d / "packet").mkdir(parents=True)
        (d / "session.json").write_text(json.dumps(
            {"harness": "claude-code", "produced_solution": True}))
        (d / "packet" / ".packet.json").write_text(
            json.dumps({"problem": key.replace("__", "/", 1)}))
    return run_root


def _bounds_file(path: Path, keys: list[str], field: str, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "part": "MI355X",
        "problems": {k: {"workloads": {"w0": {field: value}}} for k in keys},
    }))


@pytest.fixture()
def scoring_main(tmp_path, monkeypatch):
    """`score_solutions.main()` wired to a temp tree, with no GPU anywhere."""
    monkeypatch.setattr(ss, "ROOT", tmp_path)
    monkeypatch.setattr(ss, "stamp", lambda *a, **k: _fake_prov())
    monkeypatch.setattr(ss, "clock_lock_state",
                        lambda *a, **k: {"locked": False,
                                         "performance_level": "auto"})
    monkeypatch.setenv("SOLEXBENCH_CLOCK_BASIS", "unlocked")
    return ss


def test_a_card_mismatch_is_refused_and_counted_never_scored(
        tmp_path, monkeypatch, scoring_main):
    keys = ["L1__001_p", "L1__002_p"]
    _run_tree(tmp_path, keys)
    tb = tmp_path / "authoritative"
    _anchor(tb, "L1__001_p", card=CARD_A)      # this card: scored
    _anchor(tb, "L1__002_p", card=CARD_B)      # another card: refused
    _bounds_file(tmp_path / "t_sol.json", keys, "t_sol_ms", 0.1)
    _bounds_file(tmp_path / "t_b.json", keys, "t_b_ms", 1.0)

    monkeypatch.setattr(ss, "card_identity", lambda gpu, **k: dict(CARD_A))
    scored: list[str] = []

    def fake_score_one(session_dir, *a, **kw):
        scored.append(session_dir.name)
        return {"problem": session_dir.name, "harness": "claude-code",
                "outcome": "evaluated", "workloads": 1, "passed": 1,
                "reference_copy": {"kind": "distinct"},
                "records": [{"workload_uuid": "w0", "score_basis": "sol_score_v1"}]}

    monkeypatch.setattr(ss, "score_one", fake_score_one)
    monkeypatch.setattr(sys, "argv", [
        "score_solutions.py", "--run-id", "t1", "--gpu", "0",
        "--t-sol", str(tmp_path / "t_sol.json"),
        "--t-b", str(tmp_path / "t_b.json"),
        "--tb-artifacts", str(tb), "--workloads-root", "none"])
    assert ss.main() == 0

    # The mismatched problem was never evaluated ...
    assert scored == ["L1__001_p"]
    out = tmp_path / "artifacts" / "10" / "scores" / "t1" / "claude-code"
    refused = json.loads((out / "L1__002_p.json").read_text())
    # ... it produced a record of its own, with a reason ...
    assert refused["outcome"] == "refused_card_mismatch"
    assert refused["card_check"]["state"] == "mismatch"
    assert refused["records"] == []
    assert "0000:65:00.0" in refused["note"]
    # ... and it is counted, not merely absent.
    summary = json.loads((out.parent / "summary.json").read_text())
    assert summary["outcomes"]["refused_card_mismatch"] == 1
    assert summary["card_enforcement"]["refused"] == 1
    assert summary["card_enforcement"]["matched"] == 1
    assert summary["card_enforcement"]["refusals"][0]["problem"] == "L1__002_p"


def test_an_unidentifiable_scoring_card_refuses_everything_with_a_t_b(
        tmp_path, monkeypatch, scoring_main):
    keys = ["L1__001_p"]
    _run_tree(tmp_path, keys)
    tb = tmp_path / "authoritative"
    _anchor(tb, "L1__001_p", card=CARD_A)
    _bounds_file(tmp_path / "t_sol.json", keys, "t_sol_ms", 0.1)
    _bounds_file(tmp_path / "t_b.json", keys, "t_b_ms", 1.0)
    monkeypatch.setattr(ss, "card_identity",
                        lambda gpu, **k: {"identified": False, "error": "boom"})
    monkeypatch.setattr(ss, "score_one", lambda *a, **k: pytest.fail(
        "scored a solution whose card could not be established"))
    monkeypatch.setattr(sys, "argv", [
        "score_solutions.py", "--run-id", "t1", "--gpu", "0",
        "--t-sol", str(tmp_path / "t_sol.json"),
        "--t-b", str(tmp_path / "t_b.json"),
        "--tb-artifacts", str(tb), "--workloads-root", "none"])
    assert ss.main() == 0
    summary = json.loads((tmp_path / "artifacts" / "10" / "scores" / "t1"
                          / "summary.json").read_text())
    assert summary["card_enforcement"]["refused"] == 1


def test_without_a_t_b_nothing_is_refused_but_nothing_claims_a_card_either(
        tmp_path, monkeypatch, scoring_main):
    """The pre-anchor pass must still run: there is no card to match yet."""
    keys = ["L1__001_p"]
    _run_tree(tmp_path, keys)
    _bounds_file(tmp_path / "t_sol.json", keys, "t_sol_ms", 0.1)
    monkeypatch.setattr(ss, "card_identity", lambda gpu, **k: dict(CARD_A))
    seen = {}

    def fake_score_one(session_dir, *a, **kw):
        seen["card_check"] = kw["card_check"]
        return {"problem": session_dir.name, "harness": "claude-code",
                "outcome": "evaluated", "workloads": 1, "passed": 1,
                "reference_copy": {"kind": "distinct"}, "records": []}

    monkeypatch.setattr(ss, "score_one", fake_score_one)
    monkeypatch.setattr(sys, "argv", [
        "score_solutions.py", "--run-id", "t1", "--gpu", "0",
        "--t-sol", str(tmp_path / "t_sol.json"), "--workloads-root", "none"])
    assert ss.main() == 0
    assert seen["card_check"]["state"] == "not_applicable"
    summary = json.loads((tmp_path / "artifacts" / "10" / "scores" / "t1"
                          / "summary.json").read_text())
    assert summary["card_enforcement"] == {
        "matched": 0, "refused": 0, "not_applicable": 1,
        "by_state": {"not_applicable": 1}, "refusals": []}


# ------------------------------------------------------------- the basis

REAL_T_SOL = ROOT / "artifacts" / "03-MI355X" / "t_sol.json"


@pytest.mark.skipif(not REAL_T_SOL.exists(), reason="no MI355X T_SOL artifact")
def test_the_real_mi355x_t_sol_supports_the_sol_headroom_basis():
    """The ladder's lower rung has to be reachable from what is on disk.

    `t_sol_ms` on this part is derived with `f_lock_mhz: null` -- the bound
    travels as separately-scalable terms (STATE.md, decision 1) -- so this
    asserts the field a `sol_headroom` record needs is actually populated,
    rather than assuming it.
    """
    t_sol, meta = ss.load_bounds(REAL_T_SOL, "MI355X")
    assert not meta.get("rejected")
    assert len(t_sol) >= 200
    key = "L1__069_rms_norm"
    uuid = sorted(t_sol[key]["workloads"])[0]
    assert ss.workload_bound(t_sol, key, uuid, "t_sol_ms") > 0


@pytest.mark.parametrize("t_b_ms,want", [(None, "sol_headroom"),
                                         (1.0, "sol_score_v1")])
def test_the_basis_is_whatever_the_inputs_actually_support(t_b_ms, want):
    from solexbench_agents.scoring import resolve_basis
    assert resolve_basis(correct=True, t_k_ms=0.5, t_ref_ms=0.9,
                         t_sol_ms=0.1, t_b_ms=t_b_ms).value == want


REAL_SESSION = (ROOT / "artifacts" / "10" / "runs" / "full-01" / "claude-code"
                / "L1__069_rms_norm")
REAL_PROBLEM = ROOT / "data" / "SOL-ExecBench" / "benchmark" / "L1" / "069_rms_norm"
REAL_WORKLOADS = ROOT / "artifacts" / "05-MI355X" / "workloads"


@pytest.mark.skipif(
    not (REAL_SESSION.exists() and REAL_PROBLEM.exists() and REAL_T_SOL.exists()),
    reason="needs the full-01 run, the dataset and the MI355X T_SOL")
def test_every_record_says_sol_headroom_when_there_is_no_t_b(monkeypatch):
    """The pre-anchor rung of the ladder, on a real harvested solution.

    Only the timing is faked -- the packet, the solution loader, the
    authoritative spec and the T_SOL artifact are the real ones -- because the
    thing under test is what basis a record is published under, and that is
    arithmetic over inputs that exist today. No GPU is touched.
    """
    import _common

    per_workload = [{"index": 0, "workload_uuid": None, "status": "PASSED",
                     "latency_ms": 0.5, "reference_latency_ms": 0.9,
                     "methodology": "hip_events", "clock_bracket": None,
                     "reference_clock_bracket": None, "log": ""}]
    monkeypatch.setattr(_common, "evaluate", lambda *a, **k: [])
    monkeypatch.setattr(_common, "summarize", lambda traces: {
        "workloads": 1, "passed": 1, "all_passed": True,
        "per_workload": per_workload})

    t_sol, _ = ss.load_bounds(REAL_T_SOL, "MI355X")
    result = ss.score_one(
        REAL_SESSION, REAL_PROBLEM,
        REAL_WORKLOADS if REAL_WORKLOADS.exists() else None,
        t_sol=t_sol, t_b={}, part="MI355X", timeout=60, gpu=0,
        lock_clocks=True,
        card_check={"ok": True, "state": "not_applicable", "actual": CARD_A})

    assert result["outcome"] == "evaluated", result.get("note")
    assert result["records"], "a passing workload produced no record"
    for rec in result["records"]:
        assert rec["score_basis"] == "sol_headroom"
        assert rec["t_sol_ms"] > 0        # the real MI355X bound, not a stub
        assert rec["t_b_ms"] is None
        assert rec["sol_score"] is None
        assert rec["headroom_fraction"] is not None
        assert rec["card_check_state"] == "not_applicable"


@pytest.mark.skipif(not REAL_SESSION.exists(), reason="needs the full-01 run")
def test_a_harvested_packet_without_a_manifest_is_still_scoreable():
    """391 of full-01's 404 packets kept no `.packet.json`. The problem key is
    the directory name either way, and which source was used is recorded."""
    assert not (REAL_SESSION / "packet" / ".packet.json").exists()


def test_the_scoreboard_names_a_refusal_instead_of_calling_it_unknown():
    from solexbench_agents.aggregate import failure_kind

    assert failure_kind({"outcome": "refused_card_mismatch"}) == \
        "refused_card_mismatch"
    assert failure_kind({"outcome": "refused_packet_problem_mismatch"}) == \
        "refused_packet_problem_mismatch"


def test_a_packet_claiming_another_problem_is_refused(tmp_path):
    d = tmp_path / "L1__001_p"
    (d / "packet").mkdir(parents=True)
    (d / "session.json").write_text(json.dumps({"harness": "h",
                                                "produced_solution": True}))
    (d / "packet" / ".packet.json").write_text(
        json.dumps({"problem": "L1/002_other"}))
    out = ss.score_one(d, tmp_path / "prob", None, t_sol={}, t_b={},
                       part="MI355X", timeout=1, gpu=0)
    assert out["outcome"] == "refused_packet_problem_mismatch"


def _score_file(dirpath: Path, key: str, *, card: dict | None,
                basis: str = "sol_headroom") -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    doc = {
        "problem": key, "harness": "claude-code", "outcome": "evaluated",
        "card_check": ({"ok": True, "state": "not_applicable",
                        "actual": card} if card else None),
        "records": [{"workload_uuid": "w0", "correct": True, "t_k_ms": 0.5,
                     "t_ref_ms": 0.9, "t_sol_ms": 0.1, "t_b_ms": None,
                     "score_basis": basis, "sol_score": None,
                     "headroom_fraction": 0.5}],
    }
    p = dirpath / f"{key}.json"
    p.write_text(json.dumps(doc))
    return p


def _manifest(path: Path, keys: list[str], *, t_b: float | None) -> Path:
    problems = {}
    for k in keys:
        w = {"t_sol_ms": 0.1}
        if t_b is not None:
            w["t_b_ms"] = t_b
        problems[k] = {"workloads": {"w0": w}}
    path.write_text(json.dumps({
        "manifest_version": "MI355X-v1", "part": "MI355X",
        # No single authoritative_gpu/f_lock: this manifest came from the
        # 8-way card-pinned pass, so the per-problem check applies.
        "_provenance": {"part": "MI355X", "f_lock_mhz": None},
        "problems": problems}))
    return path


def _backfill(tmp_path, monkeypatch, *, extra: list[str] = ()) -> tuple[dict, str]:
    monkeypatch.setattr(bf, "ROOT", tmp_path)
    monkeypatch.setattr(bf, "stamp", lambda *a, **k: _fake_prov())
    monkeypatch.setattr(sys, "argv", [
        "backfill_scores.py", "--run-id", "t1",
        "--manifest", str(tmp_path / "manifest.json"), *extra])
    assert bf.main() == 0
    scores = tmp_path / "artifacts" / "10" / "scores" / "t1"
    return (json.loads((scores / "claude-code" / "L1__001_p.json").read_text()),
            (scores / "summary.json"))


def test_the_ladder_raises_sol_headroom_to_sol_score_v1_when_the_card_matches(
        tmp_path, monkeypatch):
    scores = tmp_path / "artifacts" / "10" / "scores" / "t1"
    _score_file(scores / "claude-code", "L1__001_p", card=CARD_A)
    (scores / "summary.json").write_text(json.dumps({"run_id": "t1"}))
    tb = tmp_path / "authoritative"
    _anchor(tb, "L1__001_p", card=CARD_A)
    _manifest(tmp_path / "manifest.json", ["L1__001_p"], t_b=1.0)

    doc, summary_path = _backfill(tmp_path, monkeypatch,
                                  extra=["--tb-artifacts", str(tb)])
    rec = doc["records"][0]
    assert rec["score_basis"] == "sol_score_v1"
    assert rec["score_basis_history"] == ["sol_headroom"]
    assert rec["t_b_ms"] == 1.0
    assert rec["sol_score"] == pytest.approx(1 / (1 + 0.4 / 0.9))
    assert "t_b_refused" not in rec
    # And no GPU was involved: the timings are the ones already on disk.
    assert rec["t_k_ms"] == 0.5 and rec["t_ref_ms"] == 0.9
    summary = json.loads(summary_path.read_text())
    assert summary["backfill_card_enforcement"]["matched"] == 1
    assert summary["backfill_card_enforcement"]["refused"] == 0


@pytest.mark.parametrize("scoring_card,anchor,why", [
    (CARD_A, CARD_B, "the anchor is on another card"),
    (None, CARD_A, "the score file never recorded which card it used"),
])
def test_the_ladder_refuses_the_raise_when_the_card_does_not_check_out(
        tmp_path, monkeypatch, scoring_card, anchor, why):
    scores = tmp_path / "artifacts" / "10" / "scores" / "t1"
    _score_file(scores / "claude-code", "L1__001_p", card=scoring_card)
    (scores / "summary.json").write_text(json.dumps({"run_id": "t1"}))
    tb = tmp_path / "authoritative"
    _anchor(tb, "L1__001_p", card=anchor)
    _manifest(tmp_path / "manifest.json", ["L1__001_p"], t_b=1.0)

    doc, summary_path = _backfill(tmp_path, monkeypatch,
                                  extra=["--tb-artifacts", str(tb)])
    rec = doc["records"][0]
    assert rec["score_basis"] == "sol_headroom", why
    assert rec["t_b_ms"] is None
    assert rec["sol_score"] is None
    assert rec["t_b_refused"]
    summary = json.loads(summary_path.read_text())
    assert summary["backfill_card_enforcement"]["refused"] == 1
    assert "L1__001_p" in summary["backfill_card_enforcement"]["refusals"]


def test_a_single_card_manifest_keeps_the_pre_4_4_path(tmp_path, monkeypatch):
    """MI350X was pinned to one GPU and `_assert_comparable` is the proof for
    it. That path must not start refusing on anchors that predate card
    identity."""
    scores = tmp_path / "artifacts" / "10" / "scores" / "t1"
    _score_file(scores / "claude-code", "L1__001_p", card=None)
    (scores / "summary.json").write_text(json.dumps({"run_id": "t1"}))
    path = tmp_path / "manifest.json"
    _manifest(path, ["L1__001_p"], t_b=1.0)
    doc = json.loads(path.read_text())
    doc["_provenance"] = {"part": "MI350X", "f_lock_mhz": 1300,
                          "authoritative_gpu": 0}
    path.write_text(json.dumps(doc))

    out, summary_path = _backfill(tmp_path, monkeypatch)
    assert out["records"][0]["score_basis"] == "sol_score_v1"
    summary = json.loads(summary_path.read_text())
    assert summary["backfill_card_enforcement"]["mode"] == "single-card-manifest"


# ----------------------------------------------------------------- the part

def test_part_resolution_does_not_fall_back_to_the_mi350x_tree():
    from verify_artifacts import ArtifactTree

    tree = ArtifactTree("MI355X")
    assert tree.path("03", "t_sol.json").name == "t_sol.json"
    assert tree.path("03", "t_sol.json").parent.name == "03-MI355X"
    assert tree.dir("06").name == "06-MI355X"
    assert tree.dir("05").name == "05-MI355X"
    # And the default part still resolves exactly where the release tree is.
    assert ArtifactTree("MI350X").dir("03").name == "03"


def test_a_foreign_part_bound_is_refused_rather_than_silently_used(tmp_path):
    """`artifacts/03/t_sol.json` is MI350X at F_LOCK 1300. Applied to an MI355X
    measurement it rescales every score by the clock ratio -- plausibly, and in
    the direction that flatters the kernel."""
    p = tmp_path / "t_sol.json"
    p.write_text(json.dumps({"part": "MI350X", "problems": {"L1__001_p": {}}}))
    bounds, meta = ss.load_bounds(p, "MI355X")
    assert bounds == {}
    assert "MI350X" in meta["rejected"] and "MI355X" in meta["rejected"]


def test_the_part_the_node_reports_wins_over_a_wrong_flag(
        tmp_path, monkeypatch, scoring_main):
    _run_tree(tmp_path, ["L1__001_p"])
    _bounds_file(tmp_path / "t_sol.json", ["L1__001_p"], "t_sol_ms", 0.1)
    monkeypatch.setattr(sys, "argv", [
        "score_solutions.py", "--run-id", "t1", "--gpu", "0",
        "--part", "MI350X",
        "--t-sol", str(tmp_path / "t_sol.json"), "--workloads-root", "none"])
    with pytest.raises(SystemExit) as exc:
        ss.main()
    assert "MI350X" in str(exc.value) and "MI355X" in str(exc.value)


def test_a_foreign_part_manifest_is_refused(tmp_path, monkeypatch, scoring_main):
    _run_tree(tmp_path, ["L1__001_p"])
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"manifest_version": "v1", "part": "MI350X",
                                    "problems": {}}))
    monkeypatch.setattr(sys, "argv", [
        "score_solutions.py", "--run-id", "t1", "--gpu", "0",
        "--manifest", str(manifest), "--workloads-root", "none"])
    with pytest.raises(SystemExit) as exc:
        ss.main()
    assert "REFUSING" in str(exc.value)
