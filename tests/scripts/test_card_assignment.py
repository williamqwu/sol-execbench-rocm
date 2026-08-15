# SPDX-License-Identifier: Apache-2.0
"""One card assignment, shared by the passes that pin a card.

`STATE.md` 4.4 requires a problem's `T_b` and its own `T_k` on ONE physical
card. `scripts/card_assignment.py` is the single owner of "which card does
problem X belong to". The properties tested here are the ones whose absence
made the old `plan[i::8]` assignment agree with the enforcement only by luck:

  * a card is named by ``(hostname, bdf, uuid)`` -- the same three fields
    `score_solutions.CARD_KEYS` compares -- so slot 5 on g05 and slot 5 on g45
    are different cards, which is the shape of 17 of the 45 measured mismatches;
  * the map covers the whole dataset and is materialised, so a problem gaining
    or losing a winning variant moves that problem and nothing else;
  * a rebuild after a card dies moves the problems on the dead card and nothing
    else (``carry``);
  * a work list is derived from the card a process ACTUALLY probed, so a
    mis-set `HIP_VISIBLE_DEVICES` raises instead of running the wrong list;
  * an assignment is recoverable, and tamper-evident, from the artifact alone;
  * the shipped MI355X assignment agrees with the anchor tree manifest v3
    publishes, so adopting it moves no published `T_b`.

CPU-only: nothing here touches a GPU, and `fleet_from_probe` is exercised with
an injected prober.
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


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ca = _load("_card_assignment", ROOT / "scripts" / "card_assignment.py")
ss = _load("_score_solutions", ROOT / "scripts" / "score_solutions.py")

SCRIPT = ROOT / "scripts" / "card_assignment.py"
SHIPPED = ROOT / "artifacts" / "06-MI355X" / "card-assignment.json"

BDFS = ("0000:75:00.0", "0000:05:00.0", "0000:65:00.0", "0000:15:00.0",
        "0000:f5:00.0", "0000:85:00.0", "0000:e5:00.0", "0000:95:00.0")


def card(host: str, slot: int, *, uuid: str | None = None) -> dict:
    """A card identity in the shape `authoritative_tb.card_identity` returns.

    The BDF is a function of the slot only -- as it is on this fleet, where all
    three nodes have the identical BDF->torch-index map -- so any test that
    passes on `bdf` alone would confuse two nodes.
    """
    return {"identified": True, "hostname": host, "bdf": BDFS[slot % 8],
            "uuid": uuid or f"{host}-{slot}-uuid",
            "drm_card": f"/sys/class/drm/card{slot}",
            "hip_visible_devices": str(slot),
            "device_name": "AMD Instinct MI355X"}


def fleet(hosts_and_counts) -> list[dict]:
    return [card(host, slot)
            for host, n in hosts_and_counts for slot in range(n)]


def keys(n: int, prefix: str = "L1") -> list[str]:
    return [f"{prefix}__{i:03d}_problem" for i in range(n)]


FLEET24 = fleet([("g46", 8), ("g45", 8), ("g05", 8)])


# --- identity ---------------------------------------------------------------

def test_card_keys_equal_the_enforcement_s_keys():
    """The partition and the enforcement must compare the same three fields.

    If these ever diverge, a partition can put a problem on a card the card
    check then refuses -- which is the "speaking different languages" defect
    this module exists to remove. A comment cannot hold that; this can.
    """
    assert ca.CARD_KEYS == ss.CARD_KEYS == ("hostname", "bdf", "uuid")


def test_token_round_trips_and_refuses_a_partial_identity():
    token = ca.card_token(card("g46", 3))
    assert token == "g46|0000:15:00.0|g46-3-uuid"
    assert ca.parse_token(token) == {"hostname": "g46", "bdf": "0000:15:00.0",
                                     "uuid": "g46-3-uuid"}
    assert ca.card_token(token) == token

    with pytest.raises(ca.UnidentifiedCard):
        ca.card_token({"identified": False, "error": "amdsmi died"})
    with pytest.raises(ca.UnidentifiedCard):
        ca.card_token({"identified": True, "hostname": "g46",
                       "bdf": "0000:15:00.0"})          # no uuid
    with pytest.raises(ca.UnidentifiedCard):
        ca.card_token("g46|0000:15:00.0")               # not a token
    with pytest.raises(ca.UnidentifiedCard):
        ca.card_token({**card("g46", 0), "hostname": "g4|6"})


def test_same_bdf_on_two_nodes_is_two_cards():
    a, b = card("g45", 5), card("g05", 5)
    assert a["bdf"] == b["bdf"]
    assert ca.card_token(a) != ca.card_token(b)


def test_a_duplicate_card_in_the_fleet_is_refused():
    with pytest.raises(ca.CardAssignmentError):
        ca.normalise_fleet([card("g46", 0), card("g46", 0)])
    with pytest.raises(ca.CardAssignmentError):
        ca.normalise_fleet([])


# --- building ---------------------------------------------------------------

def test_build_covers_every_key_exactly_once_and_balances():
    a = ca.build(keys(235), FLEET24, assignment_id="t", part="MI355X")
    assert len(a.problems) == 235
    assert set(a.problems) == set(keys(235))
    counts = a.counts()
    assert sum(counts.values()) == 235
    assert max(counts.values()) - min(counts.values()) <= 1
    # the union over the fleet is exactly the key set, nothing duplicated
    union: list[str] = []
    for row in a.fleet:
        union += a.keys_for_card(row)
    assert sorted(union) == sorted(keys(235))
    assert len(union) == len(set(union))


def test_build_is_deterministic_and_id_dependent():
    one = ca.build(keys(60), FLEET24, assignment_id="t")
    two = ca.build(list(reversed(keys(60))), FLEET24, assignment_id="t")
    assert one.problems == two.problems
    other = ca.build(keys(60), FLEET24, assignment_id="other")
    assert other.problems != one.problems


def test_three_nodes_with_unequal_card_counts():
    """Nothing computes `i % 8`; the fleet is enumerated, not counted."""
    mixed = fleet([("g46", 8), ("g45", 8), ("g05", 5)])
    assert len(mixed) == 21
    a = ca.build(keys(235), mixed, assignment_id="t")
    counts = a.counts()
    assert len(counts) == 21
    assert all(v > 0 for v in counts.values())
    assert max(counts.values()) - min(counts.values()) <= 1
    # a fleet that is not a power of two, and one node with a single card
    tiny = ca.build(keys(17), fleet([("g46", 3), ("g45", 1)]),
                    assignment_id="t")
    assert sorted(tiny.counts().values()) == [4, 4, 4, 5]


def test_a_problem_leaving_the_list_moves_only_itself():
    """The L1__094 failure mode: one problem's plan membership moved 100 of 217.

    Keyed on the dataset rather than the plan, and unbalanced, a key's card is a
    function of its own name and the fleet alone.
    """
    full = ca.build(keys(235), FLEET24, assignment_id="t", balance=False)
    dropped = ca.build([k for k in keys(235) if k != "L1__094_problem"],
                       FLEET24, assignment_id="t", balance=False)
    assert dropped.problems == {k: v for k, v in full.problems.items()
                                if k != "L1__094_problem"}


def test_balanced_rebuild_with_carry_moves_nothing():
    """Balancing is what a formula cannot do stably -- so carry the map."""
    before = ca.build(keys(235), FLEET24, assignment_id="t")
    after = ca.build(keys(235) + ["L2__001_new_problem"], FLEET24,
                     assignment_id="t", carry=before)
    assert {k: v for k, v in after.problems.items()
            if k in before.problems} == before.problems
    assert after.seed_source["L2__001_new_problem"] == "hash"
    assert set(after.seed_source.values()) == {"carried", "hash"}


def test_a_dead_card_moves_only_its_own_problems():
    before = ca.build(keys(235), FLEET24, assignment_id="t")
    dead = before.fleet[7]["card"]
    orphans = set(before.keys_for_card(dead))
    assert orphans
    survivors = [row for row in FLEET24 if ca.card_token(row) != dead]
    after = ca.build(keys(235), survivors, assignment_id="t", carry=before)
    moved = {k for k, v in after.problems.items() if before.problems[k] != v}
    assert moved == orphans
    # and each says why it moved, rather than looking like a fresh placement
    assert all(after.seed_source[k] == "hash-after-card-gone" for k in orphans)
    assert dead not in set(after.problems.values())
    # the orphans spread over the survivors instead of landing on one card
    counts = after.counts()
    assert max(counts.values()) - min(counts.values()) <= 1


def test_build_refuses_an_unknown_rule_and_an_empty_key_set():
    with pytest.raises(ca.CardAssignmentError):
        ca.build(keys(4), FLEET24, assignment_id="t", rule="round-robin")
    with pytest.raises(ca.CardAssignmentError):
        ca.build([], FLEET24, assignment_id="t")


# --- lookups: the mis-set HIP_VISIBLE_DEVICES case ---------------------------

def test_keys_for_card_refuses_a_card_that_is_not_ours():
    a = ca.build(keys(40), fleet([("g46", 8)]), assignment_id="t")
    with pytest.raises(ca.UnknownCard):
        a.keys_for_card(card("g45", 3))          # right BDF, wrong node
    with pytest.raises(ca.UnidentifiedCard):
        a.keys_for_card({"identified": False, "error": "no amdsmi"})
    with pytest.raises(ca.UnassignedProblem):
        a.card_for("L1__999_not_in_the_dataset")


def test_work_lists_are_disjoint_across_nodes_that_share_a_bdf():
    a = ca.build(keys(235), FLEET24, assignment_id="t")
    lists = {h: set(a.keys_for_card(card(h, 5))) for h in ("g46", "g45", "g05")}
    assert not (lists["g46"] & lists["g45"])
    assert not (lists["g45"] & lists["g05"])
    assert not (lists["g46"] & lists["g05"])
    assert all(lists.values())


def test_resumption_is_a_set_lookup_not_a_stride():
    """A sweep dies half way; the pending list must not depend on progress."""
    a = ca.build(keys(235), FLEET24, assignment_id="t")
    mine = a.keys_for_card(card("g45", 2))
    done = set(mine[:3])
    pending = [k for k in a.keys_for_card(card("g45", 2)) if k not in done]
    assert pending == mine[3:]
    # and a rebuild after three more problems appear leaves pending untouched
    later = ca.build(keys(235) + keys(3, "Quant"), FLEET24,
                     assignment_id="t", carry=a)
    assert [k for k in later.keys_for_card(card("g45", 2))
            if k in a.problems and k not in done] == pending


# --- the artifact -----------------------------------------------------------

def test_recovered_from_the_artifact_alone(tmp_path):
    a = ca.build(keys(50), FLEET24, assignment_id="t", part="MI355X")
    out = ca.write(a, tmp_path / "card-assignment.json")
    doc = json.loads(out.read_text())
    assert doc["_provenance"]["task"] == "06-card-assignment"
    assert doc["schema"] == ca.SCHEMA

    back = ca.load(out)
    assert back.problems == a.problems
    assert back.digest == a.digest
    assert back.part == "MI355X"
    assert back.keys_for_card(card("g05", 4)) == a.keys_for_card(card("g05", 4))
    # the fleet order is part of the assignment, not a rendering choice
    assert [r["card"] for r in back.fleet] == [r["card"] for r in a.fleet]


def test_a_hand_edited_assignment_is_detected(tmp_path):
    a = ca.build(keys(50), FLEET24, assignment_id="t", part="MI355X")
    out = ca.write(a, tmp_path / "card-assignment.json")
    doc = json.loads(out.read_text())
    key = sorted(doc["problems"])[0]
    others = [t for t in doc["problems"].values() if t != doc["problems"][key]]
    doc["problems"][key] = others[0]
    (tmp_path / "tampered.json").write_text(json.dumps(doc))
    with pytest.raises(ca.AssignmentIntegrityError):
        ca.load(tmp_path / "tampered.json")

    doc["schema"] = "card-assignment/99"
    (tmp_path / "future.json").write_text(json.dumps(doc))
    with pytest.raises(ca.AssignmentIntegrityError):
        ca.load(tmp_path / "future.json")

    with pytest.raises(ca.CardAssignmentError):
        ca.load(tmp_path / "does-not-exist.json")


def test_an_assignment_pointing_off_its_own_fleet_is_refused():
    with pytest.raises(ca.UnknownCard):
        ca.Assignment("t", fleet([("g46", 2)]),
                      {"L1__001_problem": ca.card_token(card("g45", 0))})


def test_default_path_keeps_mi350x_unsuffixed(tmp_path):
    assert ca.default_path("MI355X", tmp_path).parent.name == "06-MI355X"
    assert ca.default_path("MI350X", tmp_path).parent.name == "06"
    assert ca.default_path(None, tmp_path).parent.name == "06"


# --- reading the fleet and the corpus ---------------------------------------

def _tree(base: Path, cards: dict[str, dict], *, extra: dict | None = None):
    base.mkdir(parents=True, exist_ok=True)
    for key, identity in cards.items():
        (base / f"{key}.json").write_text(json.dumps(
            {"problem": key, "card_identity": identity, "ok": True}))
    for name, doc in (extra or {}).items():
        (base / f"{name}.json").write_text(json.dumps(doc))
    return base


def test_fleet_from_artifacts_reads_identities_without_a_gpu(tmp_path):
    _tree(tmp_path / "t", {"L1__001_problem": card("g46", 0),
                           "L1__002_problem": card("g45", 0),
                           "L1__003_problem": card("g46", 0),
                           "L1__004_problem": {"identified": False,
                                               "error": "amdsmi died"}},
          extra={"_merge-report": {"note": "not a problem"}})
    got = ca.fleet_from_artifacts([tmp_path / "t"])
    assert [r["hostname"] for r in got] == ["g45", "g46"]      # sorted, unique
    assert all("uuid" in r and "card" in r for r in got)
    assert got[0]["drm_card"] == "/sys/class/drm/card0"


def test_fleet_from_probe_uses_the_injected_prober():
    seen = []

    def prober(gpu):
        seen.append(gpu)
        return card("g46", int(gpu))

    got = ca.fleet_from_probe([0, 1, 2], prober=prober)
    assert seen == ["0", "1", "2"]
    assert [r["bdf"] for r in got] == list(BDFS[:3])

    with pytest.raises(ca.UnidentifiedCard):
        ca.fleet_from_probe([0], prober=lambda g: {"identified": False,
                                                   "error": "no amdsmi"})


def test_observed_placements_prefer_the_manifest_s_own_anchor_tree(tmp_path):
    """Rule 1 first is the neutrality property: no published `T_b` has to move."""
    merged = _tree(tmp_path / "merged", {"L1__001_problem": card("g46", 1),
                                         "L1__002_problem": card("g46", 2)})
    other = _tree(tmp_path / "g05", {"L1__001_problem": card("g05", 1),
                                     "L1__003_problem": card("g05", 3),
                                     "L1__005_problem": card("g05", 5)})
    scores = tmp_path / "scores" / "codex"
    scores.mkdir(parents=True)
    for key, cards in (("L1__001_problem", card("g45", 1)),
                       ("L1__002_problem", card("g45", 2)),
                       ("L1__003_problem", card("g45", 3)),
                       ("L1__004_problem", card("g45", 4))):
        (scores / f"{key}.json").write_text(json.dumps(
            {"problem": key, "card_check": {"actual": cards}}))
    # a second harness that disagrees about where L1__004 ran
    two = tmp_path / "scores" / "claude-code"
    two.mkdir()
    (two / "L1__004_problem.json").write_text(json.dumps(
        {"problem": "L1__004_problem", "card_check": {"actual": card("g05", 4)}}))

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"sources": {"t_b": str(merged)}}))

    placed, labels = ca.observed_placements(
        manifest=manifest, score_dirs=[tmp_path / "scores"],
        tb_trees=[other], root=tmp_path)

    assert labels["L1__001_problem"] == "manifest_anchor"
    assert placed["L1__001_problem"] == ca.card_token(card("g46", 1))
    assert labels["L1__003_problem"] == "t_k_card"          # no manifest anchor
    assert labels["L1__005_problem"] == "tb_tree:g05"       # no anchor, no T_k
    assert "L1__004_problem" not in placed                  # harnesses disagree

    manifest.write_text(json.dumps({"sources": {}}))
    with pytest.raises(ca.CardAssignmentError):
        ca.observed_placements(manifest=manifest, root=tmp_path)


def test_observed_seeds_are_never_evicted_by_balancing():
    obs = {k: ca.card_token(card("g46", 0)) for k in keys(30)}
    a = ca.build(keys(235), FLEET24, assignment_id="t", rule="observed",
                 observed=obs, observed_source={k: "manifest_anchor"
                                                for k in obs})
    assert a.counts()[ca.card_token(card("g46", 0))] == 30
    assert all(a.problems[k] == obs[k] for k in obs)
    assert a.seed_source[keys(30)[0]] == "manifest_anchor"
    # a card pinned above the water line takes none of the remainder, and the
    # remainder still spreads evenly over the cards that are below it
    rest = [v for k, v in a.counts().items()
            if k != ca.card_token(card("g46", 0))]
    assert sum(rest) == 205
    assert max(rest) - min(rest) <= 1


def test_an_observed_card_outside_the_fleet_is_named_not_dropped():
    a = ca.build(keys(4), fleet([("g46", 2)]), assignment_id="t",
                 rule="observed",
                 observed={"L1__000_problem": ca.card_token(card("g05", 7))})
    assert a.seed_source["L1__000_problem"] == "hash-observed-off-fleet"
    assert a.problems["L1__000_problem"] in a.cards


def test_observed_is_ignored_under_the_hash_rule():
    obs = {k: ca.card_token(card("g46", 0)) for k in keys(20)}
    a = ca.build(keys(20), FLEET24, assignment_id="t", rule="hash",
                 observed=obs)
    assert set(a.seed_source.values()) == {"hash"}


# --- auditing a tree --------------------------------------------------------

def test_verify_tree_names_every_way_an_artifact_can_be_off(tmp_path):
    a = ca.build(keys(6), fleet([("g46", 2), ("g45", 2)]), assignment_id="t",
                 rule="observed",
                 observed={k: ca.card_token(card("g46", 0)) for k in keys(6)})
    tree = _tree(tmp_path / "tb", {
        "L1__000_problem": card("g46", 0),                  # matched
        "L1__001_problem": card("g45", 0),                  # off: same bdf!
        "L1__002_problem": {"identified": False, "error": "x"},
        "L1__099_problem": card("g46", 0),                  # not in the map
    }, extra={"_merge-report": {"note": "book-keeping"}})
    (tree / "L1__003_problem.json").write_text(json.dumps({"problem": "x"}))

    report = ca.verify_tree(a, tree)
    assert report["matched"] == 1
    assert report["off_assignment"] == [
        {"problem": "L1__001_problem",
         "assigned": ca.card_token(card("g46", 0)),
         "measured": ca.card_token(card("g45", 0))}]
    assert sorted(report["unidentified"]) == ["L1__002_problem",
                                              "L1__003_problem"]
    assert report["unassigned"] == ["L1__099_problem"]
    assert report["digest"] == a.digest
    with pytest.raises(ca.CardAssignmentError):
        ca.verify_tree(a, tmp_path / "nope")


# --- the dataset ------------------------------------------------------------

def test_dataset_keys_are_all_235_problems():
    got, source = ca.dataset_problem_keys(ROOT)
    assert len(got) == ca.EXPECTED_TOTAL == 235
    assert ca.census_of(got) == ca.EXPECTED_CENSUS
    assert source                                   # says where it came from
    assert got == sorted(got)


# --- the shipped MI355X assignment ------------------------------------------

@pytest.mark.skipif(not SHIPPED.exists(), reason="no MI355X assignment tracked")
def test_shipped_mi355x_assignment_is_the_whole_dataset_on_the_real_fleet():
    a = ca.load(SHIPPED)
    assert a.part == "MI355X"
    assert len(a.problems) == 235
    assert ca.census_of(a.problems) == ca.EXPECTED_CENSUS
    assert len(a.fleet) == 24
    assert sorted({r["hostname"] for r in a.fleet}) == [
        "mia1-p02-g05", "mia1-p02-g45", "mia1-p02-g46"]
    # 8 cards per node, and the same BDF appears on all three -- so an
    # assignment keyed on the BDF alone would be ambiguous three ways
    per_bdf: dict[str, set[str]] = {}
    for row in a.fleet:
        per_bdf.setdefault(row["bdf"], set()).add(row["hostname"])
    assert len(per_bdf) == 8
    assert all(len(v) == 3 for v in per_bdf.values())
    assert len({r["uuid"] for r in a.fleet}) == 24

    union: list[str] = []
    for row in a.fleet:
        union += a.keys_for_card(row)
    assert sorted(union) == sorted(a.problems)


@pytest.mark.skipif(not SHIPPED.exists(), reason="no MI355X assignment tracked")
def test_shipped_assignment_covers_exactly_the_tracked_dataset_census():
    meta = json.loads((ROOT / "reference" / "dataset-meta.json").read_text())
    assert set(ca.load(SHIPPED).problems) == set(meta["problems"])


@pytest.mark.skipif(
    not SHIPPED.exists()
    or not (ROOT / "artifacts/06-MI355X/authoritative-merged").is_dir(),
    reason="MI355X T_b tree not present")
def test_shipped_assignment_moves_no_published_anchor():
    """Every identified anchor in the tree manifest v3 publishes is on its card.

    This is the property that makes adopting the assignment safe: the published
    `T_b` for a problem is already the one measured on the card the assignment
    names, so no re-selection among replicates is implied. (Re-pointing
    `merge_authoritative_tb` at an assignment that did NOT have this property
    would raise published `T_b`, and `S` with it -- the undetectable direction.)
    """
    report = ca.verify_tree(ca.load(SHIPPED),
                            ROOT / "artifacts/06-MI355X/authoritative-merged")
    assert report["off_assignment"] == []
    # >= rather than == : the tree may gain a problem, and that must show up as
    # `unassigned` (rebuild the assignment), never as a silent pass.
    assert report["matched"] >= 217
    assert report["unassigned"] == []
    assert len(report["unidentified"]) <= 3      # artifacts with no card block


# --- CLI --------------------------------------------------------------------

def _cli(*args, expect: int = 0):
    r = subprocess.run([sys.executable, str(SCRIPT), *args],
                       capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == expect, r.stderr[-2000:]
    return r


@pytest.mark.skipif(not SHIPPED.exists(), reason="no MI355X assignment tracked")
def test_cli_show_and_keys():
    out = _cli("show", "--file", str(SHIPPED)).stdout
    assert "mi355x-" in out and "24" in out

    listed = _cli("keys", "--file", str(SHIPPED),
                  "--host", "mia1-p02-g45", "--bdf", "0000:75:00.0").stdout
    got = [line for line in listed.splitlines() if line.strip()]
    a = ca.load(SHIPPED)
    want = a.keys_for_card(next(r for r in a.fleet
                                if r["hostname"] == "mia1-p02-g45"
                                and r["bdf"] == "0000:75:00.0"))
    assert got == want

    # a card named ambiguously (three nodes share every BDF) is refused
    err = _cli("keys", "--file", str(SHIPPED), "--host", "mia1",
               "--bdf", "0000:75:00.0", expect=2).stderr
    assert "matches 3 cards" in err


def test_cli_build_refuses_a_partial_census(tmp_path):
    """An assignment over fewer than 235 problems is an omission (CLAUDE.md 0)."""
    empty = tmp_path / "no-cards"
    empty.mkdir()
    err = _cli("build", "--part", "MI355X", "--assignment-id", "x",
               "--fleet-from", str(empty), "--out", str(tmp_path / "a.json"),
               expect=2).stderr
    assert "no identified card" in err or "REFUSED" in err
    assert not (tmp_path / "a.json").exists()
