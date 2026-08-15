#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One materialised card assignment, shared by every pass that pins a card.

`STATE.md` §4.4 requires a problem's `T_b` and its own `T_k` to be measured on
ONE physical card. Two passes have to agree about which card that is: the
authoritative T_b pass (`scripts/authoritative_tb.py`) and the scoring pass
(`scripts/score_solutions.py`). Today they do not agree by construction, they
agree by luck, and the three ways that goes wrong are what this module removes.

WHAT WAS WRONG
--------------
1. **The assignment was not a function of the problem name.** Both passes stride
   `plan[i::N]` over a *sorted plan whose membership is data-dependent* —
   `authoritative_tb.build_plan()` drops every problem with no all-passing
   variant. A problem entering or leaving the plan shifts every problem after it
   onto another card. Measured on the two candidate trees on disk: one problem
   (`L1__094_time_decay_exponential_stabilization`) gaining a winning variant
   moves **100 of 217** shared problems to a different card.

   Status of that number, because it was refuted once and the correction
   matters: recomputed against the plan the sweeps *actually ran under*
   (`candidates-frozen-2047`), the recorded `shard.index` reproduces with **zero**
   disagreements across 649 artifacts. So this is **latent fragility held in
   check by an untracked frozen copy**, not observed corpus damage. It is still
   worth removing — an untracked directory is not a guarantee — but it is not
   evidence that any published number is wrong.

2. **A shard index names a card SLOT, not a card.** All three nodes of this
   fleet have the identical BDF->torch-index map, so "slot 5" is g46, g45 *and*
   g05. 17 of the 45 measured anchor/T_k mismatches are exactly that shape:
   same BDF, different node. An integer in `[0, N)` cannot express "node g05,
   card 5", so the partition and the enforcement (`score_solutions.CARD_KEYS =
   (hostname, bdf, uuid)`) were speaking different languages.

3. **`i % 8` cannot describe a fleet.** Three nodes with unequal card counts, or
   one card down for maintenance, is not expressible as a modulus.

WHAT THIS MODULE IS
-------------------
A map from problem key to a **physical card identity**, covering **all 235
dataset problems** — not the candidates plan — written to
`artifacts/06-<PART>/card-assignment.json` and consumed from there.

  * **Materialised, not computed.** The seeding rule is recorded under `rule`
    for audit; `problems` is the authority. A formula is a pure function only
    while its inputs are fixed, and the fleet is not fixed: a card dying must
    move the problems on that card and nothing else. `build(..., carry=...)`
    is how a rebuild keeps every assignment it can.
  * **Keyed over the whole dataset**, so a problem gaining or losing a winning
    variant changes only itself.
  * **Identity, not slot.** Every fleet entry carries `hostname`, `bdf` and
    `uuid` — exactly `score_solutions.CARD_KEYS`, so the partition and the
    enforcement compare the same three fields. `tests/scripts/
    test_card_assignment.py` pins that equality.
  * **`keys_for_card(identity)`** takes the identity of the card a process
    ACTUALLY probed. A mis-set `HIP_VISIBLE_DEVICES` then raises `UnknownCard`
    instead of quietly running the wrong list.
  * **Recoverable from the artifact alone.** `load()` needs no dataset, no
    fleet probe and no GPU, and the document carries a `digest` over
    (`assignment_id`, `part`, `fleet`, `problems`) so a hand-edit is detected.

WHAT THIS MODULE IS NOT
-----------------------
It does not move a published number and it must not be made to. In particular
**it is not wired into `merge_authoritative_tb.py`**, and pointing that script's
tiebreak at the assigned card is *not* a neutral change: today's tiebreak is
"lowest median T_b", the published anchor is the lowest-median replicate for
**161 of 162** contested problems, and `S` is monotone increasing in `T_b`.
Re-pointing it raises published `T_b` across the 220 problems — median about
+0.4% per workload, +0.9% per problem, tail to +50% — which raises every score
on the board, in the undetectable direction, for every future run. That change
needs a maintainer's sign-off; see the fix report for the measured direction and
magnitude. This module deliberately provides no `merge` entry point.

Likewise nothing here re-derives a bound: `T_SOL` comes off `device="meta"` and
knows nothing about cards (CLAUDE.md §6).

WHICH SEEDING RULE
------------------
`rule="hash"` is blind to every measurement and exactly balanced: the right
choice for a fresh campaign where both sides will be measured under the
assignment.

`rule="observed"` (`observed_placements()`) seeds from where the corpus already
sits, in this priority order, and the ORDER IS THE ARGUMENT:

  1. the card that measured the anchor the manifest actually publishes
     (`manifest["sources"]["t_b"]`), so **no published `T_b` has to move**;
  2. else the card a `T_k` was measured on, when every score file agrees;
  3. else any other T_b tree, in the order given.

Anything not placed by those falls to the hash rule. Seeding from the corpus is
a data-dependent choice and is named as one: it looks at *where* a measurement
happened, never at its value, and rule 1 is first precisely so that adopting the
assignment cannot become a re-selection of anchors.

CLI
---
    # build the MI355X assignment from the fleet the artifacts name (no GPU)
    python scripts/card_assignment.py build --part MI355X \
        --assignment-id mi355x-2026-08-15 \
        --fleet-from artifacts/06-MI355X \
        --rule observed --manifest artifacts/09-MI355X/manifest-v3.json \
        --scores artifacts/10/scores/full-01 \
        --tb-tree artifacts/06-MI355X/authoritative \
        --out artifacts/06-MI355X/card-assignment.json

    python scripts/card_assignment.py show   --part MI355X
    python scripts/card_assignment.py keys   --part MI355X --host g45 --bdf 0000:75:00.0
    python scripts/card_assignment.py verify --part MI355X \
        --tree artifacts/06-MI355X/authoritative-merged

CPU-only. The single path that touches a card is `fleet_from_probe()`, which
runs `authoritative_tb.card_identity()` once per card; it is not on any path
used above, and it must not run while a timing run holds a card.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

SCHEMA = "card-assignment/1"

#: The fields that name a physical card, compared all-or-nothing. MUST equal
#: `score_solutions.CARD_KEYS` -- the partition and the enforcement have to
#: speak one language, and a test asserts the equality rather than a comment.
CARD_KEYS = ("hostname", "bdf", "uuid")

#: Kept alongside each fleet entry for triage. NOT part of a card's identity:
#: `drm_card` and `hip_visible_devices` are positions in orderings that change,
#: which is the whole reason `CARD_KEYS` is what it is.
CARD_EXTRAS = ("drm_card", "device_name", "hip_visible_devices",
               "pci_bus_id", "torch_index_in_child")

#: The dataset census, from CLAUDE.md 0. All 235 problems are in scope; a build
#: over fewer is an omission unless it says so out loud.
EXPECTED_CENSUS = {"L1": 94, "L2": 82, "Quant": 33, "FlashInfer-Bench": 26}
EXPECTED_TOTAL = sum(EXPECTED_CENSUS.values())   # 235


class CardAssignmentError(Exception):
    """Base for every refusal in this module. Nothing here guesses."""


class UnidentifiedCard(CardAssignmentError):
    """A card that could not name itself. Never treated as "probably the one"."""


class UnknownCard(CardAssignmentError):
    """A real card that is not in this assignment's fleet."""


class UnassignedProblem(CardAssignmentError):
    """A problem key the assignment does not cover."""


class AssignmentIntegrityError(CardAssignmentError):
    """The document does not match its own digest, or contradicts itself."""


# --- card identity ----------------------------------------------------------

def card_token(identity: Mapping | str) -> str:
    """``"hostname|bdf|uuid"`` — the assignment's stable name for a card.

    A string rather than an index into `fleet`, so reordering or extending the
    fleet list cannot silently re-point an assignment, and so a token in a T_b
    artifact is readable without the fleet in hand.
    """
    if isinstance(identity, str):
        parse_token(identity)          # raises unless it really is a token
        return identity
    if not isinstance(identity, Mapping):
        raise UnidentifiedCard(
            f"card identity must be a mapping or a token, got "
            f"{type(identity).__name__}")
    if identity.get("identified") is False:
        raise UnidentifiedCard(
            "this card could not identify itself "
            f"({str(identity.get('error'))[:200]}); refusing to name it rather "
            "than assuming which card it is")
    missing = [k for k in CARD_KEYS if not identity.get(k)]
    if missing:
        raise UnidentifiedCard(
            f"card identity incomplete on {missing}; a partial identity is not "
            f"an identity (CARD_KEYS={list(CARD_KEYS)})")
    parts = [str(identity[k]) for k in CARD_KEYS]
    for value in parts:
        if "|" in value:
            raise UnidentifiedCard(
                f"card identity field {value!r} contains '|', which is the "
                "token separator")
    return "|".join(parts)


def parse_token(token: str) -> dict:
    """``"hostname|bdf|uuid"`` -> the three `CARD_KEYS` as a dict."""
    parts = str(token).split("|")
    if len(parts) != len(CARD_KEYS) or not all(parts):
        raise UnidentifiedCard(
            f"not a card token: {token!r} (expected "
            f"{'|'.join(CARD_KEYS)})")
    return dict(zip(CARD_KEYS, parts))


def fleet_entry(identity: Mapping) -> dict:
    """One fleet row: the identity, plus the non-identifying extras for triage."""
    token = card_token(identity)
    row = {k: str(identity[k]) for k in CARD_KEYS}
    for k in CARD_EXTRAS:
        if identity.get(k) is not None:
            row[k] = identity[k]
    row["card"] = token
    return row


def normalise_fleet(fleet: Iterable[Mapping]) -> list[dict]:
    """Validated fleet rows, in the order given. Duplicates are an error.

    Order is preserved rather than sorted: it is the tiebreak the balancer uses,
    so it is part of the recorded assignment and must not be reshuffled by a
    later reader.
    """
    rows: list[dict] = []
    seen: set[str] = set()
    for item in fleet:
        row = fleet_entry(item)
        if row["card"] in seen:
            raise CardAssignmentError(
                f"card {row['card']} appears twice in the fleet")
        seen.add(row["card"])
        rows.append(row)
    if not rows:
        raise CardAssignmentError("the fleet is empty; nothing to assign to")
    return rows


# --- the assignment ---------------------------------------------------------

def digest_of(assignment_id: str, part: str | None, fleet: Sequence[Mapping],
              problems: Mapping[str, str]) -> str:
    """A content digest over everything that decides where a problem runs.

    Extras (`drm_card`, `device_name`, ...) are excluded on purpose: a card that
    re-enumerates its DRM node is the same card, and re-stamping the digest for
    that would train a reader to ignore digest changes.
    """
    payload = {
        "assignment_id": assignment_id,
        "part": part,
        "fleet": [[str(row[k]) for k in CARD_KEYS] for row in fleet],
        "problems": {str(k): str(v) for k, v in sorted(problems.items())},
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "blake2s:" + hashlib.blake2s(blob.encode(), digest_size=16).hexdigest()


class Assignment:
    """Which physical card owns which problem. Materialised; the map is truth."""

    def __init__(self, assignment_id: str, fleet: Iterable[Mapping],
                 problems: Mapping[str, str], *, part: str | None = None,
                 rule: Mapping | None = None,
                 seed_source: Mapping[str, str] | None = None):
        self.assignment_id = str(assignment_id)
        self.part = part
        self.fleet = normalise_fleet(fleet)
        self.rule = dict(rule or {})
        self.seed_source = dict(seed_source or {})
        known = {row["card"] for row in self.fleet}
        assigned: dict[str, str] = {}
        for key, value in problems.items():
            token = value if isinstance(value, str) else card_token(value)
            parse_token(token)
            if token not in known:
                raise UnknownCard(
                    f"{key} is assigned to {token}, which is not in the fleet")
            assigned[str(key)] = token
        self.problems = dict(sorted(assigned.items()))

    # -- lookups ------------------------------------------------------------

    @property
    def cards(self) -> list[str]:
        return [row["card"] for row in self.fleet]

    def entry(self, identity: Mapping | str) -> dict:
        """The fleet row for a card, by identity or token. Raises if not ours."""
        token = card_token(identity)
        for row in self.fleet:
            if row["card"] == token:
                return row
        raise UnknownCard(
            f"card {token} is not in assignment {self.assignment_id!r} "
            f"({len(self.fleet)} cards: "
            f"{sorted({r['hostname'] for r in self.fleet})}). A card that is "
            "not in the fleet has no work list; check HIP_VISIBLE_DEVICES and "
            "the host, and rebuild the assignment if the fleet really changed.")

    def token_for(self, key: str) -> str:
        try:
            return self.problems[key]
        except KeyError:
            raise UnassignedProblem(
                f"{key!r} is not in assignment {self.assignment_id!r} "
                f"({len(self.problems)} problems). Rebuild the assignment over "
                "the dataset rather than assigning it here.") from None

    def card_for(self, key: str) -> dict:
        """The full fleet row that owns *key*."""
        return self.entry(self.token_for(key))

    def keys_for_card(self, identity: Mapping | str) -> list[str]:
        """The problems this CARD owns — sorted, and a set lookup, not a stride.

        Pass the identity the process actually probed
        (`authoritative_tb.card_identity(gpu)`), never a slot number: that is
        what turns a mis-set `HIP_VISIBLE_DEVICES` from a silent wrong-card run
        into an `UnknownCard`/`UnidentifiedCard` refusal.
        """
        row = self.entry(identity)
        return sorted(k for k, v in self.problems.items() if v == row["card"])

    def counts(self) -> dict[str, int]:
        """Work list size per card, for every card in the fleet including zeros."""
        out = {row["card"]: 0 for row in self.fleet}
        for token in self.problems.values():
            out[token] += 1
        return out

    # -- (de)serialisation --------------------------------------------------

    @property
    def digest(self) -> str:
        return digest_of(self.assignment_id, self.part, self.fleet,
                         self.problems)

    def to_doc(self) -> dict:
        return {
            "schema": SCHEMA,
            "assignment_id": self.assignment_id,
            "part": self.part,
            "digest": self.digest,
            "rule": self.rule,
            "fleet": self.fleet,
            "problems": self.problems,
            "seed_source": dict(sorted(self.seed_source.items())),
        }

    @classmethod
    def from_doc(cls, doc: Mapping) -> "Assignment":
        if doc.get("schema") not in (None, SCHEMA):
            raise AssignmentIntegrityError(
                f"unknown card-assignment schema {doc.get('schema')!r}; this "
                f"reader speaks {SCHEMA}")
        got = cls(
            doc.get("assignment_id", ""),
            doc.get("fleet") or [],
            doc.get("problems") or {},
            part=doc.get("part"),
            rule=doc.get("rule"),
            seed_source=doc.get("seed_source"),
        )
        stated = doc.get("digest")
        if stated and stated != got.digest:
            raise AssignmentIntegrityError(
                f"card-assignment digest mismatch: the document states "
                f"{stated} and its contents hash to {got.digest}. Something "
                "edited the map by hand; regenerate it rather than restamping "
                "the digest.")
        return got

    def __repr__(self) -> str:      # pragma: no cover - debugging aid
        return (f"<Assignment {self.assignment_id!r} part={self.part!r} "
                f"cards={len(self.fleet)} problems={len(self.problems)}>")


def load(path: str | Path) -> Assignment:
    """Recover an assignment from the artifact alone. No dataset, no GPU."""
    p = Path(path)
    if not p.exists():
        raise CardAssignmentError(f"no card assignment at {p}")
    try:
        doc = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AssignmentIntegrityError(f"{p} unreadable: {exc}") from None
    return Assignment.from_doc(doc)


def default_path(part: str | None, root: Path = ROOT) -> Path:
    """`artifacts/06-<PART>/card-assignment.json`, or `artifacts/06/` for MI350X.

    MI350X's release artifacts are unsuffixed and frozen; every other part is
    suffixed (CLAUDE.md 7).
    """
    suffix = "" if part in (None, "", "MI350X") else f"-{part}"
    return root / "artifacts" / f"06{suffix}" / "card-assignment.json"


def write(assignment: Assignment, path: str | Path, *,
          task: str = "06-card-assignment") -> Path:
    """Write with a provenance stamp (prime directive 5)."""
    import provenance

    return provenance.write_artifact(
        path, task, assignment.to_doc(),
        extra_provenance={"part": assignment.part,
                          "assignment_id": assignment.assignment_id})


# --- building ---------------------------------------------------------------

def _hash_index(assignment_id: str, key: str, modulus: int) -> int:
    h = hashlib.blake2s(f"{assignment_id}|{key}".encode(), digest_size=8)
    return int.from_bytes(h.digest(), "big") % modulus


def build(problem_keys: Iterable[str], fleet: Iterable[Mapping], *,
          assignment_id: str, part: str | None = None,
          rule: str = "hash",
          observed: Mapping[str, str] | None = None,
          observed_source: Mapping[str, str] | None = None,
          carry: Assignment | None = None,
          balance: bool = True) -> Assignment:
    """Materialise an assignment over *problem_keys*.

    Placement, in order, and every step is recorded per key in `seed_source`:

      * `carry` — a key already assigned by a previous assignment keeps its
        card, provided that card is still in the fleet. This is what makes a
        rebuild safe: losing a card moves the problems on that card and nothing
        else, where a recomputed formula would reshuffle everything.
      * `observed` — `rule="observed"` only; the corpus's own placement, see
        `observed_placements()`.
      * the hash seed `blake2s(assignment_id|key) % len(fleet)`, then, when
        *balance*, moved to the next card with room. Balancing applies to the
        hash-placed remainder only: it is not allowed to evict a carried or
        observed placement, because that would move a measurement.

    `balance=False` gives the stronger stability property — a key's card then
    depends on nothing but its own name and the fleet — at the cost of an
    uneven work list. Balancing is the default because an idle card is the one
    resource this project cannot recover.
    """
    if rule not in ("hash", "observed"):
        raise CardAssignmentError(
            f"unknown seeding rule {rule!r}; expected 'hash' or 'observed'")
    rows = normalise_fleet(fleet)
    tokens = [row["card"] for row in rows]
    known = set(tokens)
    keys = sorted({str(k) for k in problem_keys})
    if not keys:
        raise CardAssignmentError("no problem keys to assign")

    placed: dict[str, str] = {}
    source: dict[str, str] = {}

    for key in keys:
        if carry is not None:
            prior = carry.problems.get(key)
            if prior in known:
                placed[key], source[key] = prior, "carried"
                continue
            if prior is not None:
                # The card it was on is gone. It gets re-placed below, and the
                # label says why -- a rebuild that silently moves a measured
                # problem is exactly what this module exists to prevent.
                source[key] = "hash-after-card-gone"
        if rule == "observed" and observed:
            got = observed.get(key)
            token = card_token(got) if got is not None else None
            if token in known:
                placed[key] = token
                source[key] = (observed_source or {}).get(key, "observed")
            elif token is not None:
                # An observed card that is not in the fleet is a fleet problem,
                # not a placement problem: say so rather than dropping it.
                source[key] = "hash-observed-off-fleet"

    remainder = [k for k in keys if k not in placed]
    if balance:
        # Water-filling over what the pinned placements already left behind: a
        # cap per card, then the hash seed with a probe to the next card that
        # still has room. A card already above the water line takes none of the
        # remainder -- it is never evicted, because evicting it would move a
        # measurement to a different card, which is the whole defect.
        load = {t: 0 for t in tokens}
        for token in placed.values():
            load[token] += 1
        level = min(load.values())
        while sum(max(0, level - load[t]) for t in tokens) < len(remainder):
            level += 1
        cap = {t: max(load[t], level) for t in tokens}
        slack = sum(cap[t] - load[t] for t in tokens) - len(remainder)
        for t in reversed(tokens):      # one pass suffices: slack < #(cap>load)
            if slack <= 0:
                break
            if cap[t] > load[t]:
                cap[t] -= 1
                slack -= 1
        for key in remainder:
            start = _hash_index(assignment_id, key, len(tokens))
            token = tokens[start]
            for step in range(len(tokens)):
                candidate = tokens[(start + step) % len(tokens)]
                if load[candidate] < cap[candidate]:
                    token = candidate
                    break
            load[token] += 1
            placed[key] = token
            source.setdefault(key, "hash")
    else:
        for key in remainder:
            placed[key] = tokens[_hash_index(assignment_id, key, len(tokens))]
            source.setdefault(key, "hash")

    rule_doc = {
        "name": rule,
        "balance": bool(balance),
        "seed": "blake2s(assignment_id|problem_key) mod len(fleet)",
        "carried_from": carry.assignment_id if carry is not None else None,
        "note": (
            "RECORDED FOR AUDIT, NOT AUTHORITATIVE. `problems` is the "
            "assignment; re-deriving it from this rule after the fleet or the "
            "dataset changes would reshuffle problems that are already "
            "measured. Rebuild with carry=<this assignment> instead."),
    }
    return Assignment(assignment_id, rows, placed, part=part, rule=rule_doc,
                      seed_source=source)


# --- inputs: the dataset, the fleet, the corpus ------------------------------

def dataset_problem_keys(root: Path = ROOT) -> tuple[list[str], str]:
    """Every problem in the dataset as ``Category__problem``, and where from.

    The dataset when it is present, `reference/dataset-meta.json` otherwise —
    the same precedence the leaderboard uses (CLAUDE.md 8). Deliberately NOT
    the candidates plan: keying the assignment on the plan is the defect this
    module exists to remove.
    """
    data = root / "data" / "SOL-ExecBench" / "benchmark"
    if data.is_dir():
        keys = [f"{cat}__{p.name}"
                for cat in EXPECTED_CENSUS if (data / cat).is_dir()
                for p in sorted((data / cat).iterdir())
                if (p / "definition.json").exists()]
        if keys:
            return sorted(keys), str(data.relative_to(root))
    meta = root / "reference" / "dataset-meta.json"
    if meta.exists():
        doc = json.loads(meta.read_text())
        return sorted(doc.get("problems", {})), str(meta.relative_to(root))
    raise CardAssignmentError(
        f"no dataset at {data} and no {meta}; cannot enumerate the 235 "
        "problems. See CLAUDE.md 8.")


def census_of(keys: Iterable[str]) -> dict[str, int]:
    out = {cat: 0 for cat in EXPECTED_CENSUS}
    for key in keys:
        cat = str(key).split("__", 1)[0]
        out[cat] = out.get(cat, 0) + 1
    return out


def fleet_from_artifacts(paths: Iterable[str | Path]) -> list[dict]:
    """Every distinct card that any artifact under *paths* says it ran on.

    Reads `card_identity` blocks that artifacts already stamp, so a fleet can be
    enumerated with no GPU and no ssh. Unidentified blocks are skipped — an
    artifact that could not name its card contributes no card.

    Sorted by hostname then BDF so two runs over the same tree produce the same
    fleet order (and therefore the same balancing tiebreak).
    """
    found: dict[str, dict] = {}
    for base in paths:
        base = Path(base)
        files = [base] if base.is_file() else sorted(base.rglob("*.json"))
        for f in files:
            try:
                doc = json.loads(f.read_text())
            except (OSError, json.JSONDecodeError, ValueError):
                continue
            for identity in _card_identities(doc):
                try:
                    row = fleet_entry(identity)
                except UnidentifiedCard:
                    continue
                found.setdefault(row["card"], row)
    return [found[k] for k in sorted(found, key=lambda t: (
        parse_token(t)["hostname"], parse_token(t)["bdf"]))]


def _card_identities(doc) -> list[Mapping]:
    """`card_identity` blocks in a T_b artifact and `card_check` in a score file."""
    out = []
    if isinstance(doc, Mapping):
        c = doc.get("card_identity")
        if isinstance(c, Mapping):
            out.append(c)
        check = doc.get("card_check")
        if isinstance(check, Mapping):
            for side in ("actual", "anchor"):
                if isinstance(check.get(side), Mapping):
                    out.append(check[side])
    return out


def fleet_from_probe(gpus: Iterable[int | str], prober=None) -> list[dict]:
    """Probe each visible card once for its identity.

    THE ONLY PATH IN THIS MODULE THAT TOUCHES A CARD. It takes a HIP context in
    a child process per card, so it must not run while a timing run holds one.
    Prefer `fleet_from_artifacts()`, which needs no GPU at all.
    """
    if prober is None:                      # pragma: no cover - needs a GPU
        import authoritative_tb
        prober = authoritative_tb.card_identity
    rows = []
    for gpu in gpus:
        identity = prober(str(gpu))
        if not identity or identity.get("identified") is False:
            raise UnidentifiedCard(
                f"HIP_VISIBLE_DEVICES={gpu} could not identify itself: "
                f"{str((identity or {}).get('error'))[:200]}")
        rows.append(fleet_entry(identity))
    return rows


def _tree_cards(tree: Path) -> dict[str, str]:
    """`{problem_key: card token}` for one T_b tree."""
    out: dict[str, str] = {}
    if not Path(tree).is_dir():
        return out
    for f in sorted(Path(tree).glob("*.json")):
        try:
            doc = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        if not isinstance(doc, Mapping):
            continue
        identity = doc.get("card_identity")
        if not isinstance(identity, Mapping):
            continue
        try:
            out[f.stem] = card_token(identity)
        except UnidentifiedCard:
            continue
    return out


def _score_cards(score_dirs: Iterable[Path]) -> dict[str, str]:
    """`{problem_key: card token}` where every score file for the key agrees.

    Disagreement means the two harnesses ran on different cards; the corpus then
    has no single T_k card for that problem and this returns nothing for it,
    rather than picking one.
    """
    seen: dict[str, set[str]] = {}
    for base in score_dirs:
        base = Path(base)
        if not base.is_dir():
            continue
        for f in sorted(base.rglob("*.json")):
            try:
                doc = json.loads(f.read_text())
            except (OSError, json.JSONDecodeError, ValueError):
                continue
            if not isinstance(doc, Mapping) or not doc.get("problem"):
                continue
            actual = (doc.get("card_check") or {}).get("actual")
            if not isinstance(actual, Mapping):
                continue
            try:
                token = card_token(actual)
            except UnidentifiedCard:
                continue
            seen.setdefault(str(doc["problem"]), set()).add(token)
    return {k: next(iter(v)) for k, v in seen.items() if len(v) == 1}


def observed_placements(*, manifest: str | Path | None = None,
                        score_dirs: Sequence[str | Path] = (),
                        tb_trees: Sequence[str | Path] = (),
                        root: Path = ROOT
                        ) -> tuple[dict[str, str], dict[str, str]]:
    """Where the existing corpus already put each problem, and on what evidence.

    Returns ``(placements, labels)``. Priority — and the order is the whole
    argument, see the module docstring:

      1. ``manifest_anchor`` — the tree `manifest["sources"]["t_b"]` names, i.e.
         the card that measured the anchor the manifest PUBLISHES. First so
         that adopting this assignment cannot become a re-selection of which
         measured `T_b` is published (that would raise `T_b`, and `S` with it).
      2. ``t_k_card`` — the card a `T_k` was measured on, when every score file
         for the problem agrees. Costs a re-anchor of that problem; changes no
         published bound.
      3. ``tb_tree:<name>`` — any other T_b tree, in the order given.

    Looks only at WHERE a measurement happened, never at its value.
    """
    placements: dict[str, str] = {}
    labels: dict[str, str] = {}

    def offer(mapping: Mapping[str, str], label: str) -> None:
        for key, token in mapping.items():
            if key not in placements:
                placements[key], labels[key] = token, label

    if manifest is not None:
        mpath = Path(manifest)
        doc = json.loads(mpath.read_text())
        named = ((doc.get("sources") or {}).get("t_b"))
        if not named:
            raise CardAssignmentError(
                f"{mpath} records no sources.t_b, so the card that produced "
                "its anchors cannot be established; pass --tb-tree explicitly "
                "if you know it, but do not assume.")
        tree = Path(named)
        if not tree.is_absolute():
            tree = root / tree
        offer(_tree_cards(tree), "manifest_anchor")

    offer(_score_cards([Path(d) for d in score_dirs]), "t_k_card")

    for tree in tb_trees:
        offer(_tree_cards(Path(tree)), f"tb_tree:{Path(tree).name}")

    return placements, labels


# --- auditing ---------------------------------------------------------------

def verify_tree(assignment: Assignment, tree: str | Path) -> dict:
    """Did the artifacts in *tree* land on the cards this assignment names?

    Reports rather than raises: an `off_assignment` artifact is a real
    measurement on a named card, and the decision about what to do with it
    belongs to the caller. `unassigned` counts artifacts for problems this
    assignment does not cover, which is a coverage question (CLAUDE.md 0), and
    `unidentified` counts artifacts that cannot say where they ran, which is
    neither a pass nor a fail.
    """
    tree = Path(tree)
    report = {
        "tree": str(tree),
        "assignment_id": assignment.assignment_id,
        "digest": assignment.digest,
        "matched": 0, "off_assignment": [], "unidentified": [],
        "unassigned": [], "problems": 0,
    }
    if not tree.is_dir():
        raise CardAssignmentError(f"no such T_b tree: {tree}")
    for f in sorted(tree.glob("*.json")):
        key = f.stem
        assigned = key in assignment.problems
        try:
            doc = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError, ValueError):
            if assigned:
                report["unidentified"].append(key)
                report["problems"] += 1
            continue
        if not isinstance(doc, Mapping):
            continue
        if not assigned:
            # A tree carries book-keeping files too (`_merge-report`,
            # `no-winner.shard*`). Only something that names a problem this
            # assignment covers, or that claims a card, is worth reporting.
            if "card_identity" in doc:
                report["unassigned"].append(key)
                report["problems"] += 1
            continue
        report["problems"] += 1
        try:
            actual = card_token(doc.get("card_identity") or {})
        except UnidentifiedCard:
            # Includes "no card_identity block at all": an artifact that cannot
            # say where it ran is neither a pass nor a fail, and it is not
            # silently skipped either (CLAUDE.md 0).
            report["unidentified"].append(key)
            continue
        want = assignment.token_for(key)
        if actual == want:
            report["matched"] += 1
        else:
            report["off_assignment"].append(
                {"problem": key, "assigned": want, "measured": actual})
    return report


# --- CLI --------------------------------------------------------------------

def _cmd_build(args) -> int:
    keys, key_source = dataset_problem_keys(ROOT)
    census = census_of(keys)
    print(f"dataset: {len(keys)} problems from {key_source}  {census}")
    if len(keys) != EXPECTED_TOTAL and not args.allow_partial_census:
        print(f"REFUSED: expected {EXPECTED_TOTAL} problems "
              f"({EXPECTED_CENSUS}); an assignment over fewer silently shrinks "
              "the benchmark (CLAUDE.md 0). Pass --allow-partial-census to "
              "say so on purpose.", file=sys.stderr)
        return 2

    if args.probe:
        fleet = fleet_from_probe(args.probe)
    else:
        fleet = fleet_from_artifacts(args.fleet_from)
    print(f"fleet: {len(fleet)} cards on "
          f"{sorted({r['hostname'] for r in fleet})}")
    if not fleet:
        print("REFUSED: no identified card found; nothing to assign to.",
              file=sys.stderr)
        return 2

    observed: dict[str, str] = {}
    labels: dict[str, str] = {}
    if args.rule == "observed":
        observed, labels = observed_placements(
            manifest=args.manifest, score_dirs=args.scores,
            tb_trees=args.tb_tree, root=ROOT)
        by_label: dict[str, int] = {}
        for label in labels.values():
            by_label[label] = by_label.get(label, 0) + 1
        print(f"observed placements: {len(observed)}  {by_label}")

    carry = load(args.carry) if args.carry else None
    assignment = build(keys, fleet, assignment_id=args.assignment_id,
                       part=args.part, rule=args.rule, observed=observed,
                       observed_source=labels, carry=carry,
                       balance=not args.no_balance)

    counts = assignment.counts()
    seeds: dict[str, int] = {}
    for label in assignment.seed_source.values():
        seeds[label] = seeds.get(label, 0) + 1
    print(f"seed sources: {seeds}")
    print(f"work list per card: min {min(counts.values())} "
          f"max {max(counts.values())} over {len(counts)} cards")
    out = Path(args.out) if args.out else default_path(args.part, ROOT)
    if args.dry_run:
        print(f"(dry run) would write {out}; digest {assignment.digest}")
        return 0
    write(assignment, out)
    print(f"wrote {out}  digest {assignment.digest}")
    return 0


def _cmd_show(args) -> int:
    a = load(args.file or default_path(args.part, ROOT))
    print(f"{a.assignment_id}  part={a.part}  cards={len(a.fleet)}  "
          f"problems={len(a.problems)}")
    print(f"digest {a.digest}")
    print(f"rule   {json.dumps(a.rule)}")
    for row in a.fleet:
        print(f"  {row['hostname']:<14} {row['bdf']:<13} "
              f"{row['uuid']:<40} {a.counts()[row['card']]:>4} problems")
    return 0


def _cmd_keys(args) -> int:
    a = load(args.file or default_path(args.part, ROOT))
    matches = [row for row in a.fleet
               if args.host in row["hostname"] and row["bdf"] == args.bdf]
    if len(matches) != 1:
        print(f"REFUSED: --host {args.host} --bdf {args.bdf} matches "
              f"{len(matches)} cards; a card must be named unambiguously.",
              file=sys.stderr)
        return 2
    for key in a.keys_for_card(matches[0]):
        print(key)
    return 0


def _cmd_verify(args) -> int:
    a = load(args.file or default_path(args.part, ROOT))
    report = verify_tree(a, args.tree)
    limit = args.limit
    out = {k: v for k, v in report.items()
           if k not in ("off_assignment", "unidentified", "unassigned")}
    for field in ("off_assignment", "unidentified", "unassigned"):
        out[f"{field}_count"] = len(report[field])
        out[field] = report[field][:limit]
        out[f"{field}_truncated"] = len(report[field]) > limit
    print(json.dumps(out, indent=2))         # always valid JSON, never sliced
    return 0 if not report["off_assignment"] else 1


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="materialise an assignment")
    b.add_argument("--part", required=True)
    b.add_argument("--assignment-id", required=True)
    b.add_argument("--fleet-from", action="append", default=[],
                   help="directory of artifacts to read card identities from")
    b.add_argument("--probe", nargs="*", type=str,
                   help="TOUCHES CARDS: probe these HIP_VISIBLE_DEVICES values")
    b.add_argument("--rule", choices=("hash", "observed"), default="hash")
    b.add_argument("--manifest", help="observed rule: whose sources.t_b names "
                                      "the tree that published the anchors")
    b.add_argument("--scores", action="append", default=[])
    b.add_argument("--tb-tree", action="append", default=[])
    b.add_argument("--carry", help="a previous card-assignment.json to keep")
    b.add_argument("--no-balance", action="store_true")
    b.add_argument("--allow-partial-census", action="store_true")
    b.add_argument("--dry-run", action="store_true")
    b.add_argument("--out")
    b.set_defaults(func=_cmd_build)

    s = sub.add_parser("show", help="summarise an assignment")
    s.add_argument("--part")
    s.add_argument("--file")
    s.set_defaults(func=_cmd_show)

    k = sub.add_parser("keys", help="the work list for one card")
    k.add_argument("--part")
    k.add_argument("--file")
    k.add_argument("--host", required=True)
    k.add_argument("--bdf", required=True)
    k.set_defaults(func=_cmd_keys)

    v = sub.add_parser("verify", help="did a T_b tree land on its cards?")
    v.add_argument("--part")
    v.add_argument("--file")
    v.add_argument("--tree", required=True)
    v.add_argument("--limit", type=int, default=20,
                   help="how many entries of each list to print (counts are "
                        "always complete)")
    v.set_defaults(func=_cmd_verify)

    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except CardAssignmentError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
