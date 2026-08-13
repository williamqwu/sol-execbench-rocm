# SPDX-License-Identifier: Apache-2.0
"""The input-generation contract shared by the golden and the tolerance path.

CPU-only. No GPU is touched: every assertion here is about *which device is
asked for*, never about running on one.

STATE.md D53. `scripts/gen_golden.py` drew its inputs at ``device="cpu"`` while
`scripts/runners/calibrate_tolerance.py` drew at the ``prepare_inputs`` default
of ``"cuda:0"``. Both called ``torch.manual_seed(0)`` first, so the code read as
if it were reproducing the same inputs. It was not: ``torch.manual_seed`` seeds
every device's default generator, but CPU runs ``at::mt19937`` and CUDA/HIP runs
Philox4_32_10, so the same seed on the two devices is two unrelated streams. The
golden was the correct answer to a question nobody had asked, and the
comparison against it -- computed, stored as ``vs_golden`` -- was never read.
2302 of the 2331 recorded comparisons (98.756%, across 164 problems) exceed
their own derived atol -- recomputed 2026-08-12 from artifacts/05/*.json.

Nothing raised. Nothing warned. The only way this shows up is a test that pins
the contract, which is what this file is. It covers:

  1. the device is named ONCE (`_common.INPUT_DEVICE`) and both paths use it;
  2. the seed the golden draws at is the seed task 05 compares against;
  3. `torch.randn`'s ``device=`` argument really does select the generator, so
     the contract in (1) is load-bearing rather than cosmetic;
  4. a golden with no input-draw stamp -- i.e. every golden generated before
     this fix -- is NOT treated as cached, and is NOT reported as comparable.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "scripts", ROOT / "scripts" / "runners", ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import gen_golden  # noqa: E402

# `scripts/runners` has no __init__.py, so the runners are imported flat -- the
# same way they import each other, and the same way gen_golden does.
import _common  # noqa: E402
import calibrate_tolerance  # noqa: E402


# --- 1. one device, named once ---------------------------------------------


def test_prepare_inputs_defaults_to_the_shared_constant():
    """The default is the constant, not a copy of its current value.

    A literal "cuda:0" here would let the constant move and leave the default
    behind, which is the same class of drift D53 was.
    """
    default = inspect.signature(_common.prepare_inputs).parameters["device"].default
    assert default == _common.INPUT_DEVICE


def test_the_tolerance_path_draws_at_the_shared_default():
    """calibrate_tolerance passes no device, so it gets INPUT_DEVICE."""
    src = inspect.getsource(calibrate_tolerance.main)
    calls = [ln.strip() for ln in src.splitlines() if "prepare_inputs(" in ln]
    assert calls, "the tolerance path must still generate its own inputs"
    for call in calls:
        assert "device=" not in call, (
            f"the tolerance path is what the benchmark runs; it must take the "
            f"shared default rather than name a device: {call}"
        )


def test_the_shared_constant_is_the_device_the_benchmark_runs_on():
    """A LITERAL, deliberately. This is the anchor every other test hangs off.

    Every assertion below compares something to `_common.INPUT_DEVICE`. If that
    constant could move freely, all of them would be tautologies together. It
    is pinned here, to the device the eval driver and task 05 actually run on;
    changing the benchmark's input device is a decision that must break a test.
    """
    assert _common.INPUT_DEVICE == "cuda:0"


def test_the_golden_draws_on_the_same_device_as_the_tolerance_path():
    """The whole of D53 in one assertion: the CLI's own default device.

    Read out of the parser, not out of the module. The previous version of this
    test asserted `gen_golden.INPUT_DEVICE == _common.INPUT_DEVICE` -- the same
    imported object on both sides -- so `default="cpu"` on the argparse line
    reintroduced D53 as the script's default behaviour with the suite green.
    """
    parser = gen_golden.build_parser()
    assert parser.get_default("input_device") == "cuda:0"
    assert parser.get_default("input_device") == _common.INPUT_DEVICE

    gen_src = inspect.getsource(gen_golden.gen_one)
    assert 'device="cpu"' not in gen_src and "device='cpu'" not in gen_src, (
        "gen_golden must not hardcode a CPU draw: the tolerance path draws on "
        f"{_common.INPUT_DEVICE!r} and a CPU draw is a different input"
    )
    assert "device=input_device" in gen_src


# --- 2. the seed is the seed task 05 compares against -----------------------


def test_golden_seed_matches_the_seed_the_tolerance_compares_against():
    """task 05 compares the golden to `first_outputs`, which is seed 0's."""
    assert gen_golden.GOLDEN_SEED == 0
    assert _common.GOLDEN_SEED == 0
    src = inspect.getsource(calibrate_tolerance.main)
    assert "for seed in range(a.seeds)" in src, (
        "if the seed loop changes shape, re-check which seed's outputs "
        "`first_outputs` holds -- the golden must be drawn at that one"
    )
    assert "if first_outputs is None" in src


# --- 3. the device argument really does pick the generator ------------------


def test_the_device_argument_selects_the_generator():
    """Same seed, non-CPU device => the CPU generator is not consumed.

    This is the mechanism, demonstrated without a GPU. `meta` stands in for any
    non-CPU device: if `randn(device=<not cpu>)` drew from the CPU generator,
    the interleaved draw below would shift the CPU stream. It does not, because
    the `device=` argument routes to that device's own default generator --
    which for CUDA/HIP is a Philox engine, not the CPU's mt19937, and therefore
    emits different numbers from the same seed.

    The end-to-end number (L1__067 seed-0 CPU vs seed-0 cuda:0 hidden_states,
    7.096e+00 apart) is a GPU measurement recorded in STATE.md D53. It is not
    re-measured here: this suite is CPU-only by construction.
    """
    torch = pytest.importorskip("torch")

    torch.manual_seed(0)
    cpu_only = torch.randn(8)

    torch.manual_seed(0)
    torch.randn(8, device="meta")          # a draw on another device
    after_other_device = torch.randn(8)

    assert torch.equal(cpu_only, after_other_device), (
        "a non-CPU draw consumed CPU generator state; the premise of this "
        "test no longer holds"
    )
    # ...and torch.manual_seed seeds those other generators too, which is
    # precisely why the bug was invisible: the seeding call looks sufficient.
    assert "manual_seed_all" in inspect.getsource(torch.random.manual_seed)


def test_moving_inputs_to_cpu_does_not_redraw_them():
    """gen_golden draws on the input device, then computes on the CPU.

    That is only sound if the move is a copy. If it were ever not, the golden
    would silently go back to being a different input.
    """
    torch = pytest.importorskip("torch")

    torch.manual_seed(0)
    drawn = torch.randn(16)
    moved = gen_golden._to_cpu([drawn, 3, 1.5], torch)
    assert torch.equal(moved[0], drawn)
    assert moved[1:] == [3, 1.5], "scalars must pass through untouched"


# --- 4. a pre-fix golden is neither cached nor reported as comparable -------


def _write_pt(d: Path, key: str) -> None:
    (d / f"{key}.pt").write_bytes(b"not really a tensor file")


def test_an_unstamped_golden_is_not_cached(tmp_path):
    """Every golden on disk today is unstamped. None may be reused."""
    _write_pt(tmp_path, "L1__067_x")
    assert gen_golden.is_cached(tmp_path, "L1__067_x", _common.INPUT_DEVICE) is False


def test_a_stamped_golden_from_the_wrong_device_is_not_cached(tmp_path):
    _write_pt(tmp_path, "L1__067_x")
    gen_golden.sidecar_path(tmp_path, "L1__067_x").write_text(
        json.dumps(gen_golden.contract_stamp("cpu")))
    assert gen_golden.is_cached(tmp_path, "L1__067_x", _common.INPUT_DEVICE) is False


def test_a_stamped_golden_from_an_older_contract_is_not_cached(tmp_path):
    _write_pt(tmp_path, "L1__067_x")
    stamp = gen_golden.contract_stamp(_common.INPUT_DEVICE)
    stamp["contract_version"] -= 1
    gen_golden.sidecar_path(tmp_path, "L1__067_x").write_text(json.dumps(stamp))
    assert gen_golden.is_cached(tmp_path, "L1__067_x", _common.INPUT_DEVICE) is False


def test_a_current_golden_is_cached(tmp_path):
    _write_pt(tmp_path, "L1__067_x")
    gen_golden.sidecar_path(tmp_path, "L1__067_x").write_text(
        json.dumps(gen_golden.contract_stamp(_common.INPUT_DEVICE)))
    assert gen_golden.is_cached(tmp_path, "L1__067_x", _common.INPUT_DEVICE) is True


def test_golden_contract_reads_the_sidecar_not_the_pt(tmp_path, monkeypatch):
    """143 GB of `.pt` must never be opened to answer 'is this comparable?'."""
    monkeypatch.setattr(calibrate_tolerance, "GOLDEN_DIR", tmp_path)
    assert calibrate_tolerance.golden_contract("L1__067_x") is None

    stamp = gen_golden.contract_stamp(_common.INPUT_DEVICE)
    gen_golden.sidecar_path(tmp_path, "L1__067_x").write_text(json.dumps(stamp))
    assert calibrate_tolerance.golden_contract("L1__067_x") == stamp


def test_corrupt_sidecar_reads_as_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(calibrate_tolerance, "GOLDEN_DIR", tmp_path)
    gen_golden.sidecar_path(tmp_path, "L1__067_x").write_text("{ truncated")
    assert calibrate_tolerance.golden_contract("L1__067_x") is None
    _write_pt(tmp_path, "L1__067_x")
    assert gen_golden.is_cached(tmp_path, "L1__067_x", _common.INPUT_DEVICE) is False


# --- 5. the reader is exactly as strict as the writer -----------------------
#
# D53 one level up: `gen_golden.is_cached` decides a golden must be REDRAWN
# while `calibrate_tolerance` was still stamping the same golden `comparable:
# true`. Both now call `_common.golden_stamp_matches`; these tests hold the two
# answers together for each way a stamp can be wrong.


def _reader(monkeypatch, tmp_path, stamp, *, with_pt=True):
    """calibrate_tolerance's OWN answer -- the one the artifact is written from.

    `golden_comparability` is the function `main` calls, so this is not a
    restatement of the predicate; it is the code path.
    """
    monkeypatch.setattr(calibrate_tolerance, "GOLDEN_DIR", tmp_path)
    if with_pt:
        _write_pt(tmp_path, "L1__067_x")
    if stamp is not None:
        gen_golden.sidecar_path(tmp_path, "L1__067_x").write_text(
            json.dumps(stamp))
    return calibrate_tolerance.golden_comparability("L1__067_x", with_pt)


@pytest.mark.parametrize("mutate,why", [
    (lambda s: s.update(contract_version=s["contract_version"] - 1),
     "an older contract version"),
    (lambda s: s.update(seed=7), "a different seed"),
    (lambda s: s.update(input_device="cpu"), "a different device"),
    (lambda s: s.pop("seed"), "a missing field"),
])
def test_reader_and_writer_reject_the_same_stamps(
        tmp_path, monkeypatch, mutate, why):
    stamp = gen_golden.contract_stamp(_common.INPUT_DEVICE)
    mutate(stamp)
    writer = gen_golden.is_cached(tmp_path, "L1__067_x", _common.INPUT_DEVICE)
    comparable, _, note = _reader(monkeypatch, tmp_path, stamp)
    assert writer is False, f"the writer must regenerate a golden with {why}"
    assert comparable is False, (
        f"the reader must NOT call a golden with {why} comparable; a reader "
        "more lenient than its writer is what D53 was"
    )
    assert note and "NOT a correctness signal" in note


def test_reader_and_writer_accept_the_current_stamp(tmp_path, monkeypatch):
    stamp = gen_golden.contract_stamp(_common.INPUT_DEVICE)
    comparable, gc, note = _reader(monkeypatch, tmp_path, stamp)
    assert comparable is True and note is None and gc == stamp
    assert gen_golden.is_cached(
        tmp_path, "L1__067_x", _common.INPUT_DEVICE) is True


def test_main_writes_the_artifact_from_that_one_function():
    """Not a second, looser copy of the check inlined in `main`."""
    body = inspect.getsource(calibrate_tolerance.main)
    assert "golden_comparability(" in body
    assert "golden_stamp_matches" not in body, (
        "the reader is applying the predicate by hand again instead of "
        "through the shared function"
    )
    assert "golden_stamp_matches" in inspect.getsource(gen_golden.is_cached)


def test_a_sidecar_without_its_pt_is_not_comparable_and_says_so(
        tmp_path, monkeypatch):
    """`available` and `comparable` must never disagree in silence.

    The .pt files run to 143 GB and get deleted; the 1 KB sidecar survives.
    A perfectly current stamp with nothing under it must read as "no golden",
    with a note -- not as `comparable: true` beside `available: false`.
    """
    stamp = gen_golden.contract_stamp(_common.INPUT_DEVICE)
    comparable, gc, note = _reader(monkeypatch, tmp_path, stamp, with_pt=False)
    assert comparable is False
    assert gc == stamp, "the stamp is still reported; only usability changed"
    assert note and ".pt is not" in note


def test_no_golden_at_all_gets_no_note(tmp_path, monkeypatch):
    """Absent is not suspect: a problem that never had a golden says nothing."""
    monkeypatch.setattr(calibrate_tolerance, "GOLDEN_DIR", tmp_path)
    assert calibrate_tolerance.golden_comparability("L1__067_x", False) == (
        False, None, None)


def test_an_unstamped_golden_is_flagged_as_pre_fix(tmp_path, monkeypatch):
    """The 165 goldens on disk today: a .pt, no sidecar."""
    comparable, gc, note = _reader(monkeypatch, tmp_path, None)
    assert comparable is False and gc is None
    assert note and "predates the D53 fix" in note


# --- 6. worker count, and what is actually known about it -------------------


def test_jobs_default_is_the_pre_fix_value():
    """32, unchanged, because no HIP-context cost was ever measured.

    The D53 fix briefly dropped this to 1, justified by an assertion about HIP
    context footprint that nobody measured, for a hazard (the timing card) that
    `HIP_VISIBLE_DEVICES` already handles. Restored; the debt is in TODO.md.
    """
    assert gen_golden.build_parser().get_default("jobs") == 32
    assert not hasattr(gen_golden, "resolve_jobs"), (
        "resolve_jobs encoded an unmeasured cost as a default"
    )


# --- 7. the report file keeps its shape -------------------------------------


def test_report_json_is_a_flat_problem_mapping():
    """A reader iterating its keys as problems must not meet a non-problem."""
    src = inspect.getsource(gen_golden.main)
    assert '(out / "_report.json").write_text(json.dumps(reports, indent=1))' \
        in src, "_report.json must be the reports mapping and nothing else"
    assert "_contract" not in src


def test_the_contract_is_still_recorded_per_problem():
    """Dropping `_contract` from the report loses nothing: gen_one stamps it."""
    src = inspect.getsource(gen_golden.gen_one)
    assert "**contract_stamp(input_device)" in src


# --- 8. reference-internal randomness is detected on any device -------------


def test_rng_detection_watches_more_than_the_cpu_generator():
    """A reference drawing on a device must not be stamped `false`.

    Checked structurally plus one live CPU case: the device half needs a GPU,
    and this suite has none. `_rng_changed` returns True when the fingerprint
    GAINS a generator, which is what a first device draw does.
    """
    torch = pytest.importorskip("torch")

    fp_src = inspect.getsource(gen_golden._rng_fingerprint)
    assert "torch.cuda.get_rng_state_all()" in fp_src, (
        "only the CPU generator is watched; a device draw reads as no draw"
    )

    before = gen_golden._rng_fingerprint(torch)
    assert gen_golden._rng_changed(before, gen_golden._rng_fingerprint(torch),
                                   torch) is False
    torch.randn(4)
    assert gen_golden._rng_changed(before, gen_golden._rng_fingerprint(torch),
                                   torch) is True
    # A generator appearing where there was none is a draw on a new device.
    assert gen_golden._rng_changed(before, before + [before[0]], torch) is True


# --- 9. the golden is written atomically ------------------------------------


def test_the_pt_is_written_then_renamed():
    """`_common.write_result`'s convention, for the file that is 143 GB.

    A kill mid-`torch.save` that overwrote a good `.pt` in place, with last
    run's matching sidecar still on disk, leaves a truncated golden that reads
    as cached forever.
    """
    src = inspect.getsource(gen_golden.gen_one)
    assert "os.replace(tmp, out_file)" in src
    assert "torch.save(goldens, out_file)" not in src
