# SPDX-License-Identifier: Apache-2.0
"""Bracket the timed window with a clock sample on each side.

**What this is.** ``docs/TODO-MI355X.md`` §4.3 lists four ways to score on a part
whose clock cannot be pinned. The maintainer chose **option 2, "bracket the
window"**: sample the GFX clock immediately before and immediately after the
timed region, record *both*, and **refuse** the measurement when they disagree by
more than a stated threshold.

**What this is not, stated first because it is the tempting misreading.**

* It does **not** give you the clock during the window. Two samples around a
  1-13 ms region are two samples; the finest in-loop sampler in this repo
  achieves ~860 Hz (STATE.md D20), which is 1 to 11 points across that window and
  is not a clock measurement. What the bracket gives is a *bound on how wrong
  assuming one clock is*, which is a smaller and honest claim.
* It does **not** fix, reduce, or bound the **short-window timing bias**
  (``docs/methodology.md`` §7). That bias is a separate defect and it is
  measured to be large: on ``mia1-p02-g46`` the worst shape reads **+106.9%** per
  iteration at ``time_runnable``'s own burst length (warmup 10 + iterations 50)
  against a 50,000-iteration sustained loop. It is also measured **not** to be a
  clock effect: the per-iteration cost attributes as 21.1 µs (GEMM 4096³),
  12.6 µs (GEMM 1024³), 1.2 µs (elementwise) -- an 18× spread across shapes,
  where the probe's own conclusion is that *"a depressed clock would slow all of
  them alike"*. A perfectly tight bracket on a perfectly steady clock leaves that
  bias exactly where it was. Nothing in this module may be read as addressing it.

**Where the window is.** ``time_runnable`` (``timing.py``), called per workload
from the eval driver, and *only* that call. Not ``evaluate()``: that spawns a
fresh ``eval_driver`` subprocess and includes packaging, compilation,
``max_autotune`` and the whole correctness pass, so a bracket around it reports
the clock of mostly-compilation. That was tried; it produced a kernel share of
0.8-55% of the bracketed span and refused 85% of measurements. By the time the
eval driver reaches ``time_runnable`` the user function has already been called
repeatedly by the correctness pass, so compilation and autotuning are behind it.

**The threshold.** ``DEFAULT_BRACKET_THRESHOLD`` below; its derivation is in that
constant's comment, from measured data, and it is overridable per run by
``SOLEXBENCH_CLOCK_BRACKET_THRESHOLD`` because the first MI355X sweep will
produce a better-conditioned distribution than the one it was derived from.

**Two things measured on ``mia1-p02-g46`` GPU 0 while wiring this up**, both
worth knowing before reading a refusal:

* **An SMI read costs 0.23-0.55 ms.** So on a 1 ms window the bracket spans
  roughly twice the region it brackets, and on a 13 ms window about 1.06x. That
  is recorded per measurement as ``clock_bracket_lag_ns`` rather than left for a
  reader to guess.
* **The relative spread has a floor problem at idle clocks, and only there.** An
  idle card reads ~193 MHz and jitters by 2 MHz, which is 1.04% relative and is
  refused -- while the same 2 MHz at the 1739-2394 MHz a loaded card holds is
  0.08-0.12% and is not. The threshold was derived from loaded cards and applies
  to loaded cards; the timed window always follows ``warmup_runs`` iterations, so
  a scoring measurement is never taken on an idle card. It is stated because a
  refusal seen while poking at an idle GPU is this, not a fault, and because if
  a workload ever *does* run at idle clocks the rule as written will refuse it
  and that refusal will be about quantisation rather than about the clock. No
  absolute floor has been added: that would be a methodology change made to get
  past an obstacle, which is prime directive 7.

**Off by default.** Bracketing engages only under
``SOLEXBENCH_CLOCK_BASIS=unlocked``. On a locked part (MI350X, F_LOCK 1300) the
mechanism is unnecessary and switching it on would change how every number is
taken, which is a methodology change to an already-measured corpus (prime
directive 7). The default basis is ``locked`` and takes the code path that
existed before this module.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional, Sequence

__all__ = [
    "DEFAULT_BRACKET_THRESHOLD",
    "ClockBracket",
    "bracket_threshold",
    "bracketed",
    "settle_clock",
    "settle_enabled",
    "bracketing_enabled",
    "clock_basis",
    "sample_clock_mhz",
    "clock_interval",
    "has_clock_interval",
]

CLOCK_BASIS_ENV = "SOLEXBENCH_CLOCK_BASIS"
THRESHOLD_ENV = "SOLEXBENCH_CLOCK_BRACKET_THRESHOLD"
#: Set to "0" to take bracketed measurements WITHOUT the pre-window settle.
#: Exists so the settle's effect can be measured against its own absence on the
#: same card in the same session -- which is the only comparison that means
#: anything, since a before/after taken hours apart also differs in what the
#: neighbouring GPUs were doing. Not a performance knob: turning it off restores
#: the ramp the settle exists to remove.
SETTLE_ENV = "SOLEXBENCH_CLOCK_SETTLE"

#: Relative spread, ``|after - before| / mean(before, after)``, above which a
#: measurement is refused.
#:
#: **0.0078 is the 99th percentile of 6,544 measured consecutive-sample clock
#: spreads** computed from ``artifacts/01/unlocked-clock.json`` -- every adjacent
#: pair in every ``clock_series`` in that artifact, i.e. one MI355X card, under
#: sustained load, unlocked, across four kernel shapes (gemm_dense, gemm_small,
#: memory_bound, reduction), eight cards loaded together, and an eight-minute
#: drift block. That is exactly the statistic this module computes, measured on
#: the part it is for, rather than a number chosen because it looked tidy.
#:
#: The full distribution: median 0.111%, p90 0.284%, p95 0.389%, **p99 0.778%**,
#: p99.5 1.09%, max 26.4%.
#:
#: Three properties make p99 the right cut rather than an arbitrary quantile:
#:
#: 1. **The gap in that artifact is 1 second; the window this guards is
#:    1-13 ms** -- 77x to 1000x shorter. The same drift process observed over a
#:    1000x shorter interval produces a much smaller spread, so the realised
#:    refusal rate should sit well below the 1% the quantile names. The
#:    threshold is therefore conservative in the safe direction: it was
#:    calibrated on a harder problem than the one it is applied to.
#: 2. **It is below every excursion in the artifact.** The nine largest spreads
#:    (8.2%, 10.6%, 11.6%, 13.7%, 17.4%, 19.3%, 22.1%, 26.1%, 26.4%) all occur at
#:    load-transition edges of the eight-loaded blocks -- a card changing power
#:    state. Those are precisely what refusal is for, and 0.78% refuses all of
#:    them by more than an order of magnitude.
#: 3. **It is above ordinary steady-state jitter by 7x** (median 0.111%), so it
#:    does not refuse a card that is simply running.
#:
#: **What it is NOT derived from, deliberately.** The headline unlocked-clock
#: spreads on ``mia1-p02-g46`` -- 36.8% across kernel shapes (gemm_dense
#: 1800 MHz @1383 W vs gemm_small 2392 MHz @673 W) and 3.9% across eight loaded
#: cards -- are *between* shapes and *between* cards. A bracket is within one
#: card, within one shape, within milliseconds. Those numbers say a single F_LOCK
#: cannot describe this part; they say nothing about what two samples milliseconds
#: apart on one card should read, and using them here would set a threshold ~50x
#: too loose to refuse anything.
#:
#: **What has not been measured**, and must be, before this number is treated as
#: settled: no within-window bracket-spread distribution has been collected on
#: ``mia1-p02-g46`` at all. The distribution above is from the ``g10`` artifact in
#: this tree. This is why the threshold and the refusal flag are first-class
#: fields on every artifact rather than a constant nobody can audit: the first
#: MI355X sweep re-derives the quantile from its own recorded spreads, and
#: task 01's acceptance is "the refusal rate is below a stated bound", which
#: cannot be evaluated unless every measurement carries both numbers.
DEFAULT_BRACKET_THRESHOLD = 0.0078


def clock_basis() -> str:
    """``"unlocked"`` or ``"locked"`` (the default). See the module docstring."""
    return (os.environ.get(CLOCK_BASIS_ENV) or "locked").strip().lower()


def bracketing_enabled() -> bool:
    return clock_basis() == "unlocked"


class ClockBasisUnsupported(RuntimeError):
    """A measurement claimed ``locked`` on a part where no lock is achievable."""


def checked_clock_basis(device_name: str | None) -> str:
    """``clock_basis()``, but refusing a ``locked`` claim the part cannot support.

    ``clock_basis()`` defaults to ``"locked"`` when the environment says
    nothing, which is right for the parts that have an achieved F_LOCK and is a
    **false provenance claim** on the ones that do not. On MI355X
    ``--setperfdeterminism`` is a request the cards decline: 15 of 16
    measurements landed at 0.795-0.864x of setpoint, so the preset carries
    ``requested_is_achieved=False`` and ``f_lock_mhz`` is None.

    Forgetting ``SOLEXBENCH_CLOCK_BASIS=unlocked`` on one launch therefore
    produced a sweep that reported "6 ok, 0 failed", stamped ``locked`` on every
    artifact, and was silently unusable -- the manifest dropped all six for
    having no per-measurement clock, and the only visible symptom was the
    problem count going *down*. That is the shape of defect this repo keeps
    finding: it passes.

    So the claim is checked where it is stamped. An unset environment on a part
    with no achievable F_LOCK raises rather than defaulting, because the caller
    genuinely has not said which basis they meant and either answer would be
    invented. An explicit ``locked`` on such a part raises too: saying it does
    not make it true.
    """
    basis = clock_basis()
    if basis != "locked":
        return basis
    try:
        from .config import get_clock_preset  # local: avoids an import cycle
        preset = get_clock_preset(device_name) if device_name else None
    except Exception:  # noqa: BLE001 -- absent config must not mask the check
        preset = None
    if preset is not None and preset.f_lock_mhz is not None:
        return basis
    raise ClockBasisUnsupported(
        f"clock basis 'locked' is not supported on {device_name!r}: no preset "
        f"with an achieved F_LOCK. Set {CLOCK_BASIS_ENV}=unlocked to measure "
        f"against per-measurement bracketed clocks. Refusing rather than "
        f"defaulting, because an artifact stamped 'locked' on a part that was "
        f"never locked is a false provenance claim, and everything downstream "
        f"that divides a bound by F_LOCK will silently drop it instead."
    )


def settle_enabled() -> bool:
    """Is the pre-window settle in force? Only ever when bracketing is."""
    if not bracketing_enabled():
        return False
    return (os.environ.get(SETTLE_ENV) or "1").strip() not in ("0", "false", "no")


def bracket_threshold() -> float:
    """The refusal threshold in force for this process.

    Overridable so a run can adopt its own re-derived quantile without a source
    edit -- and the value in force lands in every artifact, so an override
    cannot be silent.
    """
    raw = os.environ.get(THRESHOLD_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_BRACKET_THRESHOLD
    try:
        v = float(raw)
    except ValueError as e:
        raise ValueError(
            f"{THRESHOLD_ENV}={raw!r} is not a number. Refusing to fall back to "
            f"the default silently: a typo here would loosen or tighten every "
            f"refusal in the run with nothing in the artifact to show for it."
        ) from e
    if not (v > 0):
        raise ValueError(
            f"{THRESHOLD_ENV}={raw!r} must be positive; a threshold of {v} would "
            f"refuse (or admit) every measurement unconditionally."
        )
    return v


def sample_clock_mhz(device: Any = None) -> Optional[int]:
    """Current GFX clock for *device*, or None if it cannot be read.

    Goes through ``device.current_clock_mhz``, which is the SMI path the rest of
    the repo uses and which resolves the torch index to an amdsmi handle **by PCI
    identity**. That translation is not optional: torch and amd-smi order devices
    differently and on an 8x node the map is scrambled
    (torch [0..7] -> amdsmi [3,0,2,1,7,4,6,5]), so indexing positionally reads a
    *different physical card* and returns an entirely plausible number. Commit
    17cbd782 fixed two live instances of exactly that bug.

    **This no longer swallows exceptions, and that is the whole point.** It used
    to wrap the call in ``except Exception: return None``, reasoning that a clock
    which cannot be read is a result and that raising would lose the latency
    measurement too. That is right about *telemetry* failure and wrong about
    *programming* failure, and the two were indistinguishable. The eval driver
    passes the string ``"cuda:0"``; ``int(getattr("cuda:0", "index", ...))``
    raised ``TypeError`` on the way in; the first bracketed T_b sweep refused
    **100% of its workloads** for "no clock evidence" with nothing anywhere
    saying why. A ``None`` that actually means "this call was malformed" is the
    most expensive kind of ``None`` in this repo.

    The split now lives one level down in ``device.current_clock_mhz``: genuine
    telemetry failure still returns None, a malformed request raises, and
    ``bracketed()`` records a raise as ``sampler_error`` carrying the message
    rather than as absent evidence.
    """
    from sol_execbench.core.bench import device as device_layer

    return device_layer.current_clock_mhz(device)


@dataclass
class ClockBracket:
    """The two samples around one timed window, and the verdict they support."""

    clock_before_mhz: Optional[float]
    clock_after_mhz: Optional[float]
    clock_bracket_spread: Optional[float]
    clock_bracket_threshold: float
    clock_bracket_refused: bool
    clock_bracket_refused_reason: Optional[str]
    #: Mean of the two samples. The single number a bound may be evaluated at --
    #: and only meaningful when ``clock_bracket_refused`` is False, which is why
    #: the flag travels beside it in the same record rather than in a log line.
    clock_mhz: Optional[float]
    #: ``[t0, t1]`` in ``time.monotonic_ns()``, delimiting the timed region.
    #:
    #: Host domain, deliberately. This window is *documentation of when the two
    #: samples were taken*; it is never used to bisect activity records, so the
    #: HSA-vs-CLOCK_MONOTONIC domain trap (``reference/contracts/rocprof_shim.md``)
    #: does not apply to it. Attribution windows still come from
    #: ``source.timestamp()`` inside ``bench_gpu_time_with_rocprof``.
    window_ns: Optional[Sequence[int]]
    window_ms: Optional[float]
    #: How far each sample reaches *outside* the window, in ns: the duration of
    #: the "before" read (which ends at ``t0``) and of the "after" read (which
    #: begins at ``t1``). An SMI read costs milliseconds; on a 1 ms window the
    #: bracket can span ten times the region it brackets, and a reader deciding
    #: how much the bracket is worth needs to see that rather than infer it.
    clock_bracket_lag_ns: Optional[Sequence[int]] = None
    clock_bracket_device: Optional[str] = None
    #: Provenance of the samples: ``"amdsmi"``, ``"unavailable"`` (the sampler
    #: answered and had nothing), or ``"sampler_error"`` (it could not be asked).
    clock_bracket_source: str = "amdsmi"
    #: What the pre-window settle did, or None when it was not run. Carried per
    #: measurement because `settle_capped` means this particular measurement was
    #: NOT taken under the condition the settle claims to establish, and that is
    #: not a property of the run as a whole.
    settle: Optional[dict] = None
    #: The exception text when the sampler raised. Carried on the record rather
    #: than logged, because the log of a 3717-workload sweep is not where anyone
    #: finds out that every measurement in it was refused for the same defect.
    clock_bracket_sampler_error: Optional[str] = None

    def as_dict(self) -> dict:
        return asdict(self)


def _spread(before: Optional[float], after: Optional[float]) -> Optional[float]:
    if before is None or after is None:
        return None
    mean = (float(before) + float(after)) / 2.0
    if mean <= 0:
        return None
    return abs(float(after) - float(before)) / mean


def make_bracket(
    before: Optional[float],
    after: Optional[float],
    *,
    threshold: Optional[float] = None,
    window_ns: Optional[Sequence[int]] = None,
    lag_ns: Optional[Sequence[int]] = None,
    device: Optional[str] = None,
    sampler_error: Optional[str] = None,
    settle: Optional[dict] = None,
) -> ClockBracket:
    """Assemble a verdict from two samples. Pure; the unit under test."""
    thr = bracket_threshold() if threshold is None else float(threshold)
    spread = _spread(before, after)
    if sampler_error is not None:
        # Distinct from `no_clock_evidence`, which means the sampler answered
        # and had nothing to say. This means it could not be asked, and it is a
        # DEFECT rather than a measurement condition -- the difference between
        # "the node's telemetry is down" and "we are calling it wrong".
        refused, reason, mhz = True, "sampler_error", None
    elif spread is None:
        refused, reason, mhz = True, "no_clock_evidence", None
    elif spread > thr:
        refused, reason = True, "bracket_spread_above_threshold"
        mhz = (float(before) + float(after)) / 2.0
    else:
        refused, reason = False, None
        mhz = (float(before) + float(after)) / 2.0
    w_ms = None
    if window_ns is not None and len(window_ns) == 2:
        w_ms = (int(window_ns[1]) - int(window_ns[0])) / 1e6
    return ClockBracket(
        clock_before_mhz=before,
        clock_after_mhz=after,
        clock_bracket_spread=spread,
        clock_bracket_threshold=thr,
        clock_bracket_refused=refused,
        clock_bracket_refused_reason=reason,
        clock_mhz=mhz,
        window_ns=list(window_ns) if window_ns is not None else None,
        window_ms=w_ms,
        clock_bracket_lag_ns=list(lag_ns) if lag_ns is not None else None,
        clock_bracket_device=device,
        clock_bracket_source=(
            "sampler_error" if sampler_error is not None
            else "amdsmi" if spread is not None else "unavailable"),
        clock_bracket_sampler_error=sampler_error,
        settle=settle,
    )


#: Caps on the pre-window settle. Both exist so a card that never settles cannot
#: hang a sweep; whichever binds first is recorded on the measurement.
SETTLE_MAX_MS = 1000.0
SETTLE_MAX_ITERS = 20000
#: Consecutive clock samples that must agree before the card counts as settled.
#: Three, not two: the ramp measured on this part is monotonic rather than noisy,
#: so a pair agrees by accident every time the climb momentarily flattens, and a
#: triple is a much stronger test for the cost of one extra batch.
SETTLE_STABLE_SAMPLES = 3


def settle_clock(
    kernel: Callable[[], Any],
    *,
    device: Any = None,
    band: Optional[float] = None,
    sampler: Optional[Callable[[Any], Optional[int]]] = None,
    synchronize: Optional[Callable[[], None]] = None,
    window_iters: int = 60,
    max_ms: float = SETTLE_MAX_MS,
    max_iters: int = SETTLE_MAX_ITERS,
) -> dict:
    """Run *kernel* until the clock stops climbing, before the window opens.

    **Why this is not a methodology change, which the next reader will doubt.**
    The measured quantity does not move. The timed region is still upstream's
    ``warmup_runs=10, iterations=50`` around the same callable, and this runs
    entirely *before* the first clock sample and the first timing event. That is
    exactly the distinction from ``docs/TODO-MI355X.md`` §4.3 option 1:
    lengthening the window to ~10,000 iterations *would* change the measured
    quantity and break comparability with upstream's numbers, which is why it was
    declined. Settling changes only the state the card is in when measurement
    starts, removing a known artifact of the harness's own duty cycle. Nothing
    about ``T_k``, ``T_b`` or the score's definition changes, and no number
    already measured becomes incomparable.

    **The artifact it removes.** Measured on ``mia1-p02-g46``: the card ramps
    through the whole timed region and is still climbing when it ends. From a
    mild 3 s relax it needs ~117 ms to come within 1% of peak; from deep idle the
    first workload of a process opens its window at 1597 MHz against a ~2400 MHz
    ceiling. The bracket refused 57.8% of measurements on that, and **37 of 37
    refusals were the clock rising**.

    **Settling with the real kernel is required, not incidental.** This part's
    clock is workload-dependent -- gemm_dense holds 1800 MHz at 1383 W while
    gemm_small holds 2392 MHz at 673 W on this node, a 36.8% spread. A synthetic
    settle load would drive the card to a *different* operating point than the
    workload about to be timed, leaving it to ramp (or drop) once real work
    began. That is worse than not settling, because it would look like it worked.

    **Settling to a condition, not a duration.** A fixed sleep would be right in
    the middle of the distribution and wrong at the process's first workload,
    which is precisely the population failing now. So: batch the kernel, sample
    between batches, and stop when the clock has been flat within *band* -- by
    default the same threshold the bracket will apply.

    **Flat over the window's own duration, which is the part that is easy to get
    wrong and was.** The first version of this exited when the last three samples
    (~10 ms apart) agreed within the band. That is a test on the *slope*, and it
    passes early on a slow ramp: 2340 -> 2348 -> 2355 spans 0.64% over 30 ms and
    reads as settled, while the same ramp over the ~100 ms window the bracket
    actually measures is 6% and is refused. Measured consequence: the settle
    exited after a median of **12 ms and 6 iterations** and the refusal rate got
    *worse*, 53.1% -> 78.1%. The stability window must therefore be at least as
    long as the timed window it is protecting -- hence *window_iters*, from which
    the expected window duration is computed using the per-iteration cost this
    function measures anyway. Testing stability on a shorter horizon than the one
    you will be judged on is the whole error.

    Returns a record for the artifact. A settle that hit either cap sets
    ``settle_capped``, and that must stay visible per measurement: a card that
    never settled produces a measurement not taken under the condition this
    function claims to establish, and averaging it in silently would hide exactly
    the population we are trying to fix.
    """
    sample = sampler or sample_clock_mhz
    sync = synchronize
    if sync is None:
        def sync() -> None:
            import torch

            torch.cuda.synchronize()

    thr = bracket_threshold() if band is None else float(band)
    t_start = time.monotonic_ns()
    entry = sample(device)
    # (timestamp_ns, mhz) so stability can be judged over a TIME span rather
    # than over a sample count.
    trace: list[tuple[int, Optional[int]]] = [(time.monotonic_ns(), entry)]
    iters = 0
    batch = 1
    per_iter_ms = None
    capped_by = None

    while True:
        b0 = time.monotonic_ns()
        for _ in range(batch):
            kernel()
        sync()
        b1 = time.monotonic_ns()
        iters += batch
        trace.append((time.monotonic_ns(), sample(device)))

        # Aim each batch at ~10 ms of real work: short enough to resolve the
        # ramp, long enough that the SMI read (0.23-0.55 ms on this node) is not
        # most of the interval. Derived from the batch just run rather than
        # assumed, because per-iteration cost spans 18x across this corpus.
        per_iter_ms = max((b1 - b0) / 1e6 / batch, 1e-4)
        batch = max(1, min(1024, int(10.0 / per_iter_ms)))

        # The horizon the bracket will judge this measurement over.
        need_ms = window_iters * per_iter_ms
        now = time.monotonic_ns()
        # Samples inside the trailing horizon...
        recent = [(t, c) for t, c in trace
                  if c and (now - t) / 1e6 <= max(need_ms, 1.0)]
        # ...and proof that we have actually BEEN running for that long. The
        # first version asked `now - recent[0] >= need_ms`, but recent[0] is by
        # construction inside the horizon, so the test was
        # `age <= need_ms and age >= need_ms` -- satisfiable only by exact
        # equality, i.e. never. Every settle therefore ran to the 1000 ms cap
        # and the stability criterion was dead code that looked alive. The
        # coverage check belongs on the OLDEST sample overall, not the oldest
        # recent one.
        covered_ms = (now - trace[0][0]) / 1e6
        if len(recent) >= SETTLE_STABLE_SAMPLES and covered_ms >= need_ms:
            clks = [c for _, c in recent]
            lo, hi = min(clks), max(clks)
            mean = (lo + hi) / 2.0
            if mean > 0 and (hi - lo) / mean <= thr:
                break

        elapsed_ms = (now - t_start) / 1e6
        if elapsed_ms >= max_ms:
            capped_by = "max_ms"
            break
        if iters >= max_iters:
            capped_by = "max_iters"
            break

    samples = [c for _, c in trace]
    seen = [s for s in samples if s]
    return {
        "settled": capped_by is None,
        "settle_capped": capped_by,
        "settle_ms": (time.monotonic_ns() - t_start) / 1e6,
        "settle_iterations": iters,
        "settle_band": thr,
        "settle_entry_mhz": entry,
        "settle_exit_mhz": samples[-1],
        # The whole climb, not only its endpoints: an entry-to-exit delta of zero
        # can mean "already settled" or "never moved because telemetry is
        # broken", and the range separates them.
        "settle_min_mhz": min(seen) if seen else None,
        "settle_max_mhz": max(seen) if seen else None,
        "settle_n_samples": len(samples),
        # The horizon stability was judged over. Without it a reader cannot tell
        # a settle that held for 300 ms from one that held for 12 ms, and that
        # difference is the entire fix.
        "settle_stability_horizon_ms": (window_iters * per_iter_ms
                                        if per_iter_ms else None),
    }


def bracketed(
    thunk: Callable[[], Any],
    *,
    device: Any = None,
    sampler: Optional[Callable[[Any], Optional[int]]] = None,
    threshold: Optional[float] = None,
    settle: Optional[Callable[[], Any]] = None,
    window_iters: int = 60,
) -> tuple[Any, ClockBracket]:
    """Run *thunk*, bracketed by a clock sample on each side.

    Ordering is load-bearing and is the reason this is a function rather than
    four lines at the call site::

        sample -> t0 -> thunk() -> t1 -> sample

    The samples sit *outside* ``[t0, t1]`` so that SMI latency is not counted as
    part of the timed window, and the gap it introduces is recorded as
    ``clock_bracket_lag_ns`` rather than hidden.

    A raising *thunk* propagates: a failed timing has no measurement to attach a
    bracket to, and swallowing it here would turn a run-time error into a
    plausible number.

    A raising *sampler* does not propagate -- it is recorded as
    ``sampler_error``, the timing still runs, and the measurement is refused.
    That is deliberate and it is the opposite of what the code did before: the
    error text now travels on the artifact, so a systematic sampling defect is
    visible per measurement instead of being flattened into the same
    ``no_clock_evidence`` a quiet node produces.
    """
    sample = sampler or sample_clock_mhz
    errors: list[str] = []

    def _try_sample():
        try:
            return sample(device)
        except Exception as e:                               # noqa: BLE001
            errors.append(f"{type(e).__name__}: {e}")
            return None

    # -- Settle the card BEFORE the first sample, or the settle is pointless.
    #
    # Order matters and is the whole design: settle -> sample -> t0 -> thunk ->
    # t1 -> sample. A settle that ran after the "before" sample, or inside the
    # window, would leave the bracket measuring the ramp it was meant to remove.
    # It contributes nothing to the reported latency -- that comes from event
    # pairs recorded per iteration inside `time_runnable`, well after this
    # returns -- and `tests/.../test_setup_outside_timed_region.py` pins that.
    settle_record = None
    if settle is not None:
        try:
            settle_record = settle_clock(settle, device=device, sampler=sample,
                                         band=threshold,
                                         window_iters=window_iters)
        except Exception as e:                               # noqa: BLE001
            # A settle that fails is not a measurement that fails. Record it and
            # take the measurement anyway; the bracket will refuse it on its own
            # merits if the card is still moving.
            settle_record = {"settled": False, "settle_error": f"{type(e).__name__}: {e}"}

    # The lag is measured from the START of each sample, not its end: what a
    # reader needs is how far back in time the "before" reading could have been
    # taken, which is the whole duration of the read, not the gap after it.
    pre_start = time.monotonic_ns()
    before = _try_sample()
    t0 = time.monotonic_ns()
    result = thunk()
    t1 = time.monotonic_ns()
    after = _try_sample()
    post_done = time.monotonic_ns()
    return result, make_bracket(
        before,
        after,
        threshold=threshold,
        window_ns=(t0, t1),
        lag_ns=(t0 - pre_start, post_done - t1),
        device=str(device) if device is not None else None,
        sampler_error=errors[0] if errors else None,
        settle=settle_record,
    )


#: Keys a consumer must find on a per-workload record for it to count as
#: carrying clock evidence. Listed once so the writer, the manifest builder and
#: the tests cannot drift apart about what "has a clock" means.
BRACKET_FIELDS: tuple[str, ...] = (
    "clock_before_mhz",
    "clock_after_mhz",
    "clock_bracket_spread",
    "clock_bracket_threshold",
    "clock_bracket_refused",
)


def has_clock_evidence(record: Optional[dict]) -> bool:
    """True when *record* carries a usable, non-refused bracket.

    "Unknown clock" and "clock outside threshold" both answer False. An unknown
    clock is not a permissive one -- the same reading ``build_manifest.py``
    applies to a missing F_LOCK.

    **This is the STRICT, point-estimate predicate, and it is deliberately
    unchanged.** It answers "is there a single frequency that describes this
    window?", which a wide bracket genuinely does not have. Under the interval
    methodology the consumers that publish a bound ask a weaker and more useful
    question instead -- ``has_clock_interval`` below -- but this one is still the
    right test wherever a *single* clock is what is needed, and it is the test the
    task-06 sweep runner applies when selecting winners.
    """
    if not record:
        return False
    if record.get("clock_bracket_refused") is not False:
        return False
    mhz = record.get("clock_mhz")
    return isinstance(mhz, (int, float)) and mhz > 0


def clock_interval(record: Optional[dict]) -> Optional[tuple[float, float]]:
    """``(f_min, f_max)`` from a bracket's two samples, or None if it has neither.

    The spread verdict is **not** consulted. A bracket refused for spread is a
    bracket that measured a moving clock, and the two numbers it measured are the
    two most informative numbers about that window that exist -- refusing to look
    at them throws away the evidence and keeps the complaint.

    None comes back only when there is genuinely nothing to read: a sampler error,
    an absent sample, or a non-positive one. Those are unchanged; an unknown clock
    is still not a permissive one.

    ``sampler_error`` disqualifies a record even when both samples are present. The
    error means the sampling path is DEFECTIVE -- the case that produced it read a
    neighbouring card's clock through a scrambled index -- so the numbers beside it
    are not weak evidence, they are evidence about something else.
    """
    if not record:
        return None
    if record.get("clock_bracket_sampler_error"):
        return None
    before, after = record.get("clock_before_mhz"), record.get("clock_after_mhz")
    if not all(isinstance(v, (int, float)) and v > 0 for v in (before, after)):
        return None
    return (float(min(before, after)), float(max(before, after)))


def has_clock_interval(record: Optional[dict]) -> bool:
    """True when *record* supports an interval-valued bound. The demoted gate.

    **This is where refusal stops being a gate and becomes a label.** The reasoning
    is in ``solexbench_rocm.t_sol_at``: with both roofline terms carried, a bracket
    that spans 1607-2148 MHz does not make a measurement unusable, it makes it
    usable with a stated width. Discarding it was the right call only while T_SOL
    had to be a single number.

    What did *not* change, and must not be read as having changed:

    * ``clock_bracket_refused`` is still set, still carries its reason, and is still
      counted by ``summarize_brackets``. The label survives the demotion -- a
      consumer can still filter on it, task 01's "refusal rate below a stated bound"
      acceptance still has its rate, and the wide brackets are still visible as the
      lower-quality measurements they are.
    * ``sampler_error`` and ``no_clock_evidence`` still answer False here. Those are
      not wide brackets, they are absent ones, and no width can be stated for a
      window nobody sampled. ``clock_fatalities`` in the sweep runner keeps failing
      the run closed on them, which is a different failure and stays a gate.
    """
    return clock_interval(record) is not None


def summarize_brackets(records: Sequence[Optional[dict]]) -> dict:
    """Refusal counts over a set of bracket records. A first-class artifact field.

    Task 01's acceptance on an unlocked part is "the refusal rate is below a
    stated bound", so the rate has to be a number in the artifact, not something
    a reader recovers by grepping logs.

    **The refusal counts here are unchanged by the demotion of refusal from a gate
    to a label.** ``n_refused``, ``refusal_rate`` and ``refused_by_reason`` count
    exactly what they counted before, and ``clock_fatalities`` -- which reads all
    three -- keeps behaving identically. What is *added* is the interval-era split
    of those same refusals: how many of them still carry two usable samples and are
    therefore admitted with a stated width (``n_refused_with_interval``), against
    how many carry no clock at all and remain unusable (``n_without_interval``).
    A rate that fell because the rule loosened would be a rate that stopped meaning
    anything, so it was left alone and the new question got new names.
    """
    seen = [r for r in records if r]
    refused = [r for r in seen if r.get("clock_bracket_refused")]
    with_interval = [r for r in seen if has_clock_interval(r)]
    refused_with_interval = [r for r in refused if has_clock_interval(r)]
    reasons: dict[str, int] = {}
    for r in refused:
        k = r.get("clock_bracket_refused_reason") or "unknown"
        reasons[k] = reasons.get(k, 0) + 1
    spreads = sorted(
        r["clock_bracket_spread"] for r in seen
        if isinstance(r.get("clock_bracket_spread"), (int, float))
    )
    thresholds = sorted({
        r["clock_bracket_threshold"] for r in seen
        if isinstance(r.get("clock_bracket_threshold"), (int, float))
    })
    return {
        "n_bracketed": len(seen),
        "n_refused": len(refused),
        "refusal_rate": (len(refused) / len(seen)) if seen else None,
        "refused_by_reason": reasons,
        # -- The interval-era split of the same population. Added, never
        # substituted: the three counters above are what task 01 gates on.
        "n_with_interval": len(with_interval),
        "n_refused_with_interval": len(refused_with_interval),
        "n_without_interval": len(seen) - len(with_interval),
        # A mixture is reported as a mixture. Two thresholds in one artifact
        # means two policies produced it, which is a fact about the artifact.
        "thresholds_in_force": thresholds,
        "spread_median": spreads[len(spreads) // 2] if spreads else None,
        "spread_max": spreads[-1] if spreads else None,
    }
