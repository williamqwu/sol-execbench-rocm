# Quarantined: 143 authoritative T_b measured at the wrong clock

Every file here records `_provenance.f_lock_mhz: 1640` and was measured at
roughly **1860 MHz**. They are wall-clock times, so each one is about **12%
faster than the number it claims to be**, and nothing in the file says so.

## What happened

`clock_calibrate.py determinism-sweep` applies a setpoint per step and never
resets. The last step of `artifacts/01/determinism-sweep-gpu2.json` requested
1900 MHz at 08:48 UTC on 2026-08-04, and the node was still there eleven hours
later: `perf_determinism`, setpoint 1900, DPM level 1 at 1843 MHz, measured
1848-1870 MHz under the timing load.

## Why no check caught it

`provenance.f_lock_mhz()` resolved the stamp from `CLOCK_LOCK_PRESETS`, never
from the device. Its docstring argued that reading the value "from the same
table the lock is applied from means the recorded clock and the applied clock
cannot disagree" — but the table is not the hardware, and nothing ever applied
the lock in this run. `build_manifest.collect_t_b()` rejects artifacts measured
at a different clock (D26), and could not help either: it compares that same
stamp, so 1640 was checked against 1640 and passed.

Fixed at the source rather than by re-measuring alone: `clock_lock_state()`
reads the determinism setpoint back off every GPU via `amdsmi` `max_clk`,
`assert_clock_lock()` refuses to measure when the readback disagrees, and
`run_pipeline.sh` now applies the lock and verifies it before stage 1 instead
of printing a number it got from the table. Both the wrong-setpoint case and
the partial-lock case were planted and observed to fail.

## Scope

* `artifacts/06/authoritative/` — all 143, quarantined here, being re-measured.
* `artifacts/06/candidates/` — 232 also measured at 1900, **kept**. They only
  order variants within one problem on one GPU, so the clock largely cancels,
  and `--top-k 2 --within 0.25` was sized for exactly this class of selection
  noise.
* `artifacts/01/stability-gpu0.json` and `interference.json` — measured at
  08:50 and 08:51, so also at 1900 rather than 1650. See STATE.md D27 for why
  the interference verdict does not transfer.

Delete these once the re-measured set is complete and anchor-verified. They are
kept until then only so the two can be compared.
