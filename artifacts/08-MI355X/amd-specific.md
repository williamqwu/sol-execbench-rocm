# Task 08 (MI355X) — AMD-specific red-team verdicts

Replay corpus: `reference/exploits/` — **48 cases, 48 pass**
(`artifacts/08-MI355X/replay-results.json`, produced by
`scripts/replay_exploits.py` from the JUnit report of that run, not
transcribed by hand). Run on **`mia1-p02-g05`, GPU 0, 8× MI355X**, through the
real harness: packager, staging directory, eval-driver subprocess.
Provenance is stamped on the JSON: ROCm 7.2.0, amdgpu 6.16.6, torch
2.9.1+rocm7.2.0, git `9b7e8435-dirty`, 2026-08-14.

**Clock basis: unlocked.** Unlike MI350X, this part has no F_LOCK — 
`get_clock_preset("AMD Instinct MI355X").f_lock_mhz` is `None` by design
(STATE.md, *Decisions taken*), so `f_lock_mhz` is `null` in this artifact's
provenance and that is correct, not a missing stamp. Every case here was
replayed at stock clocks.

This is the MI355X counterpart of `artifacts/08/amd-specific.md`. It is not a
copy of it: every verdict below comes from the run above. The corpus itself has
grown from 28 cases to 48 since the MI350X run, so the two case counts are not
comparable and neither number was carried across.

A passing case means one of two things, and the corpus says which for each:

* **detected** — the submission was flagged `REWARD_HACK`, or rejected before
  staging by the static source screen.
* **neutralized** — the submission ran, produced the right answer, and gained
  no speed advantage over the honest implementation.

Per family, on this part:

| family | cases | passed |
|---|---|---|
| `test_source_screen` | 22 | 22 |
| `test_environment` | 13 | 13 |
| `test_concurrency` | 8 | 8 |
| `test_state_caching` | 5 | 5 |

---

## AMD-specific probes

### 1. smi invocation from a submission — **BLOCKED, demonstrated on MI355X**

A submission that changes the clock policy mid-run corrupts every timing taken
against it, and leaves numbers that stay perfectly self-consistent. On this
part the payoff is different from MI350X but not smaller: there is no F_LOCK to
beat, so the target is the *unlocked* basis itself — a submission that lifts or
pins clocks moves the very quantity the clock bracket is trying to observe.

Three spellings, all rejected by `static_source_screen()` before anything was
compiled or imported (`test_smi_from_submission_is_blocked`, 3/3 pass):

| invocation | outcome |
|---|---|
| `rocm-smi` via `subprocess` | rejected, static source screen |
| `amd-smi` by absolute path, through `shell=True` | rejected, static source screen |
| `os.system(...)` | rejected, static source screen |

The absolute-path case is why PATH sanitation alone is not the defense.

Two layers, unchanged from the MI350X posture and both exercised here:

* `static_source_screen()` — refuses the submission before staging. This is the
  layer that catches a **compiled HIP** submission, which runs underneath every
  Python-level guard.
* `install_smi_guard()` — wraps `subprocess.Popen/run/call/check_*`,
  `os.system/popen` and the `exec*`/`spawn*` family before user code is
  imported, so a Python submission that builds the command at run time is still
  stopped.

The C++/HIP side of the same screen is covered by
`test_hazard_is_rejected`, 7/7 pass here: `hipStreamCreate`,
`cudaStreamCreateWithFlags`, `system("rocm-smi ...")`,
`system("amd-smi set ... --power-cap 1400")`, a write to
`/sys/class/drm/card0/device/power_dpm_force_performance_level`, a `sudo`
invocation, and `hipDeviceSetCacheConfig`. `test_clean_hip_source_passes`
confirms the screen is not simply rejecting HIP.

### 2. XCD compute-partition manipulation — **BLOCKED, and recorded**

No NVIDIA analogue. Switching SPX→CPX changes how many CUs a kernel can reach,
so a submission that repartitions is not running the same benchmark.

* Blocked by the same two layers as smi — it is an `amd-smi set` invocation
  (`test_compute_partition_change_is_blocked`, passes).
* Recorded on every trace as `Environment.libs["compute_partition"]`, read from
  `/sys/class/drm/card*/device/current_compute_partition`, so a change made
  outside the submission — by another user or by the operator — is visible after
  the fact rather than silently mixed into the results.

Measured on this node: `compute_partition_mode()` returns **`SPX`**, and
`amd-smi partition` reports **SPX / NPS1 on all 8 GPUs**. Both readings taken on
`mia1-p02-g05` at the time of the replay.

### 3. HIP streams — **BLOCKED at two levels**

| form | outcome |
|---|---|
| `torch.cuda.Stream()` left current on return | detected, `REWARD_HACK` (`check_default_stream`) |
| raw `hipStreamCreate` in a C++/HIP submission | rejected, static source screen |
| side stream host-synced inside the call | neutralized, no advantage (`test_side_stream_host_sync_no_advantage`) |
| side stream joined on the *next* call | neutralized (`test_side_stream_join_deferred_to_next_call`) |
| side stream properly restored before return | **documented residual gap** — see below |

`check_default_stream` is an **interim guard** and its interim status is the
point. While timing is event-based, an event pair recorded on the default stream
does not observe a kernel running on another stream, so that work is free. The
check compensates by rejecting a submission that leaves a non-default stream
current.

Task 04's `rocprof` methodology is what restores the stronger, activity-based
assertion. On this part that path is now demonstrated to work: the shim builds
and `artifacts/04-MI355X/clock-domain-verification.log` verifies the clock
domain on real captures. Anything measured under `hip_events` was still measured
under the weaker posture, which is why `Environment.methodology` is on every
trace.

### 4. LDS-resident state across iterations — **NEUTRALIZED**

A submission cannot hold LDS across kernel launches, so the portable form of
this exploit is a persistent device-side buffer reused between iterations. The
shifting allocator defeats it: the cached answer belongs to a previous
iteration's inputs, and correctness rejects it.
`test_lds_resident_state_across_iterations` passes here, as do the four other
`test_state_caching` cases (`data_ptr`-keyed cache, shape-keyed cache, fake
tensor output, lazy proxy output).

---

## Documented residual gaps

Both are inherited from upstream, not introduced by this port, and both are
asserted as *passing* tests so they stay visible rather than becoming folklore.
Both reproduce on MI355X — they are properties of the harness, not of the part.

### Gap 1 — a side stream that is politely restored

`check_default_stream` inspects the stream that is *current on return*. A
submission that restores the default stream before returning is not caught by
that check; it is caught, if at all, by the synchronize-and-compare that
follows. `test_side_stream_restored` records the actual behaviour rather than a
wish, and is the regression anchor for an activity-sequence assertion.

### Gap 2 — threads created during warmup

`check_thread_injection` compares `threading.active_count()` before and after the
user call. A submission whose threads are created on its **first** call has them
created during warmup, so by the time a timed call is sampled the counts already
match and the check does not fire.

Found by `test_lingering_thread_pool`, which is why that case exists. It does no
harm today — `.result()` blocks, so the work stays inside the timed region — but
the evasion is real and only an activity-count assertion closes it.

---

## False positives — the other half of the guard rail

A detector that fires on a legitimate submission is also a bug.

* `test_every_dataset_reference_passes_the_screen` runs the static source screen
  over **all 235 problems' references** and asserts it stays silent. It does, on
  this part.
* The MI355X task-02 reference sweep (`artifacts/02-MI355X/references-amd/`,
  235 problems, **3957 workload instances**) ran every reference through the full
  harness with all defenses live. **No workload was flagged `REWARD_HACK`** —
  this is the assertion `verify_artifacts.py --task 08 --part MI355X` re-checks
  on every run, so it cannot go stale.

---

## What is NOT claimed here

* No MI355X-specific *new* exploit family was found or added. The corpus is the
  same 48 cases the port ships; what this artifact establishes is that all 48
  defenses fire on this part, not that the search for MI355X-only attacks is
  finished.
* The interaction between a clock-manipulating submission and the **unlocked
  clock basis** is blocked at the source screen, so it was never observed
  end-to-end. Whether the per-window clock bracket would *itself* have caught a
  clock change that got past the screen is untested, and is a task 03 question,
  not a task 08 result.
