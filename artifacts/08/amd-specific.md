# Task 08 — AMD-specific red-team verdicts

Replay corpus: `reference/exploits/` — **28 cases, 28 pass**
(`artifacts/08/replay-results.json`). Run on MI350X, GPU 0, at F_LOCK, through
the real harness: packager, staging directory, eval-driver subprocess.

A passing case means one of two things, and the corpus says which for each:

* **detected** — the submission was flagged `REWARD_HACK`, or rejected before
  staging by the static source screen.
* **neutralized** — the submission ran, produced the right answer, and gained
  no speed advantage over the honest implementation.

The distinction matters. Five cases were initially asserted as *detected* and
are in fact *neutralized*; asserting the wrong one would have made the corpus
fail the moment a defense improved, and would have described a defense posture
this port does not have.

---

## AMD-specific probes

### 1. smi invocation from a submission — **BLOCKED, demonstrated**

The highest-payoff AMD exploit and the one with the least visible trace: a
submission that raises the clock cap mid-run beats the locked-clock
calibration that every T_SOL and T_b is expressed at, and the resulting
numbers stay perfectly self-consistent.

Three spellings tested, all blocked:

| invocation | outcome |
|---|---|
| `subprocess.run(["rocm-smi", "--setperfdeterminism", "2200"])` | rejected, static source screen |
| `subprocess.run("/opt/rocm/bin/amd-smi set -g 0 -l 2200", shell=True)` | rejected, static source screen |
| `os.system("rocm-smi --resetclocks")` | rejected, static source screen |

The absolute-path case is why PATH sanitation alone is not the defense.

Two layers, deliberately:

* `static_source_screen()` — refuses the submission before anything is
  compiled or imported. This is the layer that catches a **compiled HIP**
  submission, which runs underneath every Python-level guard.
* `install_smi_guard()` — wraps `subprocess.Popen/run/call/check_*`,
  `os.system/popen`, and the `exec*`/`spawn*` family before user code is
  imported, so a Python submission that constructs the command at run time is
  still stopped.

### 2. XCD compute-partition manipulation — **BLOCKED, and recorded**

No NVIDIA analogue. Switching SPX→CPX changes how many CUs a kernel can
reach, so a submission that repartitions is not running the same benchmark.

* Blocked by the same two layers as smi (it is an `amd-smi set` invocation).
* Additionally **recorded on every trace** as
  `Environment.libs["compute_partition"]`, so a change made outside the
  submission — by another user, or by the operator — is visible after the fact
  rather than silently mixed into the results. This node reports `SPX`
  throughout.

### 3. HIP streams — **BLOCKED at two levels**

| form | outcome |
|---|---|
| `torch.cuda.Stream()` left current on return | detected, `REWARD_HACK` (`check_default_stream`) |
| raw `hipStreamCreate` in a C++/HIP submission | rejected, static source screen |
| side stream properly restored before return | **documented residual gap** — see below |

`check_default_stream` is an **interim guard**, and its interim status is the
point. While timing is event-based, an event pair recorded on the default
stream simply does not observe a kernel running on another stream, so the work
is free. The check compensates by rejecting a submission that leaves a
non-default stream current.

It is weaker than upstream's CUPTI activity-sequence assertion, which counts
what the GPU actually did. Task 04's `rocprof` methodology restores that
capability; anything measured before it is switched on was measured under the
weaker posture, which is why `Environment.methodology` is recorded on every
trace.

### 4. LDS-resident state across iterations — **NEUTRALIZED**

A submission cannot hold LDS across kernel launches, so the portable form of
this exploit is a persistent device-side buffer reused between iterations. The
shifting allocator defeats it exactly as it defeats address-keyed caching on
Blackwell: the cached answer belongs to a previous iteration's inputs, and
correctness rejects it.

---

## Documented residual gaps

Both are inherited from upstream, not introduced by this port, and both are
asserted as passing tests so they stay visible rather than becoming folklore.

### Gap 1 — a side stream that is politely restored

`check_default_stream` inspects the stream that is *current on return*. A
submission that restores the default stream before returning is not caught by
that check; it is caught, if at all, by the synchronize-and-compare that
follows. `test_side_stream_restored` records the actual behaviour rather than
a wish, and is the regression anchor for task 04's activity-sequence
assertion.

### Gap 2 — threads created during warmup

`check_thread_injection` compares `threading.active_count()` before and after
the user call. A submission whose threads are created on its **first** call
has them created during warmup, so by the time a timed call is sampled the
counts already match and the check does not fire.

Found by `test_lingering_thread_pool`, which is why that case exists. It does
no harm today — `.result()` blocks, so the work stays inside the timed region
— but the evasion is real and only an activity-count assertion closes it.

---

## False positives — the other half of the guard rail

A detector that fires on a legitimate submission is also a bug.

* `test_every_dataset_reference_passes_the_screen` runs the static source
  screen over **all 235 problems' references** and asserts it stays silent.
  It does.
* The task 02 reference sweep ran every one of those 235 references through
  the full harness with all defenses live. No reference was flagged
  `REWARD_HACK`. The 16 non-passing problems failed on numerics or on an
  unsupported NVFP4 path, not on a detector.
