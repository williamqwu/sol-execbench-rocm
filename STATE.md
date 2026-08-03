# STATE.md — progress ledger

> **Work has moved to an MI350X node. Read `HANDOFF.md` first.**
> Everything below was measured on `mia1-p02-g10`, an 8× **MI355X** node.
> The harness port transfers; **F_LOCK, the floors, the rooflines, the
> stability CV and the interference verdict do not** — MI350X is the
> air-cooled, lower-power part. `tasks/01` must be re-run there.

**Single source of truth for progress.** Update as you go, not at the end.
A session can be interrupted at any point; whatever is written here is what the
next session inherits.

Rules: record real output, not summaries of intent. If something failed, say so
and say how. Never mark a task `done` without pasting its acceptance-check
output.

---

## Environment

| Field | Value |
|---|---|
| Node | `mia1-p02-g10` |
| GPUs | 8× AMD Instinct MI355X, `gfx950:sramecc+:xnack-`, 288 GiB, 256 CUs each |
| ROCm version | **7.2.0** (in measurement container) / 7.1.1 (host `/opt/rocm`) |
| Driver (amdgpu) | 6.16.6 |
| torch version + build | `2.9.1+rocm7.2.0.git7e1940d4`, HIP `7.2.26015-fc0010cf6a` |
| F_LOCK (MHz) | **1650** — measured, applied to all 8 GPUs, verified under load |
| Sibling-GPU interference | **negligible (−0.19%)** — sweeps and authoritative timing may share the node |
| Dataset present | yes — 235 problems, L1=94 L2=82 Quant=33 FlashInfer-Bench=26 |
| Repo git SHA at start | `f97f6c6942f7d9ac938e9aa3041cb735c6936def` |
| Measurement container | `solbench:rocm7.2-torch2.9.1`, built from `env/Dockerfile` |

All measurement runs go through `env/solb`, which starts/reuses that container.
Host has no passwordless sudo; the container runs as root but with `/sys`
read-only (see blocker B1).

---

## Task status

| ID | Task | Status | Artifacts | Notes |
|---|---|---|---|---|
| 00 | Node acceptance | `done` | `artifacts/00/` | 13 checks, 0 failed |
| 01 | Clock calibration (F_LOCK) | `done` | `artifacts/01/` | **F_LOCK = 1650 MHz**; unblocks 03, 05, 06 |
| 02 | Harness port validation | `in-progress` | `src/sol_execbench/` | port written; reference sweep not yet run |
| 03 | SOL bounds (T_SOL) | `not-started` | | needs 01 |
| 04 | rocprofiler shim | `not-started` | | parallel with 05/06 |
| 05 | Tolerance calibration | `not-started` | | needs 01, 02. Long sweep. |
| 06 | Baselines (T_b) | `not-started` | | needs 01, 02. Long sweep. |
| 07 | Quant / MXFP4 | `not-started` | | highest uncertainty |
| 08 | Red team | `not-started` | | needs 02 |
| 09 | Release | `not-started` | | needs all |

Status vocabulary: `not-started` · `in-progress` · `blocked` · `done` · `deferred`

### Task 00 acceptance output (2026-08-03)

```
Acceptance check — task 00

  [PASS              ] node-report.json exists
  [PASS              ] node report has provenance
  [PASS              ] 8 GPUs present                         found 8
  [PASS              ] all GPUs are gfx950
  [PASS              ] power caps probed on every GPU         8/8
  [PASS              ] power caps uniform (±5%)               all 1400.0
  [PASS              ] max GFX clocks probed on every GPU     8/8
  [PASS              ] max GFX clocks uniform (±5%)           all 2400
  [PASS              ] idle temperatures probed on every GPU  8/8
  [PASS              ] idle temperatures uniform (±25%)       all 37
  [PASS              ] HBM roofline measured
  [PASS              ] BF16 GEMM roofline measured
  [REQUIRES-JUDGEMENT] dataset layout matches audit           confirm categories L1=94 L2=82 Quant=33 FlashInfer=26

  13 checks, 0 failed, 1 require human judgement
```

Judgement item resolved: category counts verified exactly (94/82/33/26 = 235)
against the real files, not the paper. Layout differs from the audit — see
deviation D1.

Node is uniform on every axis checked: 1400 W cap, 2400 MHz max GFX clock,
288 GiB, 256 CUs on all eight. Idle 32–42 °C, idle power 239–247 W.

### Task 01 results (2026-08-03)

**Sustained clock floors** (15 min saturating BF16 GEMM, p5 of the final
5 minutes, one GPU at a time so per-GPU variation is not confounded with
cross-GPU power coupling):

| Run | p5 | p50 | min | peak power | peak junction |
|---|---|---|---|---|---|
| GPU 0, siblings idle | **1725** | 1728 | 1723 | 1402 W | 60 °C |
| GPU 1, siblings idle | 1734 | 1738 | 1732 | 1402 W | 62 °C |
| GPU 2, siblings idle | 1757 | 1761 | 1754 | 1392 W | 59 °C |
| GPU 0, **all 7 siblings loaded** | 1728 | 1731 | 1726 | 1401 W | 63 °C |

Per-GPU spread 32 MHz — under the 50 MHz threshold at which F_LOCK would have
had to be chosen for a meaningfully worse GPU.

All runs sit at ~1400 W against a 1400 W cap while junction stays ≤63 °C, far
from any thermal limit: **MI355X is power-limited, not thermally limited**,
under this load. That is the mechanism that makes the derate real and makes a
spec-sheet clock useless here.

**Lock applied and verified:**
```
locking GPU 0 -> 1650 MHz
  8/8 card(s) at perf_determinism
locked

expected 1650 MHz, observed median 1648.0 MHz (drift 2.0)
PASS
```
The under-load form is the one that counts: unlocked, this GPU ran 1725–1830 MHz
under the same load.

**Stability at F_LOCK:** `CV = 0.0015` over 30 trials in separate processes
(gate 0.02). Timing noise is ~13× below the gate.

**Sibling interference:** quiet 0.1158 ms → busy 0.1156 ms, **−0.19%**, with all
seven siblings drawing 947–1275 W. Verdict `negligible`.

### Task 01 acceptance output (2026-08-03)

```
Acceptance check — task 01

  [PASS              ] F_LOCK recorded in STATE.md                       1650 MHz
  [PASS              ] clock floor sampled on >=3 GPUs                   4 GPUs
  [PASS              ] F_LOCK at or below lowest observed floor          F_LOCK 1650 <= min p5 1725
  [PASS              ] stability measured
  [PASS              ] timing CV < 2%                                    CV=0.0015
  [PASS              ] sibling interference measured
  [PASS              ] interference has a stated scheduling consequence  negligible

  7 checks, 0 failed, 0 require human judgement
```

### Task 02 status (port written, sweep not run)

Upstream v1.0.2 (`a9fa080`) vendored at `src/sol_execbench/`, structured as a
fork for upstreaming per PLAN.md. The AMD delta is 6 modified files plus a new
`core/bench/device/` package; every change carries an `# AMD:` marker.

Ported: vendor device layer (`detect_vendor`, LLC table, arch flags, flag
defaults) · LLC-sized cache flush · persisting-L2 reset as a vendor no-op ·
lazy CUPTI imports · `hip_events` methodology, defaulted on ROCm and recorded
per trace · `SupportedLanguages` + `SupportedHardware` AMD members ·
`--offload-arch=gfx950` injection · `.hip` sources and CK include dir ·
AMD clock-lock/verify/unlock path.

**Upstream's own test suite on this node: 463 passed, 75 skipped, 0 failed.**
The 75 skips are NVIDIA-only by construction. Tests asserting NVIDIA
*behaviour* are NOT skipped — `tests/conftest.py` pins `detect_vendor()` to
`nvidia` for them, so the NVIDIA path remains a live regression reference
rather than quietly rotting (tasks/02 guard rail).

The pytorch and triton example solutions run end-to-end on real GPUs.

**Not done: the 235-problem reference sweep**, which is the actual acceptance
criterion, and `scripts/runners/run_reference.py`, which it needs.

**Clocks were locked at 1650 MHz on all 8 GPUs, then RESET to `auto` at end of
session** when work moved to a different node. Recording the unlock as the
guard rails require. Nothing had been measured at F_LOCK beyond task 01 itself,
so no sweep spans the unlock boundary and nothing is invalidated. The reset was
done because this node is shared (D2) and leaving a node-wide 1650 MHz cap in
place would have quietly degraded another user's work for no benefit.

Verified after reset: all 8 cards report `auto`.

**Task 00 rooflines at DEFAULT clocks** (reference points only — per the task's
guard rails these are NOT scoring ceilings and must not be cited downstream):
HBM 4.87 TB/s (61% of 8.0 spec), BF16 GEMM 1433 TFLOPS (57% of 2500 spec).

---

## Blockers

### B1 — cannot apply the clock lock: no privileged access to `/sys` [task 01, opened 2026-08-03] — **RESOLVED 2026-08-03**

**Resolution:** the user authorised running the clock-setting step as root in a
privileged container. `env/solb-root` does exactly that and nothing else —
measurement still runs unprivileged as the invoking user through `env/solb`.
Verified reversible on a single GPU before applying node-wide: set → card57
reports `perf_determinism`, `rocm-smi -r -d 5` → back to `auto`. Then applied
to all 8 and confirmed 8/8.

Original report follows, because the silent-failure mode it documents is a
permanent hazard rather than a one-off.

---


**What was attempted:** `rocm-smi --setperfdeterminism 1700`, both on the host
as the login user and inside the measurement container as root.

**What happened (real output):**

Host:
```
sudo: a terminal is required to read the password; either use the -S option to
read from standard input or configure an askpass helper
sudo: a password is required
```

Container (exit status **0**, no error text, and *no effect*):
```
============================ ROCm System Management Interface ============================
================================== End of ROCm SMI Log ===================================
```
```
touch: cannot touch '/sys/class/drm/card1/device/power_dpm_force_performance_level':
Read-only file system
... power_dpm_force_performance_level: auto      (unchanged, all 8 cards)
```

**Why this is not something to work around:** the container form *looks like it
worked* — exit 0, no error. `verify` on an unloaded GPU would then report the
requested clock and appear to pass, because an idle GPU sits below any cap.
Every subsequent measurement would be taken at an unlocked boost clock while
the artifacts claim F_LOCK. `scripts/clock_calibrate.py` has been hardened to
detect exactly this (see F4) and now refuses to report success unless a card
actually reports `perf_determinism`.

**What would unblock it:** either passwordless sudo on the host, or agreement
to run the clock-setting step in a `--privileged` container (docker group
membership makes this technically possible without sudo). **Not done yet
because this node is shared** — see deviation D2. Locking clocks changes global
GPU state for the other user currently on the node.

**Not blocked by B1:** task 01 step 1 (floor measurement) needs no lock and is
running. Tasks 02 and 04 do not depend on F_LOCK.

---

## Surprises and deviations

### D1 — dataset ships as parquet, not per-problem directories
`reference/upstream-audit.md` expects `definition.json` + `workload.jsonl` +
`reference.py` per problem. The Hub actually publishes four files —
`data/{L1,L2,Quant,FlashInfer-Bench}.parquet`, one row per problem — because
that is what the dataset viewer requires. The dataset repo carries its own
`scripts/convert_to_parquet.py` doing the forward direction.

Resolution: `scripts/materialize_dataset.py` is its exact inverse, field for
field, writing the canonical layout to `data/SOL-ExecBench/benchmark/`. It
re-derives every parquet row from the files it wrote and refuses to exit 0 on
any mismatch. Round-trip verified for all 235.

Dataset is **public and ungated** — `huggingface-cli`/`snapshot_download`
needed no token. Total 5.2 MB expanded. The `hf_id`, `axes`, `inputs`,
`outputs`, `reference`, `custom_inputs_entrypoint`, `workloads` fields are all
present as the audit describes.

Also noted for task 05: upstream tolerances are stored per workload as
`{"max_atol": ..., "max_rtol": ...}` — **not** the `[atol, rtol, matched_ratio]`
triple that `verify_artifacts.check_05` assumes when detecting copied B200
constants. That comparison will need adapting when
`reference/b200-tolerances.json` is built.

### D2 — this node is NOT exclusively ours
`CLAUDE.md` §4 says "All eight GPUs are yours". In fact user `jinpan12@amd.com`
is logged in (tmux, 3 panes) and has a container `flydev` running the same ROCm
PyTorch image. At the time of the task-00 measurements `rocm-smi --showpids`
reported **no KFD processes**, so the node was genuinely idle and task 00 is
clean.

Consequences, which are not optional:
* Any node-wide clock lock affects another engineer's work (blocker B1).
* Timing runs can be contaminated at any moment by a job that is not ours.
  `clock_calibrate.py floor` now samples **sibling GPU power every second** and
  flags any sibling drawing >400 W during the window the floor is derived from,
  so contamination is detectable rather than silently averaged in.
* The task-01 interference experiment measures *our own* induced load; it
  cannot control for the other user starting work mid-run. Re-check
  `siblings_busy_during_tail` in every floor artifact before trusting it.

### D5 — 9 FlashInfer-Bench problems need a *second*, separate dataset
Found by running upstream's own e2e test, which failed with
`Failed to load safetensors`. 9 of the 26 FlashInfer-Bench problems declare
inputs of `type: "safetensors"` pointing at `data/flashinfer-trace/blob/...` —
a different HuggingFace dataset (`flashinfer-ai/flashinfer-trace`, public)
that `nvidia/SOL-ExecBench` does not carry and does not mention.

Across all 235 problems the input types are: 18521 `random`, 11680 `custom`,
1643 `scalar`, **714 `safetensors`** — the last resolving to 304 unique blobs,
all under FlashInfer-Bench. L1, L2 and Quant are entirely self-contained.

This is a scope trap of exactly the kind CLAUDE.md §0 warns about: without that
data those 9 problems fail at run time, and 9/235 would drop out looking like
ordinary runtime errors rather than a missing dependency.
`scripts/fetch_flashinfer_traces.py` downloads just the referenced blobs
(39 MB) rather than the whole trace dataset.

Second half of the trap: the eval driver resolves those relative paths against
the **staging directory**, not the CWD, so having the blobs in the repo is not
sufficient. `FLASHINFER_TRACE_DIR` must point at the directory containing
`data/flashinfer-trace/`; `env/solb` now sets it to `/work`. With both pieces
in place, upstream's `gqa_paged_decode` sample passes **2/2 workloads on
MI355X** — a real FlashInfer-Bench problem running end to end on AMD.

### D4 — another user's `docker` prune deleted our image mid-session
The tagged measurement image `solbench:rocm7.2-torch2.9.1` disappeared while a
15-minute measurement was running. The running container survived (its image
layers stayed alive, untagged), so no data was lost, but a fresh
`docker run` failed with "pull access denied ... repository does not exist".
Almost certainly a `docker image prune -a` by the other user on this shared
node (D2).

`env/solb` already rebuilds the image when it is missing, so this self-heals;
noted because the failure message points at a registry problem and not at the
actual cause, which would waste a later session's time.

### D3 — MI355X is power-limited, not clock-limited, under BF16 GEMM
Single GPU under sustained 8192³ BF16 GEMM: **1387 W of a 1400 W cap**, SCLK
~1836 MHz against a 2400 MHz ceiling, junction 57 °C. The part runs into its
power budget well before its clock ceiling, which is the mechanism that makes a
measured F_LOCK necessary rather than a spec-sheet number.

---

## Fixes to scripts on first contact

Every script below had never run on hardware. Recording each fix so a later
session can tell whether a measurement predates it.

**F1 — `build_node_report.py`: `power_limit` is in microwatts.**
amdsmi returns `1400000000`; the field was written straight into `power_cap_w`.
Now divided by 1e6. Any node report produced before this fix has a power cap
1e6 too large.

**F2 — `build_node_report.py` / `clock_calibrate.py`: the `EDGE` temperature
sensor does not exist on MI355X.** `amdsmi_get_temp_metric(..., EDGE, ...)`
raises `AMDSMI_STATUS_NOT_SUPPORTED`. `HOTSPOT` (junction — what rocm-smi
prints) works. This was the more damaging of the two: in `clock_calibrate.read_clocks`
the temperature read sat inside the *same* try block as the clock read, so an
unsupported sensor discarded the SCLK sample too. Uncaught, task 01 would have
collected **zero usable clock samples** and reported `steady_state: null`.

**F3 — amdsmi handle order ≠ torch device order.** Measured on this node:

| torch idx | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| amdsmi idx | 3 | 0 | 2 | 1 | 7 | 4 | 6 | 5 |

Both `build_node_report.smi_fields` and `clock_calibrate.read_clocks` indexed
amdsmi handles with a torch device index. `read_clocks(0)` therefore sampled a
*different physical GPU* than the one under load — it would have reported an
idle GPU's clock as the sustained floor: a low, stable, entirely plausible
number that is fiction, and that nothing downstream could detect. This is the
single most dangerous bug found so far.

New `scripts/gpu_map.py` resolves the mapping through PCI bus identity (which
both libraries agree on) instead of list position, and stays correct under
`HIP_VISIBLE_DEVICES`. Verified empirically: with load on torch device 5 only,
`read_clocks(5)` reported 1836 MHz / 1387 W / 57 °C while all seven siblings
read ~240 W idle.

**F4 — `clock_calibrate.set_perf_determinism` treated a silent no-op as
success.** See blocker B1. It now reads `power_dpm_force_performance_level` for
every card before and after, and fails unless some card actually reports
`perf_determinism`.

**F5 — `clock_calibrate.cmd_interference` drove sibling load with Python
threads.** The GIL serialises the launch loop, so "seven loaded siblings" would
in practice be intermittently idle and the measured interference would
*understate* reality — the dangerous direction, since this number decides
whether authoritative timing may share the node. Now uses subprocesses, waits
30 s for thermal steady state, records sibling power actually observed under
load, and reports if any load process died.

**F6 — `node_acceptance.sh` dataset census used `-maxdepth 3/4`** and no `-L`.
The real layout is one level deeper (`data/SOL-ExecBench/benchmark/<cat>/<problem>/`).
It counted 0 problems while the dataset was present and correct.

**F7 — `materialize_dataset.py` (new) wrote invalid JSON.** 72 of 94 L1
problems have no `custom_inputs_entrypoint`; parquet renders that as `NaN` and
`json.dumps` emitted a bare `nan` token, which only Python's lenient parser
accepts. Now normalised to `null`. All 235 `definition.json` re-checked under a
strict parser.

**F8 — `verify_artifacts.check_00` skipped its own uniformity check when the
probe failed.** `if caps:` meant that unprobed power caps (which is what
happened before F1/F3, since `amdsmi` was not installed) silently passed the
task. Now every field must be probed on all 8 GPUs *and* be uniform, and
max-GFX-clock and idle-temperature uniformity are checked too — task 00 step 2
asks for all three, and only power cap was ever tested.

**F9 — `clock_calibrate.py` aborted at teardown, turning a PASS into exit 134.**
`cmd_floor` and `cmd_verify` both started the load in a daemon thread and then
let the interpreter shut down while it was still inside a HIP call:
`terminate called without an active exception`, exit 134 — *after* the artifact
had been written correctly. All three initial floor runs did this. A good
result behind a non-zero exit status is worse than a clean failure, because
`shard_sweep.py` would score it as a dead worker and re-run it. Both paths now
signal the load thread to stop and join before exiting.

**F10 — the amdsmi handle-order bug also existed in `device/amd.py`'s clock
readback** and is fixed there the same way as F3. Recorded separately because
the ported harness has its own copy of the lookup and would otherwise
reintroduce the fault.

**F11 — upstream's flush-buffer sizing is 64× too small on CDNA4.** Not a bug
in our scripts but the single most consequential porting finding so far, and it
is measured rather than assumed: on MI355X `torch.cuda.get_device_properties()`
reports `L2_cache_size = 4 MiB` (the per-XCD L2). Upstream sizes its cold-cache
flush buffer at `2 × L2_cache_size` = **8 MiB**, against a **256 MiB** Infinity
Cache. Every "cold" iteration would in fact have run warm out of MALL, making
memory-bound kernels look dramatically faster than they are — with no symptom
anywhere in the output. `device/amd.py` now carries an explicit `LLC_BYTES`
table and *raises* on an unknown arch rather than falling back to device
properties.

---

## Decisions taken

**Measurement environment is a pinned container** (`env/Dockerfile`,
`env/solb`). The host has ROCm 7.1.1 and Python 3.10; the repo targets 3.12 and
the only available torch is in the ROCm 7.2 image. Rather than install anything
on the host, all measurement runs go through
`rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1`, plus three
additions that do not touch the pinned torch/ROCm: `amdsmi` (from the image's
own `/opt/rocm/share/amd_smi`), `huggingface_hub`, `pyarrow`. Container ROCm
(7.2.0) differs from host ROCm (7.1.1); the driver (6.16.6) is shared and is
what actually matters for the GPU. Recorded rather than "fixed" — prime
directive 6.

**Container gets `--hostname $(hostname)` and `git safe.directory`.** Without
the first, provenance records a random container ID instead of the node;
without the second, git refuses the bind-mounted repo and `git_sha` is silently
`null` in every artifact — which fails the provenance requirement.

**`data/` is a real directory in the repo, not a symlink to `/var/tmp`.** The
expanded dataset is only 5.2 MB. A symlink broke `find` in `node_acceptance.sh`
(needs `-L`) and would be a recurring footgun in every downstream script.
`/var/tmp/solbench` still holds the genuinely large things: the HF cache, the
raw parquet download, and sweep scratch.

**Floors measured sequentially, one GPU at a time, siblings idle.** Running
GPUs 0/1/2 concurrently would confound per-GPU variation with cross-GPU power
coupling, and per-GPU variation is precisely what step 1 exists to detect.

**A fourth, worst-case floor was added: GPU 0 with all seven siblings loaded.**
Not in the task's step list. The reasoning: tasks 05/06 shard across GPUs 1–7,
so for most of the project the node is fully loaded, and the guard rails forbid
raising F_LOCK later (doing so invalidates everything measured before it). A
floor measured only on a quiet node is the best case, and picking F_LOCK from
the best case is the one direction that cannot be corrected cheaply. This is an
*addition* of data under the existing methodology, not a change to how anything
is measured. It cost 15 minutes and it turned the interference question from an
inference into a measurement.

**F_LOCK = 1650 MHz.** Lowest p5 observed anywhere is 1725 MHz (GPU 0, quiet).
1650 is the round number at least 50 MHz below it, giving 75 MHz of margin — so
the cap sits below the floor even on the worst GPU sampled, in the worst node
condition, and the hardware never has to throttle to hold it. Verified at
1648 MHz median under sustained load.

Deliberately *not* chosen: 1700 MHz, which is round but only 25 MHz below the
lowest floor — less margin than the task specifies, and this node has another
user on it whose load is outside our control. No NVIDIA number entered this
decision; the B200 ratio (1500/1970 ≈ 76%) would have suggested ~1830 MHz,
which is **above** the measured floor and would have throttled continuously.
The AMD derate is milder: 1650/2400 ≈ 69% of the clock ceiling.

**Interference verdict: sweeps and authoritative timing may share the node.**
Measured −0.19% at step 5, corroborated independently by the busy-node floor
(1728 MHz loaded vs 1725 MHz quiet). Both say the same thing. The CLAUDE.md §4
working rule ("GPU 0 authoritative only, idle otherwise") can therefore be
relaxed — task 01 was explicitly the experiment that decides this. GPU 0 is
still recorded per artifact, and the shared-node caveat in D2 is unaffected:
this measures *our own* induced load, not another user's.

---

## Session log

```
### 2026-08-03 — session 1  (node: mia1-p02-g10, 8x MI355X)
Worked: task 00 (done), task 01 (done), task 02 (port written, sweep not run)
Produced: artifacts/00/ — node report, rooflines, acceptance log
          artifacts/01/ — 4 clock floors, stability, interference
          F_LOCK = 1650 MHz, measured and verified under load
          data/SOL-ExecBench/benchmark/ — all 235 problems (gitignored)
          data/flashinfer-trace/ — 304 external blobs (gitignored)
          src/sol_execbench/ — vendored upstream v1.0.2 + AMD port
          env/{Dockerfile,solb,solb-root}
          scripts/{gpu_map,materialize_dataset,fetch_flashinfer_traces}.py
          11 first-contact fixes (F1-F11); F2, F3 and F11 were each capable of
          producing plausible, undetectably wrong numbers
Ended because: work moved to a different node (MI350X)
Next session should: read HANDOFF.md. Then re-run task 01 on the new node --
          F_LOCK 1650 is an MI355X number and must not be carried over.
```
