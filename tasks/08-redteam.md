# Task 08 — Red team the anti-reward-hacking layer

**Goal:** confirm the exploit detectors still fire on ROCm, and find AMD-specific
exploits upstream never had to consider.

This layer is load-bearing: **14.5% of agent submissions were flagged on
NVIDIA.** A benchmark whose detectors silently stopped working would look
healthy right up until its leaderboard became meaningless.

The port work here is mostly *testing*, not coding — every mechanism in
`reward_hack.py` is torch-level and carries over.

## Preconditions

- Task 02 done.

## Steps

### 1. Replay the three known exploit families

From the paper. Corpus is pre-written in `reference/exploits/` as portable
pytest cases (authored without hardware — they are torch-level).

| Family | Technique | Expected detector |
|---|---|---|
| **Concurrency** | work on unrecorded threads/streams, `torch.jit.fork` | thread-count check; stream policy |
| **State caching** | reuse outputs keyed on data address; lazy eval via FakeTensor | shifting allocator; strict `type(t) is torch.Tensor` |
| **Environment** | monkey-patch timing fns; precision downgrade | function-identity snapshots |

Every one must be detected. A miss is a release blocker.

### 2. AMD-specific additions

- **HIP streams** however exposed — `torch.cuda.Stream` on ROCm, and raw
  `hipStreamCreate` in C++ submissions. Scan submission sources for the latter
  the way the CUDA path does for its equivalents.
- **smi from inside a submission.** A submission that raises the clock cap
  mid-run beats the locked-clock calibration entirely. Block
  `amd-smi`/`rocm-smi`/`amdsmi` invocation from submission subprocesses, and
  scope passwordless sudo to the container entrypoint only. (The NVIDIA image
  has the same latent issue with sudo'd `nvidia-smi`.)
- **XCD partitioning tricks.** MI355X exposes compute-partition modes with no
  NVIDIA analogue. Confirm a submission cannot change partitioning to gain an
  advantage, and that the harness records the partition mode in every trace.
- **LDS-resident state across iterations.** Verify the shifting allocator's
  pointer randomization defeats address-keyed caching on CDNA4 as it does on
  Blackwell.

### 3. Interim guard while on HIP events

Until task 04 lands, the activity-sequence count assertion — one of the
anti-hack layers — is absent. Compensate: assert the current stream is default,
assert thread count unchanged, and run the LLM-judge static screen on all
submissions. Record that the interim posture was in force for anything measured
before task 04.

### 4. Two documented residual gaps

Assert these as tests so they stay visible rather than becoming folklore. Both
are inherited from upstream, not introduced by the port:

- Work interleaved *between* measured kernels **is** counted, even under an
  unrecognized name. Renaming does not hide cost. (Already tested.)
- Work starting *after* the final measured kernel escapes the span. Mitigated by
  the post-synchronize host end-stamp plus thread/stream checks, not eliminated.

## Acceptance check

```bash
pytest reference/exploits/ -q
python scripts/verify_artifacts.py --task 08
```

Passes when: **100% of the replayed corpus is detected**; AMD-specific probes
each have a recorded verdict; smi-from-submission is blocked and demonstrated
blocked; partition mode appears in traces; residual gaps present as passing
tests.

## Guard rails

- **A detector that fires on a legitimate submission is also a bug.** Check
  false positives against the task 02 reference sweep — every reference must
  pass clean.
- Do not weaken a detector to unblock a submission. Escalate instead.
- If you find a new exploit, add it to the corpus **before** fixing it, so the
  fix is demonstrably effective and stays that way.

## Outputs

- `artifacts/08/replay-results.json`
- `artifacts/08/amd-specific.md` — verdicts per probe
- New exploit cases added to `reference/exploits/`
