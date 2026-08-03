# Task 02 — Harness port validation

**Goal:** the ported harness runs real problems on MI355X and emits valid traces.
This is the gate that unblocks 04, 05, 06, 07, 08.

## Preconditions

- Task 00 done. Task 01 done (you need the lock in place to get stable numbers).
- Dataset present under `data/`.

## Scope

Port upstream `sol-execbench` per `reference/upstream-audit.md`. The audit lists
every NVIDIA-specific call site; work through it. The pieces, in dependency
order:

### 1. Vendor device layer

Create `core/bench/device/{__init__,nvidia,amd}.py` with
`detect_vendor()` returning `"amd"` when `torch.version.hip is not None`.

Extend `CLOCK_LOCK_PRESETS` with the MI355X entry at your measured F_LOCK, and
add an explicit **`LLC_BYTES` table**. Do not derive the flush size from
`torch.cuda.get_device_properties(...).L2_cache_size` on ROCm — what that
reports is not the structure that matters here.

Keep the NVIDIA path working. It is the regression reference: when an AMD result
looks wrong, being able to run the same code on NVIDIA tells you whether the
refactor broke something or AMD genuinely differs.

### 2. Clock locking

Port `clock_lock.py` from `nvidia-smi -lgc/-lmc` to the `amdsmi` path built in
task 01. Reuse `scripts/clock_calibrate.py`'s implementation rather than writing
a second one.

HBM clock: Instinct parts do not expose independent memory-clock locking the way
`-lmc` does. Verify the memory clock is stable at max under load and *record*
it; do not attempt to lock it.

### 3. Cache hygiene

Two changes in `timing.py`:

- Flush buffer sized from `LLC_BYTES` (2× the 256 MB Infinity Cache = **512 MB**),
  not 2× reported L2.
- `cudaCtxResetPersistingL2Cache` → vendor no-op. CDNA has no L2-persistence API.

Sanity-check the sizing empirically: sweep the flush-buffer size and confirm the
measured bandwidth cliff lands near LLC capacity. If it does not, the LLC number
or the flush mechanism is wrong — find out which before proceeding.

### 4. Build path

`torch.utils.cpp_extension` auto-hipifies CUDA sources on ROCm, so the mechanism
carries over, but make the AMD path explicit:

- Accept `.hip` alongside `.cu`/`.cpp` in the source glob, so submissions can be
  written natively without a hipify pass mangling them.
- Flag translation: `-gencode arch=…,code=sm_100a` → `--offload-arch=gfx950`;
  `--use_fast_math` → `-ffast-math`; `-lcuda` → `-lamdhip64`.
- Add `hip_cpp` to `SupportedLanguages` and `MI355X` to `SupportedHardware`.

### 5. Timing path

Wire the CPU-verified `src/solexbench_rocm/activity/` package in behind the
`hip_events` methodology. **Ship on HIP events for now** — the rocprofiler shim
is task 04. Record `methodology` in every trace so the two are never confused
later.

## Steps

1. Port items 1–5 above.
2. `pytest tests/ -q` — must stay green throughout.
3. Reference sweep, sharded across GPUs 1–7:
   ```bash
   # ALL FOUR CATEGORIES — all 235 problems. Omitting a category here means
   # its problems are never validated and silently vanish from the benchmark.
   python scripts/shard_sweep.py --task references \
       --category L1 L2 Quant FlashInfer-Bench \
       --gpus 1-7 --out artifacts/02/references/
   ```
   Runs each problem's own PyTorch reference as the solution. Every problem
   should pass correctness against itself; failures are ROCm op-coverage gaps or
   input-generator breakage, not optimization problems.
4. Triage failures. Expect near-zero on mainstream L1/L2 ops at torch 2.9 /
   ROCm 7.2; budget for a handful. Record each one in `STATE.md` with the actual
   error, not a paraphrase.

## Acceptance check

```bash
python scripts/verify_artifacts.py --task 02
```

Passes when: `pytest tests/` green; **all 235 problems accounted for across all
four categories** (the checker enforces this per-category — an omitted
`--category` is the realistic way scope silently shrinks); ≥95% of references
pass correctness;
every failure individually recorded with its error; traces contain provenance
and a `methodology` field; the flush-size bandwidth-cliff check is recorded.

## Guard rails

- **Keep the NVIDIA path working.** Losing it costs you the ability to
  distinguish "I broke it" from "AMD differs" for the rest of the project.
- Do not paper over a failing reference by loosening tolerances. Tolerances are
  task 05 and are derived, not chosen to make things pass. A reference that
  fails against itself is a real bug — find it.
- Do not silently skip problems. A skipped problem must appear in `STATE.md`.

## Outputs

- Ported harness
- `artifacts/02/references/` — per-problem pass/fail with errors
- `artifacts/02/flush-sweep.json`
- `STATE.md`: failure triage
