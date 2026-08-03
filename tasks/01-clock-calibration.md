# Task 01 — Clock calibration (F_LOCK)

**Goal:** determine the frequency at which this hardware will be benchmarked,
and establish that timings at that frequency are reproducible.

**This is the most consequential measurement in the project.** Every T_SOL and
every T_b is expressed at F_LOCK. Get it wrong and every score is wrong, in a
way that looks entirely plausible. It is a hard blocker for tasks 03, 05 and 06.

## Why a locked clock at all

Boost clocks vary 10–30% under thermal and power pressure, which swamps the
differences the benchmark exists to measure. Upstream locks B200 to 1500 MHz —
roughly 76% of its ~1.97 GHz boost — precisely so that a kernel measured on
Monday is comparable to one measured on Friday.

MI355X is a 1400 W part. Expect a meaningful derate. **Do not assume the B200
ratio transfers**, and do not guess a number: measure the floor this silicon
actually sustains.

## Preconditions

- Task 00 done, all 8 GPUs healthy.

## Steps

### 1. Find the sustained clock floor

```bash
python scripts/clock_calibrate.py floor --gpu 0 --minutes 15 \
    --out artifacts/01/floor-gpu0.json
```

Runs a sustained MFMA-saturating BF16 GEMM loop while sampling SCLK, power,
temperature and any throttle/PCC status once per second. Reports the p5 of SCLK
over the **final 5 minutes** — i.e. the floor after thermal steady state, not
the boost clock in the first 30 seconds.

Run it on at least three GPUs. If their floors differ by more than ~50 MHz, the
node has meaningful per-GPU variation and F_LOCK must be chosen for the worst,
not the best.

### 2. Choose F_LOCK

Round the observed floor **down** to a round number with margin (~50 MHz below
the lowest p5 across sampled GPUs). The point is a frequency the hardware holds
without ever needing to throttle, not the highest achievable number.

Record the reasoning in `STATE.md` under *Decisions taken*, including the
observed floors. A later session will want to know why this number and not
another.

### 3. Apply and verify the lock

```bash
python scripts/clock_calibrate.py lock --freq-mhz <F_LOCK> --all-gpus
python scripts/clock_calibrate.py verify --freq-mhz <F_LOCK> --under-load
```

Uses `amdsmi` where available, falling back to `rocm-smi --setperfdeterminism`
(AMD's documented mechanism: it caps the soft max clock so power-control events
cannot perturb attainable clocks; `rocm-smi -r` resets).

`verify --under-load` is the part that matters: an unloaded GPU will report the
requested clock whether or not the lock is doing anything. Verification must
hold *while a sustained load runs*.

### 4. Measure reproducibility at F_LOCK

```bash
python scripts/clock_calibrate.py stability --gpu 0 --trials 30 \
    --out artifacts/01/stability-gpu0.json
```

Times a fixed reference kernel 30 times in separate processes. Reports
coefficient of variation.

**Gate: CV < 2%.** Above that, timing noise will swamp real optimization
differences and there is no point proceeding. If it fails, investigate before
continuing — likely causes are the lock not actually holding, thermal
saturation, or another process on the GPU.

### 5. Sibling-GPU interference — the schedule-shaping experiment

```bash
python scripts/clock_calibrate.py interference --timing-gpu 0 --load-gpus 1-7 \
    --out artifacts/01/interference.json
```

Times the same reference kernel on GPU 0 under two conditions: siblings idle,
and siblings running sustained load. At 8×1400 W, power and thermal coupling
across the node is plausible and nobody should assume either way.

Interpretation:

| Δ median time | Meaning |
|---|---|
| < 1% | Sweeps and authoritative timing can share the node. |
| 1–3% | Usable, but authoritative runs (task 06 final pass) need a quiet node. |
| > 3% | Serious. Every timing run needs an otherwise-idle node; task 05/06 sharding still fine for *correctness* work but final timings serialize. Re-plan the schedule and note it in `STATE.md`. |

This result changes how the remaining days are scheduled, so do not skip it and
do not infer it from a single sample.

## Acceptance check

```bash
python scripts/verify_artifacts.py --task 01
```

Passes when: F_LOCK is recorded in `STATE.md`; floor data exists for ≥3 GPUs;
lock verified *under load*; stability CV < 2%; interference quantified with a
stated scheduling consequence.

## Guard rails

- **Never guess F_LOCK.** Not from the spec sheet, not from the B200 ratio, not
  from MI300X's documented 1900 MHz. Those are starting hypotheses; the
  measurement decides.
- **Do not unlock clocks** for the remainder of the project without recording it.
  A sweep that spans an unlock/relock boundary is not internally comparable.
- Do not raise F_LOCK later to make results look better. If it must change,
  everything measured at the old value is invalidated and must be re-run — say
  so explicitly rather than mixing.
- If `amdsmi` and `rocm-smi` disagree about the current clock, trust neither;
  find out why first.

## Outputs

- `artifacts/01/floor-gpu{0,1,2}.json`
- `artifacts/01/stability-gpu0.json`
- `artifacts/01/interference.json`
- `STATE.md`: F_LOCK, interference verdict, decision reasoning
