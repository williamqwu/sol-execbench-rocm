# Task 09 — Scoring manifest and release

**Goal:** package the result so someone else can reproduce and extend it.

## Preconditions

- All prior tasks done or explicitly deferred with recorded reasoning.

## Steps

### 1. Freeze the scoring manifest

Scores are valid only *within* a manifest version. Emit
`artifacts/09/manifest-v1.json` containing, per problem/workload:

- `t_sol` (ms **and** cycles), `t_b`, tolerances
- F_LOCK, timing methodology, GPU model and count
- ROCm / driver / torch versions, repo git SHA
- problem-set version and any deferrals (e.g. 220 vs 235)

Any stack upgrade that moves `t_b` requires a **new manifest version**. Never
edit one in place.

### 2. Agent baseline sweep

Run a kernel-optimizing agent across the problem set — the analogue of
upstream's median-0.732 figure.

Report the score distribution, and reproduce the paper's headroom correlation on
AMD: SOL score vs *fraction of headroom reclaimed*, `(T_ref − T_k)/(T_ref − T_SOL)`.
Upstream found r = 0.981. A materially different correlation on AMD is a finding
about the port and needs explaining before release.

### 3. Submission-path hardening

Wire the LLM-judge static screen into submission intake, per task 08.

### 4. Documentation

- `README.md` — what this is, how to run, what differs from upstream
- **Methodology writeup** — how AMD SOL bounds were derived, F_LOCK and why,
  measured ceilings alongside analytic peaks, every deferral
- Backend coverage matrix (which of `hip_cpp` / `hipblaslt` / `miopen` / `ck` /
  `ck_tile` / `aiter` / `triton` have seed examples)

### 5. The cross-vendor framing — get this right

State plainly, in the README and the methodology writeup:

> An AMD SOL score and an NVIDIA SOL score are each **within-platform** measures
> of the fraction of hardware headroom reclaimed. They are comparable in spirit
> but are **not** a cross-vendor performance comparison, because analytic peaks
> are reachable to different degrees on different microarchitectures.

Publish the measured ceilings (task 00/01) next to the analytic peaks so readers
can see the achievable-fraction difference for themselves.

This will be the most misread number in the project. Say it clearly, early, and
more than once.

### 6. Upstream conversation

The vendor device layer was built to be upstreamable — upstream's
`SupportedHardware` enum already anticipates more than one target. Decide with
stakeholders: PR it to `nvidia/sol-execbench` behind a vendor flag, or maintain
`SOL-ExecBench-ROCm` as a sibling. Recommendation in `PLAN.md` §9 is to build for
upstreaming and decide based on receptiveness.

## Acceptance check

```bash
python scripts/verify_artifacts.py --task 09 --full
```

Passes when: manifest complete and self-consistent (every problem has t_sol,
t_b, tolerances, provenance); agent baseline run with distribution reported;
headroom correlation computed; all deferrals documented with counts consistent
across every artifact; cross-vendor caveat present in README and methodology.

## Guard rails

- **Do not publish scores from a manifest that fails the anchor property**
  (task 06 step 4).
- Do not quietly drop deferred problems from the count. If it is 220 and not
  235, say 220 everywhere — including in any comparison to upstream numbers.
- Do not market a cross-vendor leaderboard delta. See step 5.

## Outputs

- `artifacts/09/manifest-v1.json`
- `artifacts/09/agent-baseline.json`
- `README.md`, `docs/methodology.md`
