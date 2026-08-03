# Task dependency graph

```
00 node acceptance
 │
 ├─> 01 clock calibration (F_LOCK)  ◄── HARD BLOCKER for 03, 05, 06
 │    │
 │    ├─> 03 SOL bounds ......... mostly CPU; needs only F_LOCK
 │    ├─> 05 tolerance calib .... long sharded sweep, needs 02
 │    └─> 06 baselines T_b ...... long sharded sweep, needs 02
 │
 └─> 02 harness port validation
      ├─> 04 rocprof shim ....... independent once 02 lands
      ├─> 07 quant / MXFP4 ...... independent once 02 lands
      └─> 08 red team ........... independent once 02 lands

09 release ....................... needs all
```

## Scheduling this well

The node is the constraint, so the shape that matters is: **get the two long
sweeps launched early and do everything else while they run.**

```
Day 1     00, then 01. Nothing measured before 01 is trustworthy.
Day 2     02. As soon as it passes, launch 05 and 06 sharded across GPUs 1-7.
Day 2-4   While the sweeps run: 03 (CPU-bound), 04, 07.
Day 4-5   08, then 09.
```

Tasks 05 and 06 are the only ones whose wall-clock is dominated by GPU time
rather than by your work. Everything else is engineering that happens to need a
GPU nearby. Do not sit and watch a sweep.

## Parallelism notes

- **05 and 06 shard cleanly** across GPUs via `scripts/shard_sweep.py`. Both are
  embarrassingly parallel over problems.
- **03 barely needs the GPU at all** — SOLAR runs on meta tensors. Its only GPU
  dependency is the sanity check that T_SOL ≤ best measured time, which can wait
  for 06's output.
- **04 and 07 need a GPU but not an idle node** — put them on GPUs 1-7.
- **Authoritative timing runs are the exception.** Whether they can coexist with
  sweeps on sibling GPUs is exactly what task 01's interference experiment
  answers. If interference is measurable, 06's final authoritative pass needs an
  otherwise-quiet node and should be scheduled accordingly — that is a
  schedule-shaping result, so run task 01 properly before planning around it.
