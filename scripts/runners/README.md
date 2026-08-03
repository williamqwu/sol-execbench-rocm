# Sweep runners

`shard_sweep.py` dispatches per-problem work to a runner here. Each runner takes
`--problem <dir> --out <file.json>` and writes one JSON result.

Implement each as part of the task that needs it:

| Runner | Task |
|---|---|
| `run_reference.py` | 02 — run a problem's own reference as the solution |
| `calibrate_tolerance.py` | 05 — multi-seed probing, 1.25x margin |
| `time_tb_candidates.py` | 06 — time each pre-authored T_b variant |
| `compare_methodology.py` | 04 — hip_events vs rocprof divergence |

Contract every runner must honour: **on failure, still write an output file**
recording the error. `shard_sweep.py` treats a missing file as "not yet done"
and will redo it; a failure recorded as an artifact is a result, a failure lost
to a crashed process is a silent gap. See prime directive 1.
