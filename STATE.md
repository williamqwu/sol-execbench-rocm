# STATE.md — progress ledger

**Single source of truth for progress.** Update as you go, not at the end.
A session can be interrupted at any point; whatever is written here is what the
next session inherits.

Rules: record real output, not summaries of intent. If something failed, say so
and say how. Never mark a task `done` without pasting its acceptance-check
output.

---

## Environment

Fill this in during task 00. Everything downstream cites it.

| Field | Value |
|---|---|
| Node | _(hostname)_ |
| GPUs | _(expect 8× MI355X / gfx950)_ |
| ROCm version | _TBD_ |
| Driver (amdgpu) | _TBD_ |
| torch version + build | _TBD_ |
| F_LOCK (MHz) | **_TBD — task 01. Blocks 03, 05, 06._** |
| Sibling-GPU interference | _TBD — task 01_ |
| Dataset present | _TBD_ |
| Repo git SHA at start | _TBD_ |

---

## Task status

| ID | Task | Status | Artifacts | Notes |
|---|---|---|---|---|
| 00 | Node acceptance | `not-started` | | |
| 01 | Clock calibration (F_LOCK) | `not-started` | | **blocks 03, 05, 06** |
| 02 | Harness port validation | `not-started` | | |
| 03 | SOL bounds (T_SOL) | `not-started` | | needs 01 |
| 04 | rocprofiler shim | `not-started` | | parallel with 05/06 |
| 05 | Tolerance calibration | `not-started` | | needs 01, 02. Long sweep. |
| 06 | Baselines (T_b) | `not-started` | | needs 01, 02. Long sweep. |
| 07 | Quant / MXFP4 | `not-started` | | highest uncertainty |
| 08 | Red team | `not-started` | | needs 02 |
| 09 | Release | `not-started` | | needs all |

Status vocabulary: `not-started` · `in-progress` · `blocked` · `done` · `deferred`

---

## Blockers

_None recorded yet._

Format:
```
### B1 — <one-line summary>   [task NN, opened <date>]
What was attempted:
What happened (real output):
Why this is not something to work around:
What would unblock it:
```

---

## Surprises and deviations

Anything that differed from `PLAN.md`. This section is how the plan's
assumptions get corrected; it is not a confessional. Wrong assumptions found
early are the most valuable output of the first days on the node.

_None recorded yet._

---

## Decisions taken

Record any judgement call a later session would otherwise have to re-litigate,
with the reasoning.

_None recorded yet._

---

## Session log

Append one short entry per session. Keep it factual.

```
### <UTC date> — session N
Worked: task NN
Produced: artifacts/...
Ended because: (finished / blocked / out of time)
Next session should: <specific first action>
```
