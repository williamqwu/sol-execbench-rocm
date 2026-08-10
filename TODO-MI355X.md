# TODO-MI355X — the runbook for an 8× MI355X node

This is what an agent landing on an 8× MI355X node executes, in order. It is
derived from what this repo contains, not from a plan: every script, flag,
artifact path and acceptance command below exists today on `master`.

**The port needs no work. Every number does.** MI355X and MI350X are the same
CDNA4 die (`gfx950`), so the harness, the tolerances machinery, the shim, the
exploit corpus and the manifest builder are all part-independent. What is not
transferable is anything with a millisecond or a megahertz in it.

**What is on `master` for MI355X today: nothing.** `artifacts/00/` and
`artifacts/01/` are the MI350X record, `manifest-v1` is MI350X at F_LOCK 1300
(as are v1.1 and v1.2, which the board serves), and `leaderboard/db/` holds
`solbench-MI350X.db` only. Read §1 before you
conclude that no MI355X measurement has ever been taken — that is true of
`master` and false of the repo.

---

## 0. Read first, in this order

| | why |
|---|---|
| `CLAUDE.md` | the eight prime directives. Directives 1, 2 and 3 are the ones this task will tempt you to break. |
| `STATE.md` | the ledger. Everything in it below the environment table is MI350X unless it says otherwise. |
| `TODO.md` | the open gaps. Most of them follow you to MI355X (§8). |
| `HANDOFF.md` | the *other* direction of this same move (MI355X → MI350X), and the source of every environment gotcha in §2. Superseded as a plan, still correct as a record. |
| `tasks/DEPENDENCIES.md` + `tasks/NN-*.md` | the specification of each acceptance check. Note that `tasks/01`, `02` and `03` are written *for MI355X* — they were authored in session 1 on this part, and `master` then executed them on MI350X. |

---

## 1. Before you measure anything: `origin/feat/agent-scoreboard`

An earlier session already did much of this on an MI355X node
(`mia1-p02-g10`) and the work is **unmerged**:

```bash
git rev-list --count master..origin/feat/agent-scoreboard   # 29 as of 2026-08-06
git show origin/feat/agent-scoreboard:STATE.md | less
```

`TODO.md` says 24 commits; the branch has moved since. It carries a full node
acceptance, a re-measured F_LOCK, a re-derived T_SOL, 223 of 235 T_b candidate
selections, an agent scoreboard, and twelve scripts that do not exist on
`master` (`env/solb-native`, `env/check_stack.py`, `scripts/run_pipeline.sh`,
`scripts/score_solutions.py`, `scripts/gpu_parity_check.py`,
`scripts/burst_clock_probe.py`, `scripts/unlocked_clock_probe.py`,
`scripts/backfill_scores.py`, and the agent-sweep drivers).

Its data is deliberately **not** on the leaderboard: its `T_b` was measured
while a 404-session agent sweep saturated the node's CPUs, it fails the anchor
property (13/204), and `S` was retracted rather than published (branch blocker
B2).

Treat that branch the way you would treat a colleague's lab notebook: it tells
you what to expect and what already went wrong, and **none of it is a
measurement you may adopt**. In particular its findings about determinism mode
(§3) are properties of *that node*, possibly of that node's firmware, and were
already revised three times within the branch itself (D18 → D27 → D29 → D30).
If you are on `mia1-p02-g10`, expect them. If you are on any other MI355X node,
re-measure and expect nothing.

Decide explicitly, and record the decision in `STATE.md`: merge the branch
first, or start from `master` and cherry-pick. Starting from `master` and
silently re-implementing `run_pipeline.sh` is the expensive option.

---

## 2. Node setup

`data/` is gitignored and does not travel with the repo. Three steps, then a
baseline check. All from `README.md` *Running it*, which is the maintained copy.

```bash
# 1. Build the pinned measurement container (ROCm 7.2 / torch 2.9.1 / SOLAR).
env/solb bash -lc 'python -c "import torch; print(torch.__version__)"'

# 2. Materialize the dataset. The Hub ships parquet, not the per-problem
#    layout; materialize_dataset.py is the exact inverse of the dataset's own
#    converter and round-trip-verifies all 235.
env/solb bash -lc '
  python -c "
from huggingface_hub import snapshot_download
snapshot_download(\"nvidia/SOL-ExecBench\", repo_type=\"dataset\",
                  local_dir=\"/var/tmp/solbench/data-hf\")"
  python scripts/materialize_dataset.py \
      --parquet-dir /var/tmp/solbench/data-hf/data \
      --out data/SOL-ExecBench/benchmark'

# 3. The external FlashInfer blobs. Without them 9 of the 26 FlashInfer
#    problems fail at run time as ordinary runtime errors, which reads as a
#    port defect and is not one.
env/solb bash -lc 'python scripts/fetch_flashinfer_traces.py'

# 4. Baseline.
env/solb bash -lc 'python -m pytest tests/ -q'      # expect 519 passed, 75 skipped
```

Environment gotchas, each of which cost a session somewhere:

* **Container GPU access needs the host's numeric video/render GIDs.**
  `--group-add render` resolves against the *container's* `/etc/group` and
  grants nothing. The symptom is that `torch.cuda.device_count()` returns 8
  while any real HIP context raises `No HIP GPUs are available`. `env/solb`
  already does this; do not "simplify" it.
* **The container mounts a generated `/etc/passwd`.** Without it
  `torch.compile` dies in `getpass.getuser()` with `uid not found`, breaking
  every submission that compiles, not just tests.
* **`FLASHINFER_TRACE_DIR=/work` must be set.** The eval driver resolves the
  trace paths against the *staging* directory, not the CWD, so having the blobs
  in the repo is not enough. `env/solb` sets it.
* **Anything large goes to `/var/tmp/solbench`**, mounted at the same absolute
  path inside and outside the container so a path in an artifact means the same
  thing in both. `/home` is NFS. A path outside `/work` and outside scratch does
  not exist inside the container, and a runner told to write there dies before
  writing anything (D17).
* **`env/solb` is unprivileged; `env/solb-root` is privileged and exists only
  for the clock lock**, because `/sys` is read-only in a stock container. Note
  the trap it documents: a stock container does not *fail* to set clocks —
  `rocm-smi --setperfdeterminism` exits 0 having done nothing.
* **`env/solb` recreates the container when the image ID changes**, unless a
  sweep is running inside it. If you rebuild the image mid-sweep, read the
  warning it prints rather than working around it.
* **No docker on the node?** The branch hit this and built `env/solb-native`
  plus `env/check_stack.py`, which reproduces `env/solb`'s environment contract
  and then *asserts* the pinned stack instead of assuming it. Take that path,
  not a hand-rolled venv. Check the interpreter version before anything else:
  that node ran Python 3.10, and all 33 Quant references import `StrEnum`
  (3.11+), so they fail before any submission is involved (branch blocker B1).
  The container image is `py3.12` and does not have this problem.

Then, and this is the step that is easy to skip:

```bash
# artifacts/00 and artifacts/01 hold the MI350X record. node_acceptance.sh
# overwrites artifacts/00/ in place. Move it aside, do not delete it -- the
# MI350X manifest cites it.
git mv artifacts/00 artifacts/00-MI350X && git mv artifacts/01 artifacts/01-MI350X
mkdir -p artifacts/00 artifacts/01/logs

env/solb bash -lc 'bash scripts/node_acceptance.sh'
env/solb bash -lc 'python scripts/roofline_probe.py --gpu 0 --out artifacts/00/roofline-gpu0.json'
env/solb bash -lc 'python scripts/verify_artifacts.py --task 00'
```

Expect 8× `gfx950`, 288 GiB, **1400 W** cap and a **2400 MHz** ceiling — if the
report says 1000 W / 2200 MHz you are on an MI350X node and this file does not
apply. `roofline_probe.py` looks the spec peak up per part
(`solexbench_rocm/parts.py`), so the achieved fraction is against 2517 TFLOPS
@2.4 GHz, not MI350X's 2307. These are default-clock reference points and are
**not** scoring ceilings; do not cite them downstream.

---

## 3. Task 01 first. F_LOCK is measured, never guessed

`tasks/01` is a hard blocker for 03, 05 and 06 and it is the most consequential
measurement in the project. Nothing you measure before it is trustworthy.

### The preset in the code is labelled, not trusted

`CLOCK_LOCK_PRESETS` (`src/sol_execbench/core/bench/config/device_config.py`)
contains:

```python
"AMD Instinct MI355X": ClockPreset(gpu_clk_mhz=1650, dram_clk_mhz=None),
```

That 1650 came from **session 1, on a different node, in a different session,
with a procedure that was later shown to be wrong in fact on the other part**.
Three specific reasons not to inherit it:

1. It has **no `achieved_gpu_clk_mhz`**, so `preset.f_lock_mhz` returns the
   *requested* 1650. Session 1's own note in that comment says it verified at
   1648 under load, and the unmerged branch chose 1640 and then 1650-at-1660.
   Three numbers, one field, none of them measured on your node.
2. **`--setperfdeterminism X` does not give you X.** On MI350X it yields
   ~0.83·X, rock-steadily (D8). Whether that ratio holds on the 1400 W part
   **was never asked on `master`**, and the branch's answer on one MI355X node
   was that it does *not* hold uniformly: below 1500 the ~0.83 rule appeared,
   at 1600–1700 the request was obeyed to 0.4%, and above 1800 the part pinned
   to the power cap. A discontinuity, not a curve. That is one node's finding
   and is not yours until you reproduce it.
3. Adding a preset because "it is the same architecture" is the same class of
   error as copying a B200 constant into an AMD artifact (prime directive 2).
   The number would look entirely plausible and nothing downstream could
   detect it.

### The sequence

```bash
# 1. Sustained floors, UNLOCKED, one GPU at a time so per-GPU variation is not
#    confounded with cross-GPU power coupling. >=3 GPUs; the acceptance check
#    counts floor-gpu*.json files.
for g in 0 1 2; do
  env/solb bash -lc "python scripts/clock_calibrate.py floor --gpu $g --minutes 15 \
      --out artifacts/01/floor-gpu$g.json"
done
# and the worst case the sweeps actually run in:
env/solb bash -lc 'python scripts/clock_calibrate.py floor --gpu 0 --minutes 15 \
    --load-siblings --out artifacts/01/floor-gpu0-busy.json'

# 2. Requested vs achieved. This is the step MI350X did not know it needed.
#    It applies a setpoint per step, so it needs the privileged wrapper -- and
#    it therefore writes its artifact as root into an NFS-mounted repo. Chown it
#    back, or the next unprivileged run cannot overwrite it.
env/solb-root rocm-smi -r                     # start from a known state
env/solb-root python scripts/clock_calibrate.py determinism-sweep --gpu 0 \
    --freqs 1100 1250 1350 1500 1600 1700 1800 1900 2000 2200 2400 \
    --out artifacts/01/determinism-sweep.json
env/solb-root rocm-smi -r                     # ... and RESET. See below.

# 3. Choose F_LOCK from the ACHIEVED column, apply, verify under load.
env/solb-root python scripts/clock_calibrate.py lock --freq-mhz <SETTING> --all-gpus
env/solb bash -lc 'python scripts/clock_calibrate.py verify --freq-mhz <ACHIEVED> \
    --gpu 0 --under-load'

# 4. Reproducibility at F_LOCK. Gate: CV < 2%.
env/solb bash -lc 'python scripts/clock_calibrate.py stability --gpu 0 --trials 30 \
    --out artifacts/01/stability-gpu0.json'

# 5. Sibling interference. Schedule-shaping; do not infer it (§6).
env/solb bash -lc 'python scripts/clock_calibrate.py interference --timing-gpu 0 \
    --load-gpus 1-7 --out artifacts/01/interference.json'
```

**Reset the setpoint after every sweep.** `determinism-sweep` applies a setpoint
per step and never resets. On the branch's node it left 1900 MHz in place for
eleven hours, and 143 authoritative T_b were measured at ~1860 MHz while stamped
1640 — the stamp came from the preset table, the manifest's clock guard compared
that stamp against the same table, and it agreed with itself (F24, branch D27).
`verify_artifacts --task 01` now reads `MAX_CLK` back off every GPU for exactly
this reason. **The table is not the hardware.**

**Sample every GPU, not three.** MI350X's eight GPUs spanned 65 MHz at one
setting; the branch's MI355X node spanned **326 MHz** at one setting, with six
of eight cards at ~0.80·request while drawing 400 W below their cap. If that
reproduces on your node, F_LOCK is a per-GPU quantity and §6 is not optional.

### Then land it in three places, or the acceptance check fails

1. **`CLOCK_LOCK_PRESETS`** — set `gpu_clk_mhz` to the *setting* and
   `achieved_gpu_clk_mhz` to the *measured* clock. Two fields because on AMD
   they differ; `f_lock_mhz` returns the achieved one. Replace the session-1
   comment with your own measurement, the way the MI350X entry does.
2. **`STATE.md`** — the canonical line `**F_LOCK = <n> MHz**` at the start of a
   line. `f_lock_from_state()` takes the **first** match in the file, so edit
   the existing marker rather than adding a second one, and keep prose about
   the other part's clock in a non-marker form (F17, F20).
3. **`solexbench_rocm/parts.py`** needs nothing — F_LOCK is not stored there,
   deliberately: it separates architectural constants (shared) from part
   constants (never shared) from measured ones (never guessed).

```bash
env/solb bash -lc 'python scripts/verify_artifacts.py --task 01'
```

If `STATE.md and CLOCK_LOCK_PRESETS agree on F_LOCK` fails, one of your two
records is lying about the frequency the whole benchmark is calibrated at, and
nothing downstream can tell which. If `every GPU is at the preset's determinism
setpoint` fails, something left the node somewhere else and every artifact you
take now will be stamped with a clock it was not measured at.

---

## 4. What transfers from MI350X, and what does not

Same discipline `HANDOFF.md` used going the other way. "Transfers" means you may
use it as-is; "confirm" means it is expected to hold and cheap to check.

| Result | Transfers? | Why |
|---|---|---|
| The harness port (`src/sol_execbench/`) | **Yes** | Vendor logic keys off `gfx950` / `torch.version.hip`, not the SKU |
| `src/solexbench_rocm/activity/` and the rocprofiler shim (task 04) | **Yes** | CPU-verified, mutation-tested, and the HSA clock domain is not part-specific |
| `--offload-arch=gfx950`, `-ffast-math`, `-lamdhip64` | **Yes** | Same ISA target, same toolchain |
| `LLC_BYTES[gfx950] = 256 MiB` → 512 MiB flush | **Yes, confirm** | Same die, same Infinity Cache. `roofline_probe.py --llc-sweep`; the cliff should land near 256 MiB |
| Dataset census (235: 94/82/33/26) | **Yes** | A property of the dataset |
| AMD-derived tolerances, `artifacts/05/` | **Yes, with a caveat** | Numerics are a property of the gfx950 ISA and the torch build, not of the clock. The branch inherited them rather than re-deriving. Cheap to falsify: run task 02's reference sweep against them and see whether anything fails |
| The exploit corpus and static source screen (task 08) | **Yes** | torch-level; re-run it, do not re-derive it |
| `artifacts/03`'s `macs` / `memory_bytes` / `precision` per workload | **Yes — this is the big saver, see below** | The SOLAR trace is frequency-independent and the die is identical |
| **F_LOCK, whatever value** | **NO** | §3 |
| `T_SOL` in **ms** (`t_sol_ms`) | **NO** | Contains F_LOCK |
| `T_SOL` in **cycles** (`t_sol_cycles`) | **NO, not in general** | See below — this one is widely misstated in this repo |
| `T_b`, every one of them | **NO** | A wall-clock time at MI350X's F_LOCK on MI350X silicon |
| `manifest-v1.json` | **NO** | Every `t_b` in it is at the wrong clock. It is an MI350X artifact and cannot score an MI355X measurement |
| Floors 1335–1390, achieved 1242–1307, CV 0.0034 | **NO** | 1000 W air-cooled part |
| Rooflines 4.53 TB/s / 1168 TFLOPS | **NO** | MI350X at default clocks, and against a different spec peak |
| Sibling interference −0.11% | **NO — re-measure** | §6 |
| Script fixes F1–F24 | **Yes** | Real bugs, fixed in code |

### The T_SOL shortcut, stated correctly

`CLAUDE.md` §6 and `gen_arch_yaml.py`'s docstring both say *"T_SOL in cycles is
invariant to F_LOCK — compute it once, convert to milliseconds by one
division."* **That is true only for compute-bound workloads.** The same
docstring says, ten lines earlier, that `DRAM_byte_per_cycle` is *derived* as
`bytes_per_sec / F`, and `sol_bounds.py` computes

```python
exact_cycles = max(macs / MAC_per_cycle,            # cycles: invariant in F
                   memory_bytes / DRAM_byte_per_cycle)   # cycles: proportional to F
```

so a memory-bound workload's bound is invariant in **milliseconds** and scales
with F in **cycles** — the opposite. In `artifacts/03/t_sol.json` today, 1163 of
2998 successful workloads (39%) are `"bottleneck": "memory"`. Rescaling their
cycle counts to MI355X's F_LOCK would inflate their bounds by the clock ratio,
and the bottleneck can flip as F moves, so neither column is safe on its own.

What *is* safely reusable is the pair of quantities the bound is computed from.
`macs` and `memory_bytes` are recorded per workload, come out of a `device=meta`
trace, and depend on neither the clock nor the part. So T_SOL at any F is one
`max()` away, with `MAC_per_cycle` from `parts.py` (architectural, shared, and
justified by reproducing *both* parts' published peak FLOPS) and
`dram_bytes_per_sec` unchanged at 8.0e12.

Two ways to spend that:

* **Cheap and honest:** re-run `sol_bounds.py` with `--part MI355X --freq-mhz
  <F_LOCK>`. It is CPU-only (`device="meta"`), runs 32-way, and needs no GPU —
  so launch it in the shadow of the sweeps and do not think about it again.
* **Cheaper:** recompute from the MI350X artifact arithmetically. The branch
  already built this (`t_sol_at()`, commit `494c9a8`, which emits both roofline
  terms instead of their max precisely because the max is not invertible).

Either way, **the 51 problems SOLAR failed on MI350X will fail again** — they
are trace failures, not clock failures — and the declared-traffic tier
(`sol_traffic_floor.py`) covers them, at bytes/bandwidth, which *is* invariant
in ms. It is also the tier D18 says is wrong on paged attention (§8).

---

## 5. The order of work

`tasks/DEPENDENCIES.md` is the graph. The shape that matters: **get the two long
sweeps launched early and do everything else while they run.** The node is the
constraint; a session that documents while eight cards idle has wasted the only
thing that cannot be recovered.

```
00 ──> 01 ──┬─> 03 (CPU-only, no GPU)
            ├─> 05 (long, sharded)      both need 02
            └─> 06 (long, sharded, then serial authoritative pass)
    └─> 02 ──┬─> 04 (re-run, not re-derive)
             ├─> 07
             └─> 08
09 needs all
```

Every sweep below is resumable by construction: `shard_sweep.py` skips a problem
whose output file already parses as JSON, so re-invoking the **exact same
command** continues where it stopped. Do not restart with different settings —
mixed-settings data is unusable (prime directive 7).

```bash
# --- 02: the reference sweep. All four categories; an omitted --category is
#     the realistic way scope silently shrinks.
env/solb bash -lc 'nohup python scripts/shard_sweep.py --task references \
    --category L1 L2 Quant FlashInfer-Bench --gpus 1-7 \
    --out artifacts/02/references/ > artifacts/02/logs/references.log 2>&1 &'
env/solb bash -lc 'python scripts/check_coverage.py --artifacts artifacts/02/references'
env/solb bash -lc 'python scripts/verify_artifacts.py --task 02'

# --- 05: tolerances. Launch and walk away; then triage from the artifact the
#     sweep wrote, never from scratch state while it is still running (D10).
env/solb bash -lc 'nohup python scripts/shard_sweep.py --task tolerances \
    --category L1 L2 Quant FlashInfer-Bench --gpus 1-7 \
    --out artifacts/05/ -- --seeds 10 --margin 1.25 --low-memory \
    > artifacts/05/logs/sweep.log 2>&1 &'
env/solb bash -lc 'python scripts/apply_tolerances.py'      # -> artifacts/05/workloads/, triage.md
env/solb bash -lc 'python scripts/check_coverage.py --artifacts artifacts/05/workloads \
    --pattern workload.jsonl'
env/solb bash -lc 'python scripts/verify_artifacts.py --task 05'

# --- 06: T_b. Two passes, and the second one is not shardable.
env/solb bash -lc 'nohup python scripts/shard_sweep.py --task tb-candidates \
    --gpus 1-7 --out artifacts/06/candidates/ \
    > artifacts/06/logs/candidates.log 2>&1 &'
env/solb bash -lc 'python scripts/check_coverage.py --artifacts artifacts/06/candidates'
# then, on a quiet node, one GPU, serially. --gpu sets HIP_VISIBLE_DEVICES for
# each child itself, so do not also set it outside.
env/solb bash -lc 'python scripts/authoritative_tb.py \
    --candidates artifacts/06/candidates --out artifacts/06/authoritative --gpu 0'
env/solb bash -lc 'python scripts/verify_artifacts.py --task 06'

# --- 03: no GPU. Run it while 05/06 are burning the cards.
env/solb bash -lc 'python src/solexbench_rocm/solar/gen_arch_yaml.py \
    --part MI355X --freq-ghz <F_LOCK/1000> -o SOLAR/configs/arch/MI355X.yaml'
env/solb bash -lc 'python scripts/sol_bounds.py --part MI355X --freq-mhz <F_LOCK> \
    --arch-yaml SOLAR/configs/arch/MI355X.yaml --out artifacts/03/t_sol.json \
    --jobs 32 --resume'
env/solb bash -lc 'python scripts/sol_traffic_floor.py \
    --arch SOLAR/configs/arch/MI355X.yaml --t-b artifacts/06/authoritative \
    --out artifacts/03/t_sol_traffic.json'
env/solb bash -lc 'python scripts/sol_cross_checks.py \
    --arch SOLAR/configs/arch/MI355X.yaml --t-b artifacts/06/authoritative'
env/solb bash -lc 'python scripts/check_coverage.py --artifacts artifacts/03/t_sol.json'
env/solb bash -lc 'python scripts/verify_artifacts.py --task 03'

# --- 04, 07, 08: re-run, do not re-derive. GPUs 1-7.
env/solb bash -lc 'python scripts/shard_sweep.py --task methodology-compare \
    --category L1 --gpus 1-7 --out artifacts/04/compare/'   # L1-only is deliberate
env/solb bash -lc 'python scripts/mxfp4_spike.py --out artifacts/07/spike.json'
env/solb bash -lc 'python -m pytest reference/exploits/ -q'
for t in 04 07 08; do env/solb bash -lc "python scripts/verify_artifacts.py --task $t"; done

# --- 09: the manifest. build_manifest refuses to build when F_LOCK cannot be
#     resolved, and rejects T_b artifacts stamped at any other clock (F18/F21).
env/solb bash -lc 'python scripts/build_manifest.py --out artifacts/09/manifest-v1.json'
env/solb bash -lc 'python scripts/verify_anchor.py --sample 20'
env/solb bash -lc 'python scripts/verify_artifacts.py --task 09 --full'
```

`check_coverage.py` after **every** sweep, not at the end. It names every
problem not accounted for and exits non-zero; a gap listed in
`artifacts/deferred.json` with a reason is a decision, a gap without one is a
bug.

---

## 6. GPU discipline

**GPU 0 for authoritative timing only. Everything else on 1–7.** Pin with
`HIP_VISIBLE_DEVICES` and record which GPU produced every timing artifact.

Two things about this part specifically:

* **Whether sibling load perturbs authoritative timing is UNMEASURED on the
  liquid-cooled part and must not be inherited.** MI350X measured −0.11% and
  concluded that sweeps and authoritative timing may share the node. That
  verdict is a measurement, not a rule. The branch re-measured +0.02% on its
  MI355X node with the standard 50-iteration burst probe — and then found, under
  a *sustained* sibling load, GPU 0 losing 15% of its clock. The probe's duty
  cycle kept it on the fast branch. So: run `clock_calibrate.py interference`,
  and if you then schedule sweeps beside timing runs, know that the burst probe
  is the thing you validated.
* **CPU contention is the resource that actually bit.** The branch's `T_b` was
  voided not by GPU interference but by running the authoritative pass on GPU 0
  while an agent sweep saturated 120 CPUs: `torch.compile` and Triton autotuning
  are CPU-bound, so a compile-heavy timing run beside compile-heavy agents
  measures the scheduler. Re-running the identical variant took 5× longer than
  its recorded anchor. Quiet means quiet on both sides of the PCIe bus.

`scripts/gpu_map.py` resolves torch indices to amdsmi/rocm-smi indices through
PCI identity, because the orderings differ and are scrambled. Passing a torch
index to `rocm-smi -d` sets one card and measures another — it has produced a
completely plausible fictional clock floor at least twice in this project
(D20, branch D29). Use it.

---

## 7. The leaderboard side

One database per part, because a score measured on MI350X and one measured on
MI355X are not comparable and the safest place to enforce that is the
filesystem — a query cannot accidentally join across two files. The contract is
`leaderboard/DESIGN-v2.md` §6.

```bash
leaderboard/.venv/bin/python leaderboard/ingest.py --part MI355X \
    --agent-runs <every external run root, every single time>
# -> leaderboard/db/solbench-MI355X.db
```

Three things to know before you run it:

* **`--part` asserts, it does not relabel.** `ingest.py` reads the part from the
  manifest's own provenance (`_provenance.part`, else the MI* device name) and
  names the database from it. A `--part` that disagrees is an error. So the
  database gets written the moment there is an MI355X manifest, and not before.
* **`ingest.py` reads a fixed manifest path** — `artifacts/09/manifest-v1.json`,
  a module constant with no flag. To produce `solbench-MI355X.db` the MI355X
  manifest has to *be* that file. The frozen MI350X manifest currently occupies
  it, and the unmerged branch sidestepped this by writing
  `manifest-MI355X-v1.json`, which `ingest.py` cannot read. Resolve it
  deliberately — a `--manifest` flag is the obvious answer and is not in the
  DESIGN-v2 contract, so raise it rather than inventing it.
* **Omitting `--agent-runs` silently deletes every run kept outside the repo**
  from the board (D24, introduced three separate times). The board still
  renders and still looks complete.

Until that database exists, the part switch lands on the honest empty state,
which is a first-class page that says nothing has been measured on MI355X, that
the port itself needs no work, and links here. That is the correct rendering and
it must not be faked.

`leaderboard/solbench.db` is a *view*. Never edit it; change the artifact and
re-ingest.

---

## 8. What will still be open when you are done

None of these are MI355X problems. They follow the port.

* **D18 — paged-attention `T_SOL` over-counts traffic**, 6 problems / 249
  scoreable workloads. The declared-traffic tier prices a paged KV cache at full
  allocation while the kernel gathers 34 pages of 989,669. It will be wrong on
  MI355X in exactly the same way and by the same factor, because the defect is
  in the traffic model, not the clock. The v1.1 fix is to derive paged traffic
  from `num_kv_indices × page_size`. Scores on those six problems are not usable
  and must be marked wherever they appear.
* **D21 — `L1__005` (beaten 1.09–1.15×) and `L1__035` (1.003–1.013×)**. Not
  paged; D18 does not explain them. `L1__005` is a compute-bound SOLAR roofline
  ~15% too slow, and a compute-bound bound moves with F_LOCK, so re-derive it on
  your clock before concluding anything about whether it reproduces.
* **The 15 NVFP4 Quant problems** are deferred with evidence
  (`artifacts/deferred.json`, `shipped_total: 220`). NVFP4 has no ROCm kernel
  path and an MXFP4 twin is a re-specification, not a translation. Re-run
  `mxfp4_spike.py` — the software path may have moved — but do not quietly
  change the count: 220 means 220 everywhere, including in any comparison to
  upstream.
* **D20 — matmul timing spread is bimodal and unexplained** on MI350X: 0.13% of
  iterations cost 3.9–4.5×. The clock hypothesis was tested and falsified;
  hipBLASLt kernel selection is the untested suspect. Two upstream tests are
  skipped behind it because their thresholds were measured on RTX 4090 / B200
  and no defensible AMD constant could be derived. Re-deriving them on MI355X
  (`scripts/derive_timing_variance.py`) is a second data point on an open
  question and is worth the twenty minutes.
* **`origin/feat/agent-scoreboard`, 29 commits, unmerged** — §1.
* **No full-benchmark agent baseline anywhere.** Upstream's median of 0.732 has
  no counterpart on either part. `docs/agent-baseline.md` prices it.
* **`scripts/verify_artifacts.py` has no test coverage**, and it is the
  acceptance gate for all ten tasks. Three separate checks in it have been found
  inert (F17, F22, F23). A bug in it does not fail loudly.

---

## 9. Effort

Estimates I can support, from artifact provenance timestamps on MI350X. These
are elapsed spans between the first and last artifact of a sweep on **that**
part, sharded 7-way, with other work interleaved — so they bound the wall clock
rather than measuring the GPU time, and MI355X is a faster part with a higher
power budget.

| stage | MI350X elapsed | how measured |
|---|---|---|
| task 02, reference sweep, 235 problems, GPUs 1–7 | **0.6 h** | first→last `_provenance.utc` in `artifacts/02/references/` |
| task 05, tolerance calibration, 235 problems, GPUs 1–7 | **5.3 h** | same, `artifacts/05/*.json` |
| task 06, candidate selection, 235 problems, GPUs 1–7 | **10.2 h** | same, `artifacts/06/candidates/` |
| task 06, authoritative re-time, 220 problems, one GPU, serial | **11.4 h** | same, `artifacts/06/authoritative/`; overlapped the candidate sweep, so this is elapsed and not exclusive |

Two independent cross-checks on the T_b figures: the branch reports 223 problems
of candidate selection in **107 min** across GPUs 1–7 on MI355X, and its
mis-clocked authoritative run produced 143 artifacts in roughly eleven hours.

Task 01 has no measured duration but its cost is fixed by the script's own
parameters: 15 min per floor run × ≥4 runs, plus (settle 30 s + 60 s) per
frequency in the determinism sweep, plus 30 stability trials in separate
processes. Budget half a day including the analysis, and do not compress the
floor runs — the p5 is taken over the *final* five minutes for a reason.

**What I cannot estimate, and will not guess:** task 03 (SOLAR's per-problem
timeout is 900 s and 51 of 235 problems hit failures on MI350X, so the total is
dominated by how many fail rather than by how many succeed); tasks 04, 07 and
08, none of which have a recorded wall clock; and anything involving an agent
sweep, whose cost is a budget decision rather than a duration.

---

## 10. Checklist

Each line is a real command. Paste its real output into `STATE.md`.

```bash
# setup
env/solb bash -lc 'python -m pytest tests/ -q'                       # 519 passed, 75 skipped
env/solb bash -lc 'python scripts/verify_artifacts.py --task 00'

# F_LOCK -- nothing below this line is trustworthy until it passes
env/solb bash -lc 'python scripts/verify_artifacts.py --task 01'

# the port, on this part
env/solb bash -lc 'python scripts/check_coverage.py --artifacts artifacts/02/references'
env/solb bash -lc 'python scripts/verify_artifacts.py --task 02'
env/solb bash -lc 'python scripts/verify_artifacts.py --task 04'

# bounds, tolerances, anchors
env/solb bash -lc 'python scripts/check_coverage.py --artifacts artifacts/03/t_sol.json'
env/solb bash -lc 'python scripts/verify_artifacts.py --task 03'
env/solb bash -lc 'python scripts/check_coverage.py --artifacts artifacts/05/workloads --pattern workload.jsonl'
env/solb bash -lc 'python scripts/verify_artifacts.py --task 05'
env/solb bash -lc 'python scripts/check_coverage.py --artifacts artifacts/06/candidates'
env/solb bash -lc 'python scripts/verify_artifacts.py --task 06'

# quant, red team, release
env/solb bash -lc 'python scripts/verify_artifacts.py --task 07'
env/solb bash -lc 'python -m pytest reference/exploits/ -q'
env/solb bash -lc 'python scripts/verify_artifacts.py --task 08'
env/solb bash -lc 'python scripts/verify_artifacts.py --task 09 --full'

# the board
leaderboard/.venv/bin/python leaderboard/ingest.py --part MI355X --agent-runs <roots>
leaderboard/.venv/bin/python -m pytest tests/leaderboard -q
```

Done means: `--task 09 --full` passes, `check_coverage` exits zero for every
sweep, every deferral is in `artifacts/deferred.json` with a reason and the same
count appears in every document, and `db/solbench-MI355X.db` exists so the part
switch lands on data. Anything short of that is recorded in `STATE.md` as a
blocker, not smoothed over.
