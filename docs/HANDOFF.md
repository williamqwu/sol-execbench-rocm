# HANDOFF — moving to an MI350X node

> **Superseded — this migration is complete.** Kept as the record of *why*
> `artifacts/00/` and `artifacts/01/` were regenerated rather than reused, which
> is not visible from the artifacts themselves. For current state read
> [`STATE.md`](../STATE.md); for what is still open read [`TODO.md`](../TODO.md).
>
> What actually happened, against the predictions below: **F_LOCK on MI350X
> measured 1300 MHz**, not the 1650 carried from MI355X — a 21% difference, and
> the reason §1 exists. Sibling-GPU interference re-measured at **−0.11%**, so
> the "sweeps may share the node" verdict did hold on the air-cooled part, but
> only because it was re-measured. Everything in the "NO" rows below was
> regenerated.

Written 2026-08-03 at the end of session 1, which ran on **`mia1-p02-g10`, an
8× MI355X node**. Work continued on a **different node with MI350X**
(`gbt350-odcdh1-a08-1`).

It says which of session 1's results transfer to MI350X and which are void.

---

## 1. The thing most likely to go wrong

**`F_LOCK = 1650 MHz` is an MI355X number. Do not use it on MI350X.**

It was derived from sustained-load clock floors measured on a **liquid-cooled,
1400 W** MI355X, which sat pinned at ~1400 W and 59–63 °C for the whole
measurement — power-limited, with thermal headroom to spare. **MI350X is the
air-cooled part with a materially lower power budget.** Same CDNA4 die, same
`gfx950` ISA target, same 256 MiB Infinity Cache — but the sustained clock
floor is a property of the part *and its chassis*, not of the architecture.

> **No longer true, and this is the sentence people grep to.** The entry was
> added in **2cdb7b0** once task 01 had measured the clock:
> `ClockPreset(gpu_clk_mhz=1600, dram_clk_mhz=None, achieved_gpu_clk_mhz=1300)`.
> `--task 01` passes, 11 checks, 0 failed. The paragraph below is preserved
> because the *reasoning* — refuse rather than guess — is what earned the 1300,
> and is what an MI355X session must repeat.

`CLOCK_LOCK_PRESETS` therefore has **no MI350X entry**, deliberately. With no
entry, `lock_clocks()` logs `No GPU clock preset` and returns `False` unless
`SOL_EXECBENCH_GPU_CLK_MHZ` is set explicitly. That is a loud stop, and it is
intended. Adding `MI350X: 1650` because "it's the same architecture" is the
same class of error as copying a B200 constant into an AMD artifact
(prime directive 2): the number would look entirely plausible, nothing
downstream could detect it, and every T_SOL and T_b built on it would be wrong.

**So: re-run `tasks/01` on the new node first.** It is a hard blocker for
tasks 03, 05 and 06, exactly as before.

## 2. What transfers, and what does not

| Result | Transfers to MI350X? | Why |
|---|---|---|
| The harness port (`src/sol_execbench/`) | **Yes** | Vendor logic keys off `gfx950` / `torch.version.hip`, not the SKU |
| `LLC_BYTES[gfx950] = 256 MiB` → 512 MiB flush | **Yes**, but confirm | Same die, same Infinity Cache. Confirm with `roofline_probe.py --llc-sweep`; the cliff should land near 256 MiB |
| `--offload-arch=gfx950`, `-ffast-math`, `-lamdhip64` | **Yes** | Same ISA target and toolchain |
| GPU index mapping (`scripts/gpu_map.py`) | **Recompute** | It is resolved at run time from PCI identity, so it is automatically correct — but the *specific* permutation in the docstring is this node's |
| **F_LOCK = 1650 MHz** | **NO** | §1 |
| Floors 1725 / 1734 / 1757 MHz | **NO** | 1400 W part |
| Interference −0.19% | **NO — re-measure** | Measured at 8×1400 W liquid-cooled. An air-cooled chassis has *less* thermal margin, so coupling could plausibly be worse. Do not inherit the "sweeps may share the node" verdict |
| Rooflines 4.87 TB/s / 1433 TFLOPS | **NO** | MI355X at default clocks |
| Stability CV 0.0015 | **NO — re-measure** | Property of the locked clock, which changes |
| Dataset census (235: 94/82/33/26) | **Yes** | Property of the dataset |
| Script fixes F1–F11 | **Yes** | Real bugs, fixed in code |

Concretely: **delete or ignore `artifacts/00/` and `artifacts/01/` and
regenerate them.** They are committed as the MI355X record, not as inputs.
Nothing downstream has consumed them yet, so there is nothing to invalidate.

## 3. Setting up the new node

`data/` is gitignored (44 MB) and does **not** travel with this repo. Three
steps, in order:

```bash
# 1. Build the measurement container (also verifies torch is the ROCm build)
env/solb bash -lc 'python -c "import torch; print(torch.__version__)"'

# 2. Materialize the dataset. The Hub ships parquet, NOT the per-problem
#    layout the audit describes; this is the exact inverse of the dataset's
#    own converter and round-trip-verifies all 235.
env/solb bash -lc '
  python -c "
from huggingface_hub import snapshot_download
snapshot_download(\"nvidia/SOL-ExecBench\", repo_type=\"dataset\",
                  local_dir=\"/var/tmp/solbench/data-hf\")"
  python scripts/materialize_dataset.py \
      --parquet-dir /var/tmp/solbench/data-hf/data \
      --out data/SOL-ExecBench/benchmark'

# 3. Fetch the external FlashInfer blobs — WITHOUT these, 9 of the 26
#    FlashInfer-Bench problems fail at run time and 9/235 silently vanish.
env/solb bash -lc 'python scripts/fetch_flashinfer_traces.py'
```

Then confirm the baseline is sane before measuring anything:

```bash
env/solb bash -lc 'python -m pytest tests/ -q'   # expect 463 passed, 75 skipped
env/solb bash -lc 'bash scripts/node_acceptance.sh'
env/solb bash -lc 'python scripts/verify_artifacts.py --task 00'
```

Environment notes that cost time to discover:

- **`env/solb`** runs unprivileged as the invoking user. **`env/solb-root`**
  runs privileged and exists *only* for clock locking, because `/sys` is
  read-only in a stock container.
- Container GPU access needs the **host's numeric** video/render GIDs.
  `--group-add render` resolves against the container's `/etc/group` and grants
  nothing; the symptom is that `torch.cuda.device_count()` returns 8 while any
  real HIP context raises `No HIP GPUs are available`.
- The container mounts a generated `/etc/passwd`. Without it `torch.compile`
  dies in `getpass.getuser()` with `uid not found` — which would break any
  submission using `torch.compile`, not just tests.
- Anything large goes to `/var/tmp/solbench`, mounted at the same absolute path
  inside and outside the container. `/home` is NFS and nearly full.
- **`FLASHINFER_TRACE_DIR=/work` must be set** or the 9 safetensors-backed
  problems fail. The eval driver resolves those relative paths against the
  *staging* directory, not the CWD, so having the blobs in the repo is not
  enough. `env/solb` sets it. With it set, the upstream `gqa_paged_decode`
  sample passes 2/2 on real hardware.
- `env/solb` also pins `UV_PROJECT_ENVIRONMENT`/`VIRTUAL_ENV` to `/opt/venv`;
  otherwise `uv run` (used by the e2e test) builds an isolated environment
  without pydantic.

## 4. Where task 02 stopped

The port is written and imports cleanly on ROCm; **the 235-problem reference
sweep has not been run.** That sweep is the actual acceptance criterion.

Done:
- Vendored upstream v1.0.2 (`a9fa080`) at `src/sol_execbench/`, fork-structured
  for upstreaming. AMD deltas are 6 files plus a new `core/bench/device/`
  package, each marked `# AMD:` so `git diff` against upstream stays legible.
- Vendor device layer, LLC-sized cache flush, lazy CUPTI imports (upstream
  imported it at module scope, which made the whole timing module unimportable
  on ROCm), `hip_events` methodology, AMD languages/hardware enums,
  `--offload-arch` injection, `.hip` sources, AMD clock-lock path.
- Upstream's own test suite runs here: **463 passed, 75 skipped, 0 failed.**
  Skips are NVIDIA-only by construction (cupti/cuDNN/CUTLASS/cuTile deps and
  example solutions in NVIDIA-only languages). Tests that assert NVIDIA
  *behaviour* are not skipped — `tests/conftest.py` pins the vendor to
  `nvidia` for them, so the NVIDIA path stays a live regression reference.
- **The pytorch and triton example solutions pass end-to-end on real GPUs.**
  That is the evidence the ported harness actually works, not just imports.

Not done — next actions, in order:

1. Re-run `tasks/01` → new F_LOCK → add the MI350X entry to
   `CLOCK_LOCK_PRESETS`. *(Done: 1300 MHz, commit 2cdb7b0.)*
2. Write `scripts/runners/run_reference.py` (contract in
   `scripts/runners/README.md`: `--problem <dir> --out <file>`, and **on
   failure still write an output file recording the error** — a missing file
   means "not yet run" and gets redone; a recorded failure is a result).
3. Run the reference sweep over **all four categories**, then
   `python scripts/check_coverage.py --artifacts artifacts/02/references`.
   An omitted `--category` is the realistic way scope silently shrinks.
4. `artifacts/02/flush-sweep.json` — the LLC cliff check from §2.

## 5. Open items carried forward

- **Shared node.** `mia1-p02-g10` had another active user. Check whether the
  new node does too; if so, keep treating timings as provisional (the user's
  standing decision) and keep the sibling-power contamination flagging in
  `clock_calibrate.py floor`.
- **Clocks on the old node were reset to `auto`** before handoff, so
  `mia1-p02-g10` is left as found.
- `verify_artifacts.check_05` assumes tolerances are an
  `[atol, rtol, matched_ratio]` triple; upstream actually stores
  `{"max_atol", "max_rtol"}` per workload. That comparison needs adapting when
  `reference/b200-tolerances.json` is built.
- `torch==2.9.0` is upstream's pin; the ROCm build available is `2.9.1`.
  Recorded, not "fixed".
- Another user's `docker image prune` deleted the measurement image mid-session
  once. `env/solb` rebuilds automatically, but the error message points at a
  registry problem rather than the real cause.
