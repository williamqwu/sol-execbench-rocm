# Node defect report — `mia1-p02-g10` (MI355X x8)

**One line:** the eight GPUs are healthy and uniform when left alone, but
`rocm-smi --setperfdeterminism` — the documented mechanism for pinning a clock —
**does not work correctly on any of them**, in two distinct ways: GPUs 0 and 1
ignore the requested frequency entirely, and GPUs 2–7 land ~20% below it while
leaving 400 W of their power budget unused.

Date: 2026-08-04. Everything below is measured; reproduction at the end.
Supersedes an earlier version of this report that blamed per-socket power
provisioning — that hypothesis is disproved by finding 1.

---

## Finding 1 — the hardware is fine

Same fixed-shape BF16 GEMM (8192³, back to back) on all eight GPUs at once for
45 s, with `perf_level=auto`, i.e. nothing pinned:

| GPU | throughput | clock | power | temp |
|---|---|---|---|---|
| 0 | 1498 TFLOPS | 1824 MHz | 1382 W | 56 °C |
| 1 | 1476 | 1724 | 1399 | 63 |
| 2 | 1482 | 1768 | 1383 | 59 |
| 3 | 1475 | 1791 | 1382 | 57 |
| 4 | 1468 | 1812 | 1390 | 60 |
| 5 | 1447 | 1837 | 1377 | 59 |
| 6 | 1469 | 1810 | 1382 | 63 |
| 7 | 1490 | 1798 | 1380 | 59 |

**Spread: 3.4% in throughput.** Every card boosts to 1724–1837 MHz, draws
1377–1399 W against its 1400 W cap, and runs at 56–63 °C. There is no weak card,
no cooling imbalance, and no shortfall in power delivery — all eight sockets
deliver the full 1400 W simultaneously (~11 kW of GPU draw).

This is what rules out the baseboard/BMC power-provisioning theory in the previous
version of this report. The power is available; six cards simply decline to use it
once determinism is enabled.

## Finding 2 — enabling determinism is what breaks uniformity

Identical run, the only change being `rocm-smi --setperfdeterminism 1660` applied
to all eight first (all eight accept it and read back as `perf_determinism`):

| GPU | throughput | clock | power | vs. unlocked | honours setpoint? |
|---|---|---|---|---|---|
| 1 | 1403 TFLOPS | 1656 MHz | 1292 W | −4.9% | looks correct — but see finding 3 |
| 0 | 1184 | 1410 | 994 | −21.0% | no |
| 2 | 1111 | 1326 | 976 | −25.0% | no |
| 3 | 1111 | 1350 | 957 | −24.7% | no |
| 4 | 1123 | 1367 | 984 | −23.5% | no |
| 5 | 1106 | 1360 | 1000 | −23.6% | no |
| 6 | 1123 | 1356 | 988 | −23.5% | no |
| 7 | 1106 | 1340 | 962 | −25.8% | no |

**Spread goes from 3.4% to 21.2%.** The feature whose entire purpose is
reproducibility is what makes this node non-uniform. Six cards sit ~330 MHz below
the frequency they acknowledged, at ~980 W against a 1400 W cap, at 45–53 °C —
*cooler* than the healthy card — and `amd-smi` reports **no** violation of any
kind on them: not PPT, not thermal, not VR, not HBM, not PROCHOT. The one card
that does hold its clock is the only one reporting `PPT_VIOLATION_STATUS: ACTIVE`,
which is what a card at full tilt should look like.

Not contention: GPU 2 alone, all others idle, still sits at 1325 MHz / 977 W.
GPU 0 alone *does* hold 1657 MHz — its degradation above only appears when all
eight are loaded.

## Finding 3 — GPUs 0 and 1 ignore the setpoint completely

This is the part that matters most, and it inverts the obvious reading of the
table above. Sweeping the requested frequency, one GPU at a time, under load:

| requested | GPU 1 achieved | GPU 1 power | GPU 2 achieved | GPU 2 power |
|---|---|---|---|---|
| 1200 MHz | **1657 MHz** (1.38x) | 1297 W | 1015 MHz (0.85x) | 784 W |
| 1400 | **1656** (1.18x) | 1298 | 1155 (0.83x) | 852 |
| 1500 | **1655** (1.10x) | 1295 | 1214 (0.81x) | 898 |
| 1660 | 1656 (1.00x) | 1303 | 1322 (0.80x) | 979 |

GPU 1 runs at 1655–1657 MHz **no matter what it is asked for**. Ask for 1200 and
it runs 1657. The lock has no effect on it; it only *appeared* to be the one
healthy card because 1656 happens to coincide with the 1660 we were requesting.
GPU 0 behaves the same way when it is the only card loaded.

GPUs 2–7, by contrast, respond monotonically to the setpoint — they are the ones
where determinism is functioning as a control, just with a ~0.80–0.85 scale error.

So neither group is correct, and the two failure modes are opposite: one group
cannot be slowed down, the other cannot reach the requested speed.

## The clock telemetry is honest — this is not a reporting artifact

Worth stating explicitly, because it was our first suspicion. Dividing measured
throughput by reported clock gives **813–856 TFLOPS/GHz on every card in every
condition above** — a 5% band across healthy cards, degraded cards, locked,
unlocked, alone and loaded. A card reporting 1326 MHz delivers exactly the work
1326 MHz predicts.

That means the clocks are real: the cards genuinely run slow, the sensors are not
lying, and none of this is an artifact of our sampling or our benchmark harness.
The measurement above deliberately uses no part of that harness — just torch, one
GEMM shape, and wall-clock timing (`scripts/gpu_parity_check.py`).

## What we would ask IT / AMD to look at

This now looks like an SMU firmware or `amdgpu` bug in the performance-determinism
path, not a hardware or power-delivery fault. Specifically:

1. **Why does `perf_determinism` cap at ~980 W on GPUs 2–7** when the cap is
   1400 W, no violation bit is set, and the same cards reach 1385 W with
   determinism off? Something in the determinism path appears to substitute a
   different (~1000 W) power budget.
2. **Why is the setpoint a no-op on GPUs 0 and 1?** They accept the request,
   report `perf_determinism`, and then hold ~1656 MHz regardless. A lock that
   silently does nothing is worse than one that fails loudly.
3. Is this **known for this SMC/VBIOS combination**? All eight are identical:
   VBIOS `113-M355-01-1K1-000C`, SMC `04.86.10.05`, MEC 38, RLC 43,
   SOS `0x00450028`; ROCm-SMI-LIB 7.8.0, amdgpu 6.16.6, kernel 6.8.0-60.
4. Is there a **firmware update** for the determinism path, or a supported
   alternative for pinning clocks that we should use instead?

We are not asking for a clock change or a tuning tweak: we need one card, ideally
eight, that holds a frequency we specify and reports it accurately.

## Reproduction

```bash
# The whole thing, ~4 min: unlocked vs locked, together and alone.
python scripts/gpu_parity_check.py --seconds 45 --out /tmp/parity.json

# Finding 3 in isolation — ask GPU 1 for 1200 MHz and watch it run 1657:
rocm-smi -d 1 --setperfdeterminism 1200
rocm-smi --showperflevel          # reports perf_determinism
#   ...then load GPU 1 and read its clock under load
```

Results: `artifacts/00/gpu-parity.json`, plus per-GPU sweeps in
`artifacts/01/det1650-gpu*.json`.

Note when reproducing: **`rocm-smi -d N` and torch device order differ on this
node** (torch 0 is `rocm-smi` 3). `scripts/gpu_map.py` prints the mapping. Reading
one card's clock while setting another's is an easy mistake here, and we made it
once.

## Why it matters to us, and what we do about it

This benchmark scores kernels against a Speed-of-Light bound, so every timing must
be taken at a known fixed clock. Findings 2 and 3 mean **we cannot choose that
clock on this node** — we can only find out what a card happens to settle at and
verify it afterwards.

What we do: take all authoritative timings on **GPU 1**, which is invariantly
1655–1657 MHz at ~1295 W whether idle-adjacent or with all eight cards loaded.
That stability is what the benchmark needs, and it is genuinely stable — but per
finding 3 it comes from firmware pinning the card, **not** from the lock we apply.
Our `--setperfdeterminism 1650` on GPU 1 is a no-op, and a firmware update could
change the pinned value silently. So the pipeline asserts the *achieved* clock on
every timing run rather than trusting that the lock took effect, and the recorded
`F_LOCK` is a measurement, not a setting.

Cost: authoritative timings run serially on one card, ~20 h where eight would be
~2.5 h. The other seven remain useful for work that does not need a pinned clock.
