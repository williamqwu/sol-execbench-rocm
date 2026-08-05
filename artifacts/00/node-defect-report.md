# Node defect report — `mia1-p02-g10`

**Summary in one line:** seven of the eight MI355X GPUs cap themselves at about
**1350 MHz and 975 W** while reporting a 1400 W limit, running at 45–53 °C, and
reporting **no throttling condition of any kind**. Only one GPU reaches its rated
envelope.

Date: 2026-08-04. All figures measured, reproduction steps below.

---

## What we observe

The eight GPUs are given the same clock lock:

```
rocm-smi --setperfdeterminism 1660
```

All eight accept it and report `perf_determinism`. Then, with all eight running the
same saturating BF16 GEMM for 60+ seconds:

| GPU (torch index) | clock held | power | junction temp | honours the request? |
|---|---|---|---|---|
| 1 | **1654 MHz** | **1275 W** | 58 °C | **yes** |
| 0 | 1423 MHz | 980 W | 45 °C | no |
| 2 | 1326 MHz | 972 W | 48 °C | no |
| 3 | 1342 MHz | 950 W | 45 °C | no |
| 4 | 1365 MHz | 980 W | 47 °C | no |
| 5 | 1359 MHz | 993 W | 48 °C | no |
| 6 | 1354 MHz | 982 W | 53 °C | no |
| 7 | 1340 MHz | 948 W | 47 °C | no |

Seven cards stop at ~80% of the requested clock and ~70% of their stated power
limit.

## Why we do not think this is normal behaviour

Each of the usual explanations is ruled out by measurement:

| candidate cause | ruled out because |
|---|---|
| Thermal throttling | The slow cards are **cooler** (45–53 °C) than the healthy one (58 °C). MI355X slows around 100 °C. |
| Power cap reached | `rocm-smi --showmaxpower` reports **1400 W on all eight**. The slow cards sit at 948–993 W, leaving ~420 W unused. |
| Cooling loop imbalance | Would show as higher temperatures on the affected cards. The opposite is observed. |
| Firmware / VBIOS mismatch | **Identical on all eight**: VBIOS `113-M355-01-1K1-000C`, SMC `04.86.10.05`, MEC 38, RLC 43, SOS `0x00450028`. |
| Different clock range | **Identical on all eight**: supported sclk 500–2400 MHz. |
| A throttle we are not seeing | See below — the slow cards report no violation at all. |

## The most telling measurement

`amd-smi metric` under the same full-node load:

```
GPU that WORKS (torch 1)              GPU that DOES NOT (torch 2)
  SOCKET_POWER:            1274 W       SOCKET_POWER:            975 W
  PPT_VIOLATION_STATUS:  ACTIVE         PPT_VIOLATION_STATUS:  NOT ACTIVE
  PPT_VIOLATION_ACTIVITY:     1 %       PPT_VIOLATION_ACTIVITY:    0 %
  PROCHOT / SOCKET_THERMAL / VR_THERMAL / HBM_THERMAL: NOT ACTIVE on both
```

The healthy card is bumping against its package power limit, which is what a card
at full tilt should do. The other seven report **no limit being hit at all** — no
power, no thermal, no VR, no HBM, no PROCHOT — and still refuse to go faster while
cold with 420 W of headroom.

A card that is neither hot nor power-limited nor reporting a throttle, and which is
still 20% below the clock it was asked for and acknowledged, is not being limited by
anything it will admit to.

## What we would ask IT to check

In rough order of likelihood:

1. **Per-socket power provisioning on the OAM baseboard.** Seven of eight cards
   behave as though budgeted around **1000 W** rather than 1400 W. 1000 W happens to
   be the MI350X (air-cooled) envelope, so a baseboard or BMC configured for the
   lower-TDP SKU on seven of eight sockets would produce exactly this. The 1400 W
   figure the driver reports may be the card's rating rather than what the board
   will deliver.
2. **BMC / sled-level power capping.** Total node draw at full load is ~8.1 kW. If
   the sled or PDU budget is near 8 kW, firmware may be clamping per-card. Worth
   checking whether a chassis power limit is configured, and what it is.
3. **Voltage regulator or power stage health on the affected sockets** — seven
   cards, all identical behaviour, is more likely a configuration than seven
   coincident faults, but worth ruling out.
4. Whether this node was **re-provisioned from an MI350X configuration**, or whether
   any per-socket power policy was set in firmware.

We are **not** asking for a clock change or a tuning tweak. The request is that all
eight sockets deliver the 1400 W the driver advertises, so that all eight cards can
hold a requested clock.

## Reproduction

```bash
# 1. Apply the same lock to every GPU
rocm-smi --setperfdeterminism 1660
rocm-smi --showperflevel          # all eight report perf_determinism

# 2. Load all eight and watch, for at least 60 s
rocm-smi --showuse --showpower --showgpuclocks --showtemp

# 3. The throttle counters are the interesting part
amd-smi metric -g 0 | grep -A12 THROTTLE     # a healthy card: PPT ACTIVE
amd-smi metric -g 2 | grep -A12 THROTTLE     # an affected card: nothing ACTIVE
```

Our own reproduction is `scripts/clock_calibrate.py determinism-sweep --gpu N
--freqs 1660`, one GPU at a time, and the results are in
`artifacts/01/det1650-gpu*.json` and `artifacts/01/equalized-clocks-REJECTED-unpinned.json`.

Note when reproducing: **`rocm-smi -d N` and the torch device order are different
orderings on this node** (torch 0 is `rocm-smi` 3). `scripts/gpu_map.py` prints the
mapping. Reading one card's clock while setting another's is an easy mistake here
and we made it once.

## Why it matters to us

This benchmark scores GPU kernels against a Speed-of-Light bound, and every timing
must be taken at a known, fixed clock. A card that holds a requested clock is
usable; a card that silently lands 20% low is not, because the resulting times are
wrong by that ratio and nothing in the output shows it.

With one healthy card out of eight we can still produce correct results, but the
timing passes run **serially on that one GPU**: measured at roughly 20 hours where
eight cards would be two and a half. The other seven are usable for work that does
not need a pinned clock, so the machine is not idle — but the measurement that
defines the benchmark's scale is bottlenecked on a single card.
