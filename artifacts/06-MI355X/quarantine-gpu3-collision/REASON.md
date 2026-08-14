# Quarantine: GPU-3 timing/exploration collision, 2026-08-14

`authoritative_tb.py --shard 3/8 --gpu 3` was timing on g46 GPU 3 while
`run_agents.py --run-id quant-fill --gpus 2,3,4,5` ran agent verification on
the same card. Four quant-fill units are logged on GPU 3 inside the window.

CLAUDE.md 4: timing runs and exploration must not share a GPU. These five T_b
records were written between 22:06:47Z (quant-fill start) and the collision
being noticed at 22:24Z, so each may have been anchored against a card that
was simultaneously running another workload. An anchor measured under
contention is biased slow, which biases every S computed against it HIGH --
the flattering direction, and therefore the one that must not be shipped
unnoticed.

Not deleted, because they are real measurements of something; not used,
because what they measured is not what the anchor is defined to be.
Re-measured on an idle card; the replacements carry their own card_identity.

Cause was mine: I launched quant-fill onto a GPU set that overlapped a running
authoritative shard without checking. The fleet monitor added the same day
(`scripts/fleet_monitor.py`) exists so the overlap is visible before launch.
