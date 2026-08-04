# artifacts/00-MI350X — archived node record

Task 00 (node acceptance) as measured on **`gbt350-odcdh1-a08-1`, 8× MI350X**
(air-cooled, 1000 W cap, 2200 MHz max GFX clock) during session 2.

Archived here, not deleted, when session 3 moved to **`mia1-p02-g10`, 8× MI355X**
(liquid-cooled, 1400 W cap, 2400 MHz max). `artifacts/00/` always describes the
node currently being measured, which is what makes
`python scripts/verify_artifacts.py --task 00` an acceptance check rather than a
history lookup. `HANDOFF.md` §2 set that precedent going the other way.

Nothing in here is an input to anything. The rooflines were taken at *default*
clocks and were never scoring ceilings; task 00's own guard rails say so.

The MI350X numbers remain useful as a second data point on the same gfx950 die
at a different power budget — 1000 W vs 1400 W is the whole reason the sustained
clock floors differ by ~350 MHz — so they are kept and labelled rather than
discarded.
