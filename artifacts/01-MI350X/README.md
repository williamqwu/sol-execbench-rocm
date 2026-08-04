# artifacts/01-MI350X — archived clock calibration

Task 01 (clock calibration) as measured on **`gbt350-odcdh1-a08-1`, 8× MI350X**
during session 2. **F_LOCK = 1300 MHz achieved, at `--setperfdeterminism 1600`.**

Archived when session 3 moved to **`mia1-p02-g10`, 8× MI355X**. Read
`artifacts/01/` for the live node.

## Do not carry F_LOCK across parts

MI350X and MI355X are the same gfx950 die in different chassis. The sustained
clock floor is a property of the part *and its chassis*, not of the
architecture: MI350X is air-cooled at 1000 W, MI355X liquid-cooled at 1400 W,
and both are power-limited rather than thermally limited. Every T_SOL in
milliseconds and every T_b depends on F_LOCK, so a carried-over value would
rescale every score by a constant that nothing downstream could detect. Prime
directive 3.

## The finding worth keeping

Deviation D8: on MI350X, `rocm-smi --setperfdeterminism X` does **not** yield X.
It yields about 0.81–0.85·X, rock-steadily, and above ~1900 it stops responding
to X at all and pins to the power cap. So F_LOCK must be the *achieved* clock,
measured under sustained load, never the requested one.

The determinism sweep in `determinism-sweep*.json` is the evidence for that, and
it is the reason the same sweep is re-run on every new part rather than assumed.
