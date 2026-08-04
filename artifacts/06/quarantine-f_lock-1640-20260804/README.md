# Quarantined: T_b measured at the retired F_LOCK

13 authoritative T_b artifacts measured at F_LOCK 1640 MHz on GPU 0, before the
anchor clock was changed to 1480 MHz with per-GPU setpoints.

They are not wrong — they are correct measurements at a clock the benchmark no
longer uses. T_b is a wall-clock time, so combining them with anything measured at
1480 would rescale those problems' scores by 1640/1480, per problem, invisibly.
`collect_t_b()` would in fact reject them now (F18), which is the check working.

Retired because 1640 sat on a discontinuity — setpoint 1650 yielded 1644 MHz in one
measurement and 1397 in another — and because only two of eight GPUs could reach it,
which forced the T_b pass to run serially on one card.
