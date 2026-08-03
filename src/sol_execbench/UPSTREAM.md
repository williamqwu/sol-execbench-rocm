# Vendored upstream

This tree is a fork of NVIDIA's `sol-execbench`, vendored so the AMD port can
be kept as a thin, reviewable delta (PLAN.md R9: "keep AMD deltas as a thin
layer"; PLAN.md §"Fork identity" recommends building as a fork structured for
upstreaming).

| | |
|---|---|
| Source | https://github.com/NVIDIA/SOL-ExecBench |
| Version | 1.0.2 (`pyproject.toml`) |
| Commit | `a9fa0804c793d438e70850c33fe34426e66d53dd` |
| Vendored | 2026-08-03T17:11:54Z |
| License | Apache-2.0 — see `LICENSE.upstream` |

**The NVIDIA path must keep working.** It is the regression reference: when an
AMD number looks wrong, running the same code on NVIDIA distinguishes "the
refactor broke something" from "AMD genuinely differs" (tasks/02 guard rail).

Every AMD delta is marked with a `# AMD:` comment so `git diff` against
upstream stays legible.
