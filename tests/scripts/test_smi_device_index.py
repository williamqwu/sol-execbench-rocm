# SPDX-License-Identifier: Apache-2.0
"""Per-device SMI calls must translate the torch index, whichever CLI is used.

`amd-smi -g` and `rocm-smi -d` both order devices by PCI bus; torch does not. On the
node these tests were written against, torch 1 is device 0 to both tools, so passing a
torch index straight through addresses a *different physical GPU* — and returns a
perfectly plausible number, because the card actually asked about was left alone.

This has now cost the project three findings (STATE.md D11, D20's clock alignment, and
a claim that `--setperfdeterminism` was a no-op on two cards, which was really a
setpoint landing on their neighbours). Two call sites in `clock_calibrate.py` still had
it latent when these tests were added: the `read_clocks` CLI fallback and the per-GPU
form of `set_perf_determinism`.

The point worth keeping in mind while reading: **switching from rocm-smi to amd-smi
does not fix this.** Both tools enumerate identically. The translation is the fix.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import clock_calibrate as cc  # noqa: E402

# The mapping measured on mia1-p02-g10; only 2 and 6 are fixed points, which is why a
# test that only ever checks device 2 would pass while the bug was live.
SCRAMBLED = {0: 3, 1: 0, 2: 2, 3: 1, 4: 7, 5: 4, 6: 6, 7: 5}


@pytest.fixture
def scrambled(monkeypatch):
    """Force the scrambled mapping, so these tests do not need a GPU."""
    fake = mock.MagicMock()
    fake.torch_to_amdsmi.return_value = dict(SCRAMBLED)
    monkeypatch.setitem(sys.modules, "gpu_map", fake)
    return fake


def test_index_is_translated_not_passed_through(scrambled):
    assert cc.smi_device_index(1) == 0, "torch 1 is device 0 to both CLIs"
    assert {t: cc.smi_device_index(t) for t in range(8)} == SCRAMBLED


def test_the_fixed_points_are_not_what_a_test_should_rely_on(scrambled):
    """torch 2 and 6 map to themselves. A check written only against those cannot
    distinguish a translating implementation from a broken one, which is how the
    original bug survived being 'verified' on GPU 2."""
    assert cc.smi_device_index(2) == 2 and cc.smi_device_index(6) == 6
    assert [t for t in range(8) if cc.smi_device_index(t) != t] == [0, 1, 3, 4, 5, 7]


def test_set_perf_determinism_sends_the_translated_index(scrambled, monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return mock.MagicMock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(cc.subprocess, "run", fake_run)
    monkeypatch.setattr(cc, "perf_levels", lambda: {"/x": "perf_determinism"})

    cc.set_perf_determinism(1660, gpu=1)
    assert "-g" in seen["cmd"], "per-GPU form must address a device"
    assert seen["cmd"][seen["cmd"].index("-g") + 1] == "0", \
        f"torch 1 must become device 0, got {seen['cmd']}"
    assert "1" not in seen["cmd"][seen["cmd"].index("-g") + 1]


def test_node_wide_form_takes_no_device_at_all(scrambled, monkeypatch):
    """The form that cannot be got wrong, and the reason the node-wide measurements
    survived the bug that invalidated the per-card ones."""
    seen = {}
    monkeypatch.setattr(cc.subprocess, "run",
                        lambda cmd, **kw: (seen.update(cmd=cmd),
                                           mock.MagicMock(returncode=0, stdout="",
                                                          stderr=""))[1])
    monkeypatch.setattr(cc, "perf_levels", lambda: {"/x": "perf_determinism"})

    cc.set_perf_determinism(1660, gpu=None)
    assert "-g" not in seen["cmd"]


def test_unresolvable_index_refuses_rather_than_guessing(monkeypatch):
    """If the mapping cannot be built, reading the wrong card is worse than failing:
    a wrong clock is indistinguishable from a right one downstream."""
    broken = mock.MagicMock()
    broken.torch_to_amdsmi.side_effect = RuntimeError("no amdsmi")
    monkeypatch.setitem(sys.modules, "gpu_map", broken)
    monkeypatch.setattr(cc, "_amdsmi", lambda: None)

    got = cc.read_clocks(1)
    assert got["source"] == "unresolved"
    assert "cannot resolve torch 1" in got["error"]


def test_no_per_device_rocm_smi_calls_remain():
    """A grep-level guard. `rocm-smi -d` and `amd-smi -g` are equally dangerous with
    an untranslated index, so what this forbids is the raw literal next to a device
    argument anywhere in the module."""
    src = (ROOT / "scripts" / "clock_calibrate.py").read_text()
    for bad in ('"-d", str(gpu)', '"-g", str(gpu)',
                '"-d", str(args.gpu)', '"-g", str(args.gpu)'):
        assert bad not in src, (
            f"{bad!r} passes a torch index to an SMI CLI; route it through "
            f"smi_device_index()")
