# SPDX-License-Identifier: Apache-2.0
"""A device index must be resolved from every spelling, or refused.

This file exists for one bug. `_clock_info` resolved its argument with

    idx = 0 if device is None else int(getattr(device, "index", device))

which is correct for None, for an int and for a `torch.device` — and silently
wrong for the string `"cuda:0"`, because `str` HAS an `.index` attribute: the
`str.index` *method*. `getattr` returns it, the default is never reached, and
`int(<built-in method>)` raises TypeError inside a bare `except Exception:
return None`.

`eval_driver.py:351` sets `_device = "cuda:0"`. So every clock sample in the
first bracketed T_b sweep returned None, every measurement was refused for
"no clock evidence", 100% of workloads lost their anchor, and the exit status
was 0.

The tests are mostly about the two ways this can go wrong again: a spelling that
resolves to the WRONG card, and a spelling that resolves to a plausible default
instead of refusing.
"""

from __future__ import annotations

import pytest

from sol_execbench.core.bench.device.amd import torch_index_of


@pytest.mark.parametrize("device,expected", [
    (None, 0),
    (0, 0),
    (3, 3),
    (7, 7),
    ("cuda:0", 0),          # the exact string the eval driver passes
    ("cuda:3", 3),
    ("hip:5", 5),
    ("cuda", 0),            # bare type: the current device
    ("0", 0),
    ("  cuda:2  ", 2),
])
def test_every_accepted_spelling_resolves(device, expected):
    assert torch_index_of(device) == expected


def test_the_string_the_eval_driver_actually_passes():
    """Pinned on its own, by value, because this is THE case that failed. If a
    future refactor breaks it again, this test names the sweep it costs."""
    assert torch_index_of("cuda:0") == 0


def test_a_torch_device_object_still_resolves():
    torch = pytest.importorskip("torch")
    assert torch_index_of(torch.device("cuda", 4)) == 4


def test_str_index_is_a_method_which_is_why_the_old_code_failed():
    """The trap itself, so nobody reintroduces `getattr(device, "index", device)`.

    `hasattr(str, "index")` is True. Any resolution that goes through `getattr`
    without a type check walks straight into it.
    """
    assert hasattr("cuda:0", "index")
    assert not isinstance(getattr("cuda:0", "index"), int)
    with pytest.raises(TypeError):
        int(getattr("cuda:0", "index", "cuda:0"))


@pytest.mark.parametrize("bad", [
    "cpu", "meta", "cuda:", "cuda:x", "gpu-1", "", "   ",
    object(), 1.5, True, False, [0], {"index": 0},
])
def test_anything_unrecognised_raises_rather_than_defaulting_to_zero(bad):
    """Refusing beats guessing, and 0 is the most dangerous possible guess.

    A default of 0 reads *some* card and returns a perfectly plausible
    frequency. That is the §8.1 ordering failure again — and unlike the bug this
    file is about, it would be undetectable rather than merely total.
    """
    with pytest.raises(ValueError):
        torch_index_of(bad)


def test_bool_is_not_a_device_even_though_it_is_an_int():
    """`isinstance(True, int)` is True. `torch_index_of(True) == 1` would read
    card 1 because someone passed a flag."""
    with pytest.raises(ValueError):
        torch_index_of(True)
