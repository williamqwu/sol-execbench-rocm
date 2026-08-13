# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the tolerance floor in scripts/runners/calibrate_tolerance.py.

CPU-only. The defect these cover (STATE.md D52) produced no error and no
warning: `_dtype_floor` read `torch.finfo(tensors[0].dtype)` inside a
`except TypeError: return {"atol": 0.0, "rtol": 0.0}`, so a problem whose FIRST
output is an index tensor got a ZERO floor -- and that zero was then the
tolerance for the problem's FLOAT outputs too. With a bit-exact reference
(`max_abs == 0`) the shipped band was exactly zero, i.e. bit-identity with
eager. `artifacts/05` records it verbatim on 76 workloads:

    "max_atol": 0.0, "max_rtol": 0.0,
    "_derivation": "... floored at torch.int64 epsilon"

32 of those 76 (L2__049, Quant__011) also return a float32 output and were
unpassable by construction. The other 44 (L1__028, L1__058, L2__006) are
all-integer problems, where a zero band IS exact equality and is correct --
so the fix must keep those at zero.

D52b is the same defect one dtype over: after the integer outputs were filtered
out, `tensors[0].dtype` still decided the epsilon and the RMS scale was still
summed across dtypes. 17 of the 235 problems return more than one float dtype
(16 scoreable, 396 workloads; counted from the definition files this session),
every one of them `{bfloat16, float32}` with bf16 first, so their fp32 outputs
were floored at bf16's epsilon -- 0.0078125 against 1.1920929e-07.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "runners"))

from calibrate_tolerance import (  # noqa: E402
    _dtype_floor,
    _floor_desc,
    _is_exact,
)

F32_EPS = float(torch.finfo(torch.float32).eps)
BF16_EPS = float(torch.finfo(torch.bfloat16).eps)


def _idx(*values) -> torch.Tensor:
    """An index tensor, the shape L2__049's topk_idx has."""
    return torch.tensor(list(values), dtype=torch.int64)


def _weights(*values) -> torch.Tensor:
    return torch.tensor(list(values), dtype=torch.float32)


class TestIsExact:
    def test_integer_and_boolean_outputs_are_exact(self):
        assert _is_exact(_idx(0, 1, 2))
        assert _is_exact(torch.tensor([True, False]))
        assert _is_exact(torch.tensor([1, 2], dtype=torch.int32))

    def test_floating_point_outputs_are_not(self):
        assert not _is_exact(_weights(1.0, 2.0))
        assert not _is_exact(torch.tensor([1.0], dtype=torch.bfloat16))


class TestDtypeFloor:
    def test_an_index_output_first_does_not_zero_the_float_floor(self):
        """The D52 case: (int64 topk_idx, float32 topk_weight)."""
        floor = _dtype_floor([_idx(3, 7, 1, 0), _weights(1.0, 1.0, 1.0, 1.0)])

        assert floor["rtol"] == F32_EPS
        assert floor["atol"] > 0.0, (
            "a zero atol on a float output is bit-identity-with-eager, which "
            "no reassociating kernel can pass"
        )
        assert floor["dtype"] == "torch.float32"

    def test_the_float_floor_is_the_same_wherever_the_index_output_sits(self):
        """Order must not matter, and it did: only `tensors[0]` was inspected."""
        first = _dtype_floor([_idx(3, 7), _weights(1.0, 1.0)])
        second = _dtype_floor([_weights(1.0, 1.0), _idx(3, 7)])
        assert first == second

    def test_index_magnitudes_do_not_inflate_the_float_scale(self):
        """The reverse leak, with a float output first so no TypeError fired.

        The RMS scale summed over every output, so an index of 4096 bought the
        float output a floor 2900x wider than its own magnitudes justify.
        """
        floats_only = _dtype_floor([_weights(1.0, 1.0)])
        with_indices = _dtype_floor([_weights(1.0, 1.0), _idx(4096, 4096)])
        assert with_indices["atol"] == pytest.approx(floats_only["atol"])

    def test_all_integer_outputs_still_floor_at_zero(self):
        """L1__058, L1__028, L2__006: zero here means exact, and is right.

        Each half of the invariant is asserted on its own. The dict as a whole
        grew keys (per-dtype floors, D52b), so an equality check would now be
        testing the record format; what must not change is that an all-integer
        problem gets a band of exactly zero and names no dtype.
        """
        floor = _dtype_floor([_idx(0, 1), torch.tensor([True, False])])

        assert floor["atol"] == 0.0, "a non-zero atol on an index is slack"
        assert floor["rtol"] == 0.0, "a non-zero rtol on an index is slack"
        assert floor["dtype"] is None
        assert floor["per_dtype"] == []

    def test_the_all_integer_band_composes_to_exact_equality(self):
        """The two halves of "zero means exact", checked together.

        Stated honestly, because the two are not the same mechanism: the
        REJECTION comes from the harness (`compute_error_stats` routes an
        integer output to exact comparison whatever the spec says, D52), and
        the ZERO comes from the floor above. Widening the floor would not move
        this verdict on an all-integer problem -- verified this session by
        mutating the all-integer return to `{"atol": 1.0, "rtol": 1.0}`, which
        fails the assertions above and leaves this test passing. So this test
        pins the composition, and the explicit `== 0.0` assertions above are
        what pin the band; neither substitutes for the other.
        """
        from sol_execbench.core.bench.correctness import compute_error_stats
        from sol_execbench.core.data.workload import ToleranceSpec

        floor = _dtype_floor([_idx(0, 1), torch.tensor([True, False])])
        band = ToleranceSpec(max_atol=floor["atol"], max_rtol=floor["rtol"],
                             required_matched_ratio=1.0)

        ref = _idx(*range(64))
        _c, exceeds = compute_error_stats(ref.clone(), ref, band)
        assert not exceeds, "an identical index tensor must pass"

        off_by_one = ref.clone()
        off_by_one[7] += 1
        _c, exceeds = compute_error_stats(off_by_one, ref, band)
        assert exceeds, (
            "one wrong index must fail: a band of zero is the whole of what "
            "an all-integer problem's floor buys"
        )

    def test_a_second_float_dtype_gets_its_own_floor(self):
        """D52b: 17 problems return more than one float dtype, bf16 first.

        Counted from the 235 `data/SOL-ExecBench/benchmark/*/*/definition.json`
        files (16 scoreable, 396 workloads; Quant__033 is deferred). Deriving
        one floor from the FIRST output gave every one of their fp32 outputs
        bf16's epsilon.
        """
        bf16 = torch.tensor([1.0, 1.0], dtype=torch.bfloat16)
        fp32 = _weights(1.0, 1.0)
        floor = _dtype_floor([bf16, fp32])

        by_dtype = {d["dtype"]: d for d in floor["per_dtype"]}
        assert set(by_dtype) == {"torch.bfloat16", "torch.float32"}
        assert by_dtype["torch.float32"]["rtol"] == F32_EPS
        assert by_dtype["torch.bfloat16"]["rtol"] == BF16_EPS

    def test_the_rms_scale_does_not_cross_a_dtype_boundary(self):
        """The reverse leak again, between two FLOAT dtypes this time.

        The bf16 output's magnitudes used to be summed into the same RMS as the
        fp32 output's, so a large bf16 output widened the fp32 floor.
        """
        big_bf16 = torch.tensor([1024.0, 1024.0], dtype=torch.bfloat16)
        small_fp32 = _weights(1.0, 1.0)
        mixed = _dtype_floor([big_bf16, small_fp32])

        # The APPLIED atol, which is the number that ships. bf16 owns it here
        # (0.0078125 x 1024 against float32's 1.1920929e-07 x 1), and 1024 is
        # the bf16 outputs' own RMS.
        assert mixed["atol"] == pytest.approx(BF16_EPS * 1024.0)
        # What the cross-dtype sum used to give: RMS over all four elements,
        # sqrt((1024**2 + 1024**2 + 1 + 1)/4) = 724.077..., so the shipped atol
        # was 5.657 where the bf16 outputs justify 8.0.
        cross = BF16_EPS * ((1024.0 ** 2 * 2 + 2.0) / 4) ** 0.5
        assert mixed["atol"] != pytest.approx(cross)

        fp32_group = [d for d in mixed["per_dtype"]
                      if d["dtype"] == "torch.float32"][0]
        assert fp32_group["atol"] == pytest.approx(_dtype_floor(
            [small_fp32])["atol"])
        assert fp32_group["rms"] == pytest.approx(1.0)

    def test_a_large_fp32_output_no_longer_widens_the_bf16_floor(self):
        """The other side of the same sum, where the leak is permissive.

        bf16 outputs at magnitude 1 next to fp32 outputs at 1024: the summed
        RMS was 724.077, so the applied atol was 0.0078125 x 724.077 = 5.657
        for outputs whose own dtypes justify 0.0078125 (bf16) and
        1.2207e-04 (fp32).
        """
        bf16 = torch.tensor([1.0, 1.0], dtype=torch.bfloat16)
        big_fp32 = _weights(1024.0, 1024.0)
        floor = _dtype_floor([bf16, big_fp32])

        assert floor["atol"] == pytest.approx(BF16_EPS * 1.0)
        assert floor["atol"] < 5.0, (
            "the fp32 outputs' magnitudes must not scale the bf16 floor"
        )

    def test_the_applied_floor_is_the_widest_and_says_so(self):
        """One ToleranceSpec per workload, so the floors must collapse to one.

        `Workload.tolerance` is a single ToleranceSpec applied to every output,
        so the max is the only safe collapse: the min would hold the bf16
        output to fp32's epsilon, which is D52's unpassable-by-construction
        failure again. The cost to fp32 is recorded, not hidden.
        """
        bf16 = torch.tensor([1.0, 1.0], dtype=torch.bfloat16)
        floor = _dtype_floor([bf16, _weights(1.0, 1.0)])

        assert floor["rtol"] == BF16_EPS
        assert floor["rtol_dtype"] == "torch.bfloat16"
        assert floor["over_grant_rtol"] == pytest.approx(BF16_EPS / F32_EPS)
        # 0.0078125 / 1.1920929e-07, computed in this session.
        assert floor["over_grant_rtol"] == pytest.approx(65536.0)

    def test_a_single_float_dtype_over_grants_nothing(self):
        floor = _dtype_floor([_weights(1.0, 2.0), _weights(3.0, 4.0)])
        assert floor["over_grant_atol"] == pytest.approx(1.0)
        assert floor["over_grant_rtol"] == pytest.approx(1.0)

    def test_no_outputs_at_all_floors_at_zero(self):
        assert _dtype_floor([])["atol"] == 0.0

    def test_float_floor_is_one_ulp_at_the_output_rms(self):
        """Unchanged behaviour for a float-only problem."""
        floor = _dtype_floor([_weights(3.0, 4.0)])          # RMS = 3.5355...
        assert floor["atol"] == pytest.approx(F32_EPS * (12.5 ** 0.5))
        assert floor["rtol"] == F32_EPS


class TestFloorDerivationString:
    """What `artifacts/05` will say, since that is where a reader looks."""

    def test_one_float_dtype_reads_as_before(self):
        desc = _floor_desc(_dtype_floor([_weights(1.0, 1.0)]))
        assert desc == "floored at torch.float32 epsilon x output RMS"

    def test_all_integer_says_exact_equality(self):
        desc = _floor_desc(_dtype_floor([_idx(1, 2)]))
        assert desc == "no floating-point output: exact equality"

    def test_two_float_dtypes_name_both_floors_and_the_over_grant(self):
        bf16 = torch.tensor([1.0, 1.0], dtype=torch.bfloat16)
        desc = _floor_desc(_dtype_floor([bf16, _weights(1.0, 1.0)]))
        assert "floored at torch.bfloat16 epsilon" in desc
        assert "torch.float32" in desc and "torch.bfloat16" in desc
        assert "65536x on rtol" in desc

    def test_a_zero_scale_does_not_format_none(self):
        """`over_grant_atol` is None when a dtype's RMS is 0; say so, not 1.0."""
        bf16 = torch.tensor([1.0, 1.0], dtype=torch.bfloat16)
        zeros = _weights(0.0, 0.0)
        floor = _dtype_floor([bf16, zeros])
        assert floor["over_grant_atol"] is None
        assert "unboundedly" in _floor_desc(floor)


class TestWhatTheFloorBuysL2049:
    """The consequence, composed with the harness's own band.

    L2__049's reported divergence is exactly one fp32 ulp on 4488 of 16384
    elements of `topk_weight`, with `topk_idx` bit-identical -- `mr = 0.726`
    against the shipped band of zero, and no multiplier up to 2**20 passes,
    because no multiple of zero is non-zero.

    This is not a re-derivation (that is a GPU sweep). It composes the floor
    this module computes with `compute_error_stats`, and it holds for any
    scale: `ulp(y) = eps * 2**floor(log2|y|) <= eps * |y|`, so a one-ulp
    difference always sits inside `atol + rtol*|y|` once rtol is the dtype
    epsilon -- which is what the floor now guarantees for a float output.
    """

    def _one_ulp_apart(self):
        torch.manual_seed(0)
        y = torch.rand(16384, dtype=torch.float32) * 0.4 + 0.05
        x = y.clone()
        # 4488 elements moved by exactly one ulp, as measured on L2__049.
        moved = torch.arange(4488)
        x[moved] = torch.nextafter(y[moved], torch.tensor(float("inf")))
        assert not torch.equal(x, y)
        return x, y

    def test_one_ulp_fails_the_zero_band_that_D52_shipped(self):
        from sol_execbench.core.bench.correctness import compute_error_stats
        from sol_execbench.core.data.workload import ToleranceSpec

        x, y = self._one_ulp_apart()
        shipped = ToleranceSpec(max_atol=0.0, max_rtol=0.0,
                                required_matched_ratio=0.99)
        _c, exceeds = compute_error_stats(x, y, shipped)
        assert exceeds, "this is the state artifacts/05 records today"

    def test_one_ulp_passes_the_band_the_corrected_floor_gives(self):
        from sol_execbench.core.bench.correctness import compute_error_stats
        from sol_execbench.core.data.workload import ToleranceSpec

        x, y = self._one_ulp_apart()
        floor = _dtype_floor([_idx(*range(16384)), y])   # int64 output first
        corrected = ToleranceSpec(max_atol=floor["atol"], max_rtol=floor["rtol"],
                                  required_matched_ratio=0.99)
        _c, exceeds = compute_error_stats(x, y, corrected)
        assert not exceeds
