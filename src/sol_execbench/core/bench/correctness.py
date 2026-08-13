# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Correctness computation utilities."""

from __future__ import annotations

import random
from typing import Optional, Tuple

import torch

from sol_execbench.core.data import Correctness
from sol_execbench.core.data.workload import ToleranceSpec


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across Python, PyTorch CPU and CUDA."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_tensor_sanity(
    sol_tensor: torch.Tensor,
    ref_tensor: torch.Tensor,
    allow_negative_inf: bool = False,
) -> Optional[Correctness]:
    """Check for non-finite values and all-zeros output.

    Returns a ``Correctness`` describing the failure when either tensor
    contains inf/nan values, or the solution is all zeros while the
    reference is not.  Returns ``None`` when both tensors look sane.

    Inf and NaN are always treated as incorrect, even when both tensors
    have the same non-finite value at a given position — unless
    *allow_negative_inf* is True, in which case positions where **both**
    tensors are -inf are tolerated.
    """
    # Non-finite values in either tensor are always wrong.
    # Even matching infinities (e.g. both -inf) are rejected because
    # non-finite values indicate degenerate numerics, not correct output.
    ref_nonfinite = ~torch.isfinite(ref_tensor)
    sol_nonfinite = ~torch.isfinite(sol_tensor)

    if allow_negative_inf:
        # Exclude positions where both tensors are -inf.
        both_neg_inf = (ref_tensor == float("-inf")) & (sol_tensor == float("-inf"))
        ref_nonfinite = ref_nonfinite & ~both_neg_inf
        sol_nonfinite = sol_nonfinite & ~both_neg_inf

    has_nonfinite = ref_nonfinite.any().item() or sol_nonfinite.any().item()
    if has_nonfinite:
        has_nan = (
            torch.isnan(sol_tensor).any().item() or torch.isnan(ref_tensor).any().item()
        )
        # has_inf is True only when there are Inf values but NO NaN values.
        # NaN takes priority because it's a stricter failure mode (Inf can
        # result from overflow, but NaN indicates undefined computation).
        return Correctness(has_nan=has_nan, has_inf=not has_nan)

    # Non-zero output check: if reference has non-trivial values
    # but solution is all zeros, fail immediately.
    ref_norm = torch.linalg.vector_norm(ref_tensor.to(torch.float32))
    if (
        ref_norm.item() > 0
        and torch.linalg.vector_norm(sol_tensor.to(torch.float32)).item() == 0
    ):
        abs_err = float(ref_norm.item())
        return Correctness(
            max_absolute_error=abs_err,
            max_relative_error=abs_err,
        )

    return None


# AMD: D52. An integer or boolean output carries no representation error, so
# "within atol + rtol*|y|" is not a meaningful relaxation for one -- an index
# is either the index the reference chose or it is not. The benchmark's data
# model cannot say that per output: `Workload.tolerance` is a single
# ToleranceSpec and the driver applies it to every output of the workload
# (`eval_driver.py`, "-- Numerical correctness check --"). Task 05 used to
# resolve the mismatch at derivation time by giving the WHOLE problem the
# integer floor of {atol: 0, rtol: 0}, which made the float outputs' comparison
# bit-identity-with-eager: L2__049 and Quant__011 return
# (int64 topk_idx, float32 topk_weight) and were unpassable by construction.
#
# Splitting it here instead keeps both halves right no matter what a tolerance
# says: float outputs get the derived band, integer and boolean outputs get
# exact equality, and neither leaks into the other.
def _is_integral(t: torch.Tensor) -> bool:
    """True for integer and boolean tensors (everything with no ulp)."""
    return not (t.is_floating_point() or t.is_complex())


def _exact_error_stats(
    output: torch.Tensor, reference: torch.Tensor, tolerance: ToleranceSpec
) -> Tuple[Correctness, bool]:
    """Exact comparison, for integer and boolean outputs.

    The equality itself is decided in the OUTPUT'S OWN DTYPE, not through a
    cast. The usual path casts to float32, which cannot represent every int64:
    2**53 and 2**53+1 land on the same float and would compare equal. Only the
    reported error magnitudes go through float64, and they are triage output,
    not the verdict.

    `required_matched_ratio` and `max_error_cap` still apply. This changes only
    the WIDTH of the accepted band, from `atol + rtol*|y|` to zero; it does not
    change the harness's own rule about how many elements may miss.
    """
    x, y = output, reference
    if x.dtype != y.dtype:
        x, y = x.to(torch.int64), y.to(torch.int64)

    total_elements = x.numel()
    if total_elements == 0:
        return Correctness(), False

    mismatched = float((x != y).sum().item())
    # float64 for the magnitudes: exact for everything below 2**53, and it
    # cannot overflow the way an int64 subtraction can.
    diff = torch.abs(x.to(torch.float64) - y.to(torch.float64))
    max_abs = float(diff.max().item())
    matched_ratio = 1.0 - (mismatched / float(total_elements))
    matched_ratio = max(0.0, min(1.0, matched_ratio))

    exceeds_tol = matched_ratio < tolerance.required_matched_ratio
    if tolerance.max_error_cap is not None and max_abs > tolerance.max_error_cap:
        exceeds_tol = True

    # Reported for triage only, with a denominator floor of 1 -- the smallest
    # non-zero magnitude an integer has, so this cannot divide by near-zero.
    rel = diff / torch.clamp(torch.abs(y.to(torch.float64)), min=1.0)
    return Correctness(
        max_absolute_error=max_abs, max_relative_error=float(rel.max().item())
    ), exceeds_tol


def compute_error_stats(
    output: torch.Tensor, reference: torch.Tensor, tolerance: ToleranceSpec
) -> Tuple[Correctness, bool]:
    """Compute numerical error between *output* and *reference*.

    Returns ``(correctness, exceeds)`` where *correctness* is a
    :class:`Correctness` carrying error metrics (and ``has_nan`` /
    ``has_inf`` flags when non-finite values are detected), and *exceeds*
    is ``True`` when the tolerance is violated.
    """
    x = output.to(torch.float32)
    y = reference.to(torch.float32)

    allow_neg_inf = tolerance.allow_negative_inf

    # Automatically fail on infs/nans in either tensor even if they're in the same position.
    infs_nans = check_tensor_sanity(x, y, allow_negative_inf=allow_neg_inf)
    if infs_nans is not None:
        return infs_nans, True

    # AMD: D52 -- integer and boolean outputs are compared exactly. Placed
    # AFTER check_tensor_sanity on purpose: the all-zeros check above is an
    # anti-reward-hack guard, and an all-zero integer output against a mostly
    # zero reference would otherwise slip through on matched ratio alone.
    if _is_integral(output) and _is_integral(reference):
        return _exact_error_stats(output, reference, tolerance)

    # When allow_negative_inf is set, exclude matching -inf positions from
    # error computation — they have already been validated by check_tensor_sanity.
    if allow_neg_inf:
        both_neg_inf = (x == float("-inf")) & (y == float("-inf"))
        finite_mask = ~both_neg_inf
        x = x[finite_mask]
        y = y[finite_mask]

    abs_error = torch.abs(x - y)
    total_elements = abs_error.numel()
    if total_elements == 0:
        return Correctness(), False

    max_abs = float(abs_error.max().item())

    # torch.allclose style: |a - b| <= atol + rtol * |b|
    # ensure nans automatically exceed tolerance
    tol_bound = tolerance.max_atol + tolerance.max_rtol * torch.abs(y)
    exceeds_tol_mask = (abs_error > tol_bound) | ~torch.isfinite(abs_error)
    del tol_bound  # save VRAM

    exceeds_count = float(exceeds_tol_mask.sum().item())
    matched_ratio = 1.0 - (exceeds_count / float(total_elements))
    matched_ratio = max(0.0, min(1.0, matched_ratio))

    # Hard ceiling on max absolute error (cuDNN pattern).
    # Prevents accepting solutions where most elements match but rare outliers
    # have arbitrarily large errors.
    exceeds_tol = matched_ratio < tolerance.required_matched_ratio
    if tolerance.max_error_cap is not None and max_abs > tolerance.max_error_cap:
        exceeds_tol = True

    # Relative error using max_atol as floor to avoid division-by-near-zero
    rel_error = abs_error / torch.clamp(torch.abs(y), min=tolerance.max_atol)
    max_rel = float(rel_error.max().item())

    return Correctness(
        max_absolute_error=max_abs, max_relative_error=max_rel
    ), exceeds_tol
