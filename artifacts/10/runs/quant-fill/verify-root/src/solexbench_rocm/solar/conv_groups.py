# SPDX-License-Identifier: Apache-2.0
"""Teach SOLAR that a grouped convolution is not a dense one.

SOLAR resolves a convolution's ``groups`` from ``module_args``, which the graph
processor populates only for ``nn.Module`` convolutions. Every convolution in
this benchmark is functional -- ``F.conv1d(x, w, bias, groups=g)`` -- and
torchview's attribute parser has branches for ``transpose``, ``permute``,
``view`` and the reductions and none for ``conv``, so ``groups`` never arrives.
It defaults to 1 and the arithmetic term is priced as though every input
channel fed every output channel.

Confirmed exactly, not inferred:

    L1__006  true 599,040 = 2 x 768 x 130 x 3   (the depthwise conv1d, alone)
             SOLAR    460,062,720               ratio 768.0 = the group count

    L1__029  SOLAR 10,995,116,277,760
           = dense conv 16 x 16384 x 16384 x 512 x 4   8,796,093,022,208
           + in_proj    16 x 512 x 32768 x 8192        2,199,023,255,552
             summing to SOLAR's figure to the unit.

Seven problems call a convolution with non-1 ``groups``: L1__005, L1__006,
L1__029, L2__035, L2__036, L2__051, L2__058. (A plain grep for ``groups=``
returns 44 and is wrong -- most are GQA's ``num_key_value_groups``.)

**Why groups is recovered from shapes and not from the call.** A convolution
weight is ``[out_channels, in_channels // groups, *kernel]`` and its input is
``[N, in_channels, ...]``, so ``groups = in_channels // weight.shape[1]``
exactly, for every convolution, however it was called. The argument itself can
be positional, a keyword, or carried on a module, and an argument parser that
misses one spelling fails silently in the direction that inflates the bound --
which is precisely the failure being fixed. The shapes cannot be spelled two
ways.

``in_channels`` is taken from the input for the same reason: SOLAR's
``_infer_module_args_from_param`` reads it off ``weight_shape[1]``, which for a
grouped convolution is channels *per group*, so its depthwise test
``groups == in_channels == out_channels`` cannot fire on a depthwise
convolution even when ``groups`` is known.

**Why a wrapper here rather than a patch to SOLAR.** SOLAR is pinned by SHA in
``env/Dockerfile`` and deliberately not vendored, because a silently newer SOLAR
would move every bound. A patch file would mean rebuilding the measurement image
to change a bound derivation, and the image is the thing that must not move
under a measurement. This wrapper lives in the port, is version-controlled with
the artifacts it produces, is unit-tested, and is applied explicitly by
``scripts/sol_bounds.py`` -- so an artifact either was generated with it or was
not, and ``git log`` says which.

Not applied automatically on import. Call :func:`apply`.
"""

from __future__ import annotations

from typing import Any

_APPLIED = False

#: Handler classes whose ``generate_einsum`` takes the corrected module_args.
_HANDLERS = ("Conv1dHandler", "Conv2dHandler", "Conv3dHandler")


def derive(input_shape, weight_shape, module_args: dict | None) -> tuple[int, int, int]:
    """``(groups, in_channels, out_channels)`` from the tensor shapes.

    Raises if a ``groups`` already present in *module_args* disagrees with what
    the shapes imply. Both cannot be right, and picking one silently would move
    a bound with nothing to show for it.
    """
    in_channels = int(input_shape[1])
    out_channels = int(weight_shape[0])
    per_group = int(weight_shape[1]) if len(weight_shape) > 1 else 0

    if per_group <= 0 or in_channels % per_group:
        # Not a shape a convolution weight can have against this input. Leave
        # whatever SOLAR had rather than inventing a group count for it.
        declared = int((module_args or {}).get("groups", 1))
        return declared, in_channels, out_channels

    groups = in_channels // per_group
    declared = (module_args or {}).get("groups")
    if declared is not None and int(declared) != groups:
        raise ValueError(
            f"conv groups disagree: shapes imply {groups} "
            f"(in_channels {in_channels} / {per_group} per group) but the "
            f"graph declares {declared}. One is wrong and guessing which "
            f"would move a bound silently."
        )
    return groups, in_channels, out_channels


def apply() -> bool:
    """Wrap SOLAR's conv handlers. Idempotent. True if it took effect."""
    global _APPLIED
    if _APPLIED:
        return True

    from solar.einsum.ops import conv_ops

    wrapped = 0
    for name in _HANDLERS:
        cls = getattr(conv_ops, name, None)
        if cls is None or getattr(cls.generate_einsum, "_solexbench_groups", False):
            continue
        cls.generate_einsum = _wrap(cls.generate_einsum)
        wrapped += 1

    if not wrapped:
        raise RuntimeError(
            "no SOLAR conv handler was wrapped -- the class names moved. "
            "Refusing rather than deriving bounds that quietly lost the fix."
        )
    _APPLIED = True
    return True


def _wrap(orig):
    def generate_einsum(self, op_name: str, tensor_shapes, **kwargs: Any):
        inputs = getattr(tensor_shapes, "inputs", None) or []
        input_shape = inputs[0] if len(inputs) > 0 else None
        weight_shape = inputs[1] if len(inputs) > 1 else None
        if input_shape is not None and weight_shape is not None and len(input_shape) > 1:
            module_args = dict(kwargs.get("module_args") or {})
            groups, in_channels, out_channels = derive(
                input_shape, weight_shape, module_args
            )
            # Written back in the shape the unmodified handler already reads, so
            # the einsum selection below it -- dense / depthwise / group-wise --
            # is SOLAR's own logic reached with correct inputs, not logic
            # reimplemented here.
            module_args.update(
                groups=groups, in_channels=in_channels, out_channels=out_channels
            )
            kwargs["module_args"] = module_args
        return orig(self, op_name, tensor_shapes, **kwargs)

    generate_einsum._solexbench_groups = True
    generate_einsum.__doc__ = (orig.__doc__ or "") + "\n\n" + __doc__
    return generate_einsum
