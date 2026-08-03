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

from __future__ import annotations

import bisect
import statistics
from collections.abc import Callable
from typing import Any, Literal, Union

import torch

# AMD: `cuda.bindings` and `cupti` are NVIDIA-only wheels, absent on a ROCm
# host, and `cupti_utils` imports `cupti` at module scope too. Importing any of
# them here made this whole module unimportable on AMD -- including the
# vendor-neutral CUDA/HIP-event timing path, which is what AMD actually ships
# on. They are now resolved lazily, inside the CUPTI methodology only.
from sol_execbench.core.bench import device as device_layer
from sol_execbench.core.bench.io import ShiftingMemoryPoolAllocator


def _cupti():
    """Import the CUPTI binding on demand."""
    from cupti import cupti

    return cupti


def gpu_timing_activity_kinds() -> tuple:
    """CUPTI activity kinds that count as GPU work."""
    cupti = _cupti()
    return (
        cupti.ActivityKind.CONCURRENT_KERNEL,
        cupti.ActivityKind.MEMCPY,
        cupti.ActivityKind.MEMSET,
    )


def get_l2_cache_size(device) -> int:
    """
    Get L2 cache size in bytes for the given CUDA device.

    Args:
        device: CUDA device (int, torch.device, or None for current device).

    Returns:
        L2 cache size in bytes.
    """
    props = torch.cuda.get_device_properties(device)
    return props.L2_cache_size


def _summarize_statistics(
    times: list[float],
    return_mode: Literal["mean", "median", "all"],
) -> Union[float, list[float]]:
    """Summarize timing statistics based on return mode."""
    if return_mode == "all":
        return times
    elif return_mode == "mean":
        return statistics.mean(times)
    elif return_mode == "median":
        return statistics.median(times)
    raise ValueError(f"Unknown return_mode: {return_mode}")


def _get_empty_cache_for_benchmark(device) -> torch.Tensor:
    """Create a buffer large enough to evict the last-level cache.

    AMD: sized from the vendor device layer rather than from
    ``2 x props.L2_cache_size``. On CDNA4 that property reports the 4 MiB
    per-XCD L2, not the 256 MiB Infinity Cache the working set actually lands
    in, so the old sizing would leave the timed iteration running warm.
    """
    cache_size = device_layer.flush_buffer_bytes(device)
    return torch.empty(int(cache_size), dtype=torch.int8, device=device)


def _clear_cache(cache: torch.Tensor) -> None:
    """Clear the cache buffer by zeroing it."""
    cache.zero_()


def _reset_persisting_l2_cache(
    device: int | str | torch.device | None = None,
) -> None:
    """Reset persisting L2 cache lines to normal status.

    AMD: a no-op -- CDNA exposes no L2-persistence API, so there are no
    persisting lines to reset. See ``device/amd.py``.
    """
    device_layer.reset_persisting_l2_cache(device)


def clone_args(args: list[Any]) -> list[Any]:
    """Clone tensor arguments to prevent cross-iteration data contamination.

    Returns fresh copies of all tensor arguments so each benchmark iteration
    starts with independent data.  Non-tensor arguments are passed through.
    """
    return [arg.clone() if isinstance(arg, torch.Tensor) else arg for arg in args]


def bench_gpu_time_with_cupti(
    fn: Callable,
    warmup: int = 10,
    rep: int = 100,
    setup: Callable[[], Any] | None = None,
    cold_l2_cache: bool = True,
    device="cuda",
):
    """Benchmark GPU time using the discovered user CUPTI activity sequence.

    Setup and cache-management work is excluded by discovering the user-call
    sequence after warmup, then selecting only that sequence from every timed
    iteration. The end timestamp is captured after synchronization so delayed
    or non-default-stream work remains inside the attribution window.

    NVIDIA only. The AMD equivalent is the rocprofiler-sdk shim (task 04); the
    selection logic it will reuse lives in ``solexbench_rocm.activity`` and is
    already vendor-neutral. AMD ships on the event methodology meanwhile.
    """
    # AMD: imported here rather than at module scope so that this module stays
    # importable without the NVIDIA-only CUPTI wheels.
    from cupti import cupti

    from sol_execbench.core.bench.cupti_utils import (
        CuptiKernelInfo,
        collect_cupti_activities,
        kernel_activity_counts,
        kernel_activity_sequence,
        select_activity_sequence,
    )

    GPU_TIMING_ACTIVITY_KINDS = gpu_timing_activity_kinds()

    if setup is None:
        _fn = fn

        def fn(_):
            return _fn()

        def setup():
            return None

    buffer = None
    if cold_l2_cache:
        buffer = _get_empty_cache_for_benchmark(device)

    # Prepare runner (either direct fn or CUDA graph replay)
    runner: Callable = fn
    runner_args: Callable = setup

    def prepare_iteration(*, synchronize: bool = True):
        args = runner_args()
        if cold_l2_cache:
            _reset_persisting_l2_cache(device)
            _clear_cache(buffer)
        if synchronize:
            torch.cuda.synchronize()
        return args

    torch.cuda.synchronize()
    for _ in range(warmup):
        args = prepare_iteration()
        runner(args)
    torch.cuda.synchronize()

    # Discover the user-call GPU activity sequence after setup/cache work drains.
    args = prepare_iteration()
    with collect_cupti_activities(
        activity_kinds=GPU_TIMING_ACTIVITY_KINDS
    ) as discovery_buffers:
        runner(args)
        torch.cuda.synchronize()
    expected_kernels = sorted(
        discovery_buffers.kernels,
        key=lambda kernel: (kernel.start, kernel.end, kernel.correlation_id),
    )
    expected_kernel_names = kernel_activity_sequence(expected_kernels)
    if not expected_kernel_names:
        raise ValueError("No kernel activities recorded during discovery iteration")
    expected_kernel_counts = kernel_activity_counts(expected_kernels)

    iter_timestamps = []
    with collect_cupti_activities(
        activity_kinds=GPU_TIMING_ACTIVITY_KINDS
    ) as cupti_buffers:
        torch.cuda.synchronize()
        for _ in range(rep):
            args = prepare_iteration(synchronize=False)
            start_cpu = cupti.get_timestamp()
            runner(args)
            torch.cuda.synchronize()
            end_cpu = cupti.get_timestamp()
            iter_timestamps.append((start_cpu, end_cpu))
        torch.cuda.synchronize()

    sorted_kernels = sorted(
        cupti_buffers.kernels,
        key=lambda kernel: (kernel.start, kernel.end, kernel.correlation_id),
    )
    kernel_starts = [kernel.start for kernel in sorted_kernels]
    measured_times = []
    for idx, (start_cpu, end_cpu) in enumerate(iter_timestamps):
        left_idx = bisect.bisect_left(kernel_starts, start_cpu)
        right_idx = bisect.bisect_right(kernel_starts, end_cpu)
        window_kernels: list[CuptiKernelInfo] = sorted_kernels[left_idx:right_idx]
        iter_kernels = select_activity_sequence(
            window_kernels,
            expected_kernel_names,
            iteration=idx,
        )
        assert kernel_activity_counts(iter_kernels) == expected_kernel_counts
        min_start = min(k.start for k in iter_kernels)
        max_end = max(k.end for k in iter_kernels)
        measured_times.append((max_end - min_start) / 1e6)
    return measured_times


def bench_time_with_cuda_events(
    fn: Callable[..., Any],
    warmup: int = 10,
    rep: int = 100,
    setup: Callable[[], Any] | None = None,
    device: str = "cuda",
) -> Union[float, list[float]]:
    """Benchmark the runtime of the provided function.

    Derived from triton.testing.do_bench (MIT licence), with fixes from
    sol-bench: explicit synchronization before each start event, L2 cache
    clearing during warmup, and a setup callback for argument cloning.

    Parameters
    ----------
    fn : Callable[..., Any]
        The function to benchmark.  If *setup* is provided, *fn* receives
        the return value of *setup* as its sole argument.
    warmup : int
        Number of warmup iterations (default: 10).
    rep : int
        Number of timed iterations (default: 100).
    setup : Callable[[], Any] | None
        Called before each timed iteration; its return value is passed to *fn*.
        Setup time is **not** included in measurements. This method should only enqueue operations onto the default stream and should not explcitly synchronize.
    device : str
        CUDA device for cache-clearing buffer (default: ``"cuda"``).

    Returns
    -------
    float | list[float]
        Benchmark result(s) in milliseconds.
    """
    cache = _get_empty_cache_for_benchmark(device)
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    torch.cuda.synchronize()

    if setup is None:
        _fn = fn

        def fn(_):
            return _fn()

        def setup():
            return None

    for _ in range(warmup):
        args = setup()
        # always clear cache after setup to prevent data residing in L2
        _reset_persisting_l2_cache(device)
        _clear_cache(cache)
        fn(args)

    # Timed iterations.
    # Avoid synchronizations after warmup and in this hot loop
    # to keep the driver's GPU queue full.
    for i in range(rep):
        args = setup()
        _reset_persisting_l2_cache(device)
        _clear_cache(cache)
        start_events[i].record()
        fn(args)
        end_events[i].record()

    torch.cuda.synchronize()
    measured_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return measured_times


def time_runnable(
    fn: Any,
    inputs: list,
    outputs: list,
    device: str,
    warmup: int = 10,
    rep: int = 100,
    return_mode: Literal["mean", "median", "all"] = "median",
    methodology: Literal["cuda_events", "hip_events", "cupti"] | None = None,
    seed: int = 0,
) -> Union[float, list[float]]:
    """Time the execution of a callable using CUDA events.

    Creates a :class:`ShiftingMemoryPoolAllocator` from *inputs* and *outputs*
    so each timed iteration receives arguments with a unique ``data_ptr``.
    Allocator setup time is excluded from measurements. Crucially, the allocator
    pre-allocates all tensors before the benchmark loop, so the timed region
    is not affected by cudaMalloc times (which increase measured kernel time by 300%).

    Parameters
    ----------
    fn : callable
        The function to benchmark.  Receives unpacked arguments each iteration.
    inputs : list
        Input tensors/scalars as returned by :func:`gen_inputs`.
    outputs : list
        Pre-allocated output tensors for DPS kernels (from
        :func:`allocate_outputs`), or an empty list for non-DPS kernels.
    device : str
        The CUDA device to run the benchmark on (e.g. ``"cuda:0"``).
    warmup : int
        Number of warmup iterations (default: 10).
    rep : int
        Number of timed iterations (default: 100).
    return_mode : {"mean", "median", "all"}
        How to summarize the timing results (default: ``"median"``).
    methodology : {"cuda_events", "hip_events", "cupti"}, optional
        The methodology to use for timing. CUPTI measures actual GPU kernel
        execution time, excluding CPU-side launch overhead, and is what nsys
        uses. ``hip_events`` is the AMD spelling of ``cuda_events`` -- the same
        code path, named distinctly so a trace can never be mistaken for having
        been captured with the other vendor's tooling.

        AMD: defaults to ``hip_events`` on ROCm and ``cupti`` on CUDA, because
        CUPTI does not exist on ROCm. The rocprofiler-sdk equivalent is task 04;
        until it lands, ``methodology`` in the emitted trace is what
        distinguishes the two, so it is recorded rather than assumed.
    seed : int
        Seed for the allocator's randomized pointer-shift sequence.

    Returns
    -------
    float | list[float]
        Benchmark result(s) in milliseconds.
    """
    if methodology is None:
        methodology = (
            "hip_events" if device_layer.detect_vendor() == "amd" else "cupti"
        )
    total_iterations = warmup + rep
    with torch.cuda.device(device):
        if methodology in ("cuda_events", "hip_events"):
            allocator = ShiftingMemoryPoolAllocator(
                inputs, outputs, total_iterations, seed=seed
            )
            try:
                times = bench_time_with_cuda_events(
                    fn=lambda args: fn(*args),
                    warmup=warmup,
                    rep=rep,
                    setup=allocator.get_unique_args,
                    device=device,
                )
            finally:
                del allocator
        elif methodology == "cupti":
            if device_layer.detect_vendor() == "amd":
                raise ValueError(
                    "methodology='cupti' is NVIDIA-only; CUPTI has no ROCm "
                    "build. Use 'hip_events', or the rocprofiler-sdk "
                    "methodology once task 04 lands."
                )
            allocator = ShiftingMemoryPoolAllocator(
                inputs, outputs, total_iterations + 1, seed=seed
            )
            try:
                times = bench_gpu_time_with_cupti(
                    fn=lambda args: fn(*args),
                    warmup=warmup,
                    rep=rep,
                    setup=allocator.get_unique_args,
                    device=device,
                )
            finally:
                del allocator
        else:
            raise ValueError(f"Unknown methodology: {methodology}")
        if not times:
            raise ValueError(f"No timing results for methodology: {methodology}")
        return _summarize_statistics(times, return_mode)
