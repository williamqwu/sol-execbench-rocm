# SPDX-License-Identifier: Apache-2.0
"""Keep work on a submission's own streams inside the timing bracket.

``bench_time_with_cuda_events`` brackets each timed iteration with two events
recorded on the *current* stream::

    start_events[i].record()
    fn(args)
    end_events[i].record()

Anything the submission enqueues on a stream of its own is not between them.
The loop deliberately does not synchronize between iterations -- "to keep the
driver's GPU queue full" -- so the default stream carries a deep backlog while
a freshly created side stream is empty. Work launched there runs immediately,
against earlier iterations' default-stream work, and its duration never lands
inside the bracket that paid for it.

Measured on MI350X, GPU 0, this part (`artifacts/11/side-stream-timing-hole.json`):

    a 12.06 ms fp32 GEMM on a side stream, joined at the START of the next
    call, was reported as 0.0069 ms -- 1743x fast.

That is not a shave, it is a bypass, and it is reachable by a pattern that
looks ordinary. `L1__054`'s submitted kernel reaches a mild form of it without
appearing to try: it puts one of three GEMMs on a second stream and joins with
a host-side ``synchronize()``, and reads 0.4773 ms against 0.6777 ms for the
identical single-stream computation -- a 1.42x "speedup" whose real device time
(0.685 ms vs 0.678 ms) is no speedup at all.

The defense is to make the measurement whole rather than to police the pattern.
Streams created after ``install()`` are tracked, and the timing loop joins them
into the current stream before recording the end event, exactly as
``current_stream().wait_stream(s)`` would have done if the submission had used
the joining idiom. A submission that overlaps honestly and joins correctly is
unaffected -- measured at 0.6856 ms in the same run, matching the single-stream
control -- and a submission that creates no stream at all takes no new code
path, so no existing measurement moves.

Neutralizing beats flagging here: a kernel that overlaps its work is a
legitimate kernel, and there is no way to tell the incidental case from the
adversarial one by inspecting the source. What must not happen is that either
one is reported faster than it ran.

Not needed for the rocprofiler path, which already stamps each window's end
after a full synchronize for this exact reason, nor for CUPTI, which attributes
by activity record rather than by stream.
"""

from __future__ import annotations

import weakref
from typing import Any

# Streams constructed after install(). Weak, so a stream the submission drops
# stops being joined rather than keeping a dead handle alive for the run.
_TRACKED: "weakref.WeakSet[Any]" = weakref.WeakSet()
_INSTALLED = False
_JOINS = 0


def install() -> None:
    """Start tracking stream construction. Idempotent; call before user import.

    Wraps ``__new__`` rather than replacing the class: ``torch.cuda.Stream`` is
    a Python subclass of a C-extension type and instances are passed back into
    C++, where a substituted class would fail an ``isinstance`` check somewhere
    far from here. Wrapping leaves the type identity untouched.

    ``__new__`` and not ``__init__``: the C base does its construction in
    ``__new__`` and inherits ``object.__init__``, so wrapping ``__init__``
    forwards the constructor's arguments to ``object.__init__`` and every
    ``torch.cuda.Stream()`` dies with "object.__init__() takes exactly one
    argument". Which it did, on the first run of the corpus.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    import torch

    if not hasattr(torch, "cuda") or not hasattr(torch.cuda, "Stream"):
        return

    cls = torch.cuda.Stream
    orig_new = cls.__new__

    def tracking_new(subclass, *args, **kwargs):
        obj = orig_new(subclass, *args, **kwargs)
        try:
            _TRACKED.add(obj)
        except TypeError:  # not weak-referenceable; nothing to track
            pass
        return obj

    tracking_new._solexbench_wrapped = orig_new  # type: ignore[attr-defined]
    cls.__new__ = tracking_new  # type: ignore[assignment]
    _INSTALLED = True


def tracked_count() -> int:
    """How many live submission-created streams are being tracked."""
    return len(_TRACKED)


def joins_performed() -> int:
    """Total joins issued. Zero means no submission stream was ever seen."""
    return _JOINS


def fence_from_current() -> int:
    """Make every tracked stream wait on the current stream. Returns how many.

    The closing join alone is not enough, and the reason is worth stating
    because the first version of this defense shipped with only the join and
    only half worked.

    A submission that host-synchronizes its own stream *inside* ``run()`` has
    already drained it by the time control returns, so joining at the end waits
    on nothing. The work still happened -- concurrently with the default
    stream's backlog from EARLIER iterations, because the timed loop does not
    synchronize between them -- and it still landed outside every bracket.
    ``L1__054`` measured 0.5007 ms against a true 0.7075 ms with the closing
    join in place.

    Called immediately after the start event is recorded, this ties each
    tracked stream to the default stream's position at the start of THIS
    iteration, so nothing it launches can execute before the bracket opens.
    Together with ``join_into_current`` that is exactly what the
    ``wait_stream`` idiom does, and a kernel already written that way measures
    the same either way (0.7096 vs 0.7012 single-stream, on this part).

    What it costs a legitimate overlapping kernel: cross-iteration overlap,
    which was never that kernel's own speedup -- it is one timed iteration
    borrowing another's wall clock.
    """
    if not _TRACKED:
        return 0

    import torch

    current = torch.cuda.current_stream()
    n = 0
    for stream in list(_TRACKED):
        if stream is None or stream == current:
            continue
        try:
            stream.wait_stream(current)
        except Exception:
            continue
        n += 1
    return n


def join_into_current() -> int:
    """Make the current stream wait on every tracked stream. Returns how many.

    A no-op, and free, when the submission created no stream of its own --
    which is every reference implementation and the overwhelming majority of
    submissions, so the methodology of record does not change for them.
    """
    global _JOINS
    if not _TRACKED:
        return 0

    import torch

    current = torch.cuda.current_stream()
    n = 0
    for stream in list(_TRACKED):
        if stream is None or stream == current:
            continue
        try:
            current.wait_stream(stream)
        except Exception:
            # A stream on another device cannot be joined into this one and is
            # not what this defends against. Skipping is correct; swallowing
            # silently is not, so the count reflects what actually happened.
            continue
        n += 1
    _JOINS += n
    return n
