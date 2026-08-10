#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Is the authoritative card ours alone, right now?

    python scripts/gpu_exclusive.py --gpu 0          # exit 0 clean, 1 dirty

`TODO.md` D29 has been open since a fleet sweep put 34 jobs on GPU 0 while
holding a scheduler reservation on it. On 2026-08-10 the reservation was found
to be weaker still: it binds the fleet's own placer and has no authority at all
over a container somebody started by hand. `miles-lora`, another team's sglang
and Megatron work, runs with `/dev/kfd` and no device restriction, and the
sampling guard caught it on the authoritative card five separate times.

Observing that is not enough. A timing that shared its card produces a
plausible number with nothing on it to say so, and by the time the guard's
artifact is read the run is over. This is the same check at the point where it
can still change the outcome: called before each authoritative measurement, so
a contaminated one is refused rather than published.

Membership is by KFD `gpu_id`, resolved from torch, for the reasons in
`guard_authoritative_gpu.py`: `rocm-smi --showpids`'s `GPU(s)` column is a
count and not an index, and rocm-smi's device order is not torch's.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

KFD_PROC = Path("/sys/class/kfd/kfd/proc")


def authoritative_gpu_id(hip_index: int = 0) -> str | None:
    """The KFD `gpu_id` of HIP device *hip_index*, via its PCI bus id.

    Needs torch, so it only answers inside the measurement container. A caller
    on the host gets None and must pass the id explicitly -- better than a
    host-side guess that silently names the wrong card.
    """
    try:
        import torch
    except Exception:
        return None
    try:
        props = torch.cuda.get_device_properties(hip_index)
        bus = int(getattr(props, "pci_bus_id", -1))
    except Exception:
        return None
    for node in sorted((Path("/sys/class/kfd/kfd/topology/nodes")).glob("*")):
        try:
            gid = (node / "gpu_id").read_text().strip()
            if gid == "0":
                continue
            props_txt = (node / "properties").read_text()
        except OSError:
            continue
        for line in props_txt.splitlines():
            if line.startswith("location_id"):
                # location_id packs bus << 8 | device << 3 | function.
                if (int(line.split()[1]) >> 8) & 0xFF == bus:
                    return gid
    return None


def processes_on(gpu_id: str, exclude: set[int] | None = None) -> list[int]:
    """PIDs holding a queue on *gpu_id*, excluding this process tree."""
    exclude = exclude or set()
    out = []
    if not KFD_PROC.is_dir():
        return out
    for proc in KFD_PROC.iterdir():
        try:
            pid = int(proc.name)
        except ValueError:
            continue
        if pid in exclude:
            continue
        for q in (proc / "queues").glob("*/gpuid"):
            try:
                if q.read_text().strip() == gpu_id:
                    out.append(pid)
                    break
            except OSError:
                continue
    return sorted(out)


def describe(pid: int) -> str:
    try:
        cmd = " ".join(Path(f"/proc/{pid}/cmdline").read_bytes()
                       .decode("utf-8", "replace").split("\0")).strip()
    except OSError:
        cmd = "(gone)"
    cg = "host"
    try:
        for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines():
            tail = line.rsplit(":", 1)[-1]
            if "docker" in tail or "containerd" in tail:
                cg = tail.strip()[-24:]
                break
    except OSError:
        pass
    return f"pid {pid} [{cg}] {cmd[:120]}"


def check(gpu_id: str) -> tuple[bool, list[str]]:
    """`(clean, descriptions)`. Clean means nothing outside this process tree."""
    mine = {os.getpid(), os.getppid()}
    foreign = processes_on(gpu_id, exclude=mine)
    return (not foreign), [describe(p) for p in foreign]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0, help="HIP device index")
    ap.add_argument("--gpu-id", default=None, help="KFD gpu_id, if known")
    a = ap.parse_args()

    gid = a.gpu_id or authoritative_gpu_id(a.gpu)
    if gid is None:
        print("could not resolve the authoritative gpu_id (no torch here?); "
              "pass --gpu-id", file=sys.stderr)
        return 2
    clean, who = check(gid)
    if clean:
        print(f"gpu_id {gid} is exclusively ours")
        return 0
    print(f"gpu_id {gid} is NOT exclusive -- {len(who)} foreign process(es):",
          file=sys.stderr)
    for w in who:
        print(f"  {w}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
