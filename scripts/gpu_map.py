#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Map torch CUDA/HIP device indices to amdsmi processor handles.

**These orderings are not the same, and on this node they are badly scrambled.**
Measured on mia1-p02-g10 (8x MI355X):

    torch idx   0   1   2   3   4   5   6   7
    amdsmi idx  3   0   2   1   7   4   6   5

Indexing amdsmi handles with a torch device index therefore samples a
*different physical GPU*. In task 01 that means reading the clock of an idle
GPU while the load runs on another one -- producing a low, stable, entirely
plausible "floor" that is pure fiction. Nothing downstream could detect it.

The mapping is resolved through PCI bus identity, which both libraries agree
on, rather than through position in either list. It also stays correct under
HIP_VISIBLE_DEVICES, because torch still reports the true bus id of whatever
device it exposes as index 0.
"""

from __future__ import annotations

_TORCH_TO_AMDSMI: dict[int, int] | None = None


def _bdf_bus(bdf) -> int | None:
    """Bus number from amdsmi's BDF, which may be a str or a dict."""
    if isinstance(bdf, dict):
        b = bdf.get("bus_number", bdf.get("bus"))
        return int(b) if b is not None else None
    try:
        # "0000:75:00.0" -> 0x75
        return int(str(bdf).split(":")[1], 16)
    except Exception:
        return None


def torch_to_amdsmi() -> dict[int, int]:
    """{torch device index -> amdsmi handle index}, resolved via PCI bus."""
    global _TORCH_TO_AMDSMI
    if _TORCH_TO_AMDSMI is not None:
        return _TORCH_TO_AMDSMI

    import torch
    import amdsmi

    amdsmi.amdsmi_init()
    handles = amdsmi.amdsmi_get_processor_handles()

    by_bus: dict[int, int] = {}
    for i, h in enumerate(handles):
        bus = _bdf_bus(amdsmi.amdsmi_get_gpu_device_bdf(h))
        if bus is not None:
            by_bus[bus] = i

    mapping: dict[int, int] = {}
    for t in range(torch.cuda.device_count()):
        bus = getattr(torch.cuda.get_device_properties(t), "pci_bus_id", None)
        if bus is None:
            raise RuntimeError(
                "torch does not expose pci_bus_id; cannot prove which physical "
                "GPU an index refers to. Refusing to guess -- every clock "
                "sample would be of unknown provenance.")
        if int(bus) not in by_bus:
            raise RuntimeError(
                f"torch device {t} is on PCI bus {int(bus):#x}, which amdsmi "
                f"does not report. Known buses: "
                f"{[hex(b) for b in sorted(by_bus)]}")
        mapping[t] = by_bus[int(bus)]

    _TORCH_TO_AMDSMI = mapping
    return mapping


def amdsmi_handle(torch_index: int):
    """The amdsmi handle for the GPU torch calls *torch_index*."""
    import amdsmi
    amdsmi.amdsmi_init()
    return amdsmi.amdsmi_get_processor_handles()[torch_to_amdsmi()[torch_index]]


def torch_to_drm_card() -> dict[int, str]:
    """{torch device index -> /sys/class/drm/cardN path}, resolved via PCI bus.

    The card numbering is unrelated to every other ordering here: on this node
    card1 is torch 0 and card57 is torch 7. Needed to verify that the clock
    lock landed on the specific GPUs intended.
    """
    import glob
    import re

    import torch

    by_bus: dict[int, str] = {}
    for dev in glob.glob("/sys/class/drm/card*/device"):
        try:
            uevent = open(f"{dev}/uevent").read()
        except OSError:
            continue
        m = re.search(r"PCI_SLOT_NAME=[0-9A-Fa-f]{4}:([0-9A-Fa-f]{2}):", uevent)
        if m:
            by_bus[int(m.group(1), 16)] = dev.rsplit("/device", 1)[0]

    mapping = {}
    for t in range(torch.cuda.device_count()):
        bus = int(getattr(torch.cuda.get_device_properties(t), "pci_bus_id"))
        if bus in by_bus:
            mapping[t] = by_bus[bus]
    return mapping


def torch_to_rocm_smi() -> dict[int, int]:
    """{torch device index -> `rocm-smi -d N` index}, resolved via PCI bus.

    `rocm-smi -d N` is what applies the clock lock, so getting this wrong locks
    a different GPU than the one being timed -- and, as with the amdsmi
    ordering, the result would look entirely reasonable. Parsed from
    `rocm-smi --showbus` rather than assumed equal to the amdsmi order, even
    though on this node the two happen to coincide.
    """
    import re
    import subprocess

    import torch

    out = subprocess.run(["rocm-smi", "--showbus"],
                         capture_output=True, text=True, timeout=60)
    by_bus: dict[int, int] = {}
    for m in re.finditer(r"GPU\[(\d+)\][^\n]*?PCI Bus:\s*([0-9A-Fa-f]{4}:([0-9A-Fa-f]{2}):)",
                         out.stdout):
        by_bus[int(m.group(3), 16)] = int(m.group(1))
    if not by_bus:
        raise RuntimeError(f"could not parse `rocm-smi --showbus`:\n{out.stdout}")

    mapping = {}
    for t in range(torch.cuda.device_count()):
        bus = int(getattr(torch.cuda.get_device_properties(t), "pci_bus_id"))
        if bus not in by_bus:
            raise RuntimeError(
                f"torch device {t} on bus {bus:#x} not reported by rocm-smi")
        mapping[t] = by_bus[bus]
    return mapping


if __name__ == "__main__":
    for label, fn in (("amdsmi", torch_to_amdsmi),
                      ("rocm-smi", torch_to_rocm_smi)):
        try:
            m = fn()
        except Exception as e:
            print(f"torch -> {label}: FAILED: {e}")
            continue
        identity = all(k == v for k, v in m.items())
        print(f"torch -> {label:<9}", {k: m[k] for k in sorted(m)},
              "" if identity else "  <-- indexing by position would be WRONG")
