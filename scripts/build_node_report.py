#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Emit artifacts/00/node-report.json — the environment record everything cites.

Degrades gracefully: a field that cannot be probed is recorded as null with the
reason, never omitted and never guessed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance import write_artifact  # noqa: E402


def gpu_inventory() -> list[dict]:
    gpus: list[dict] = []
    try:
        import torch
    except ImportError:
        return [{"error": "torch not importable"}]
    try:
        n = torch.cuda.device_count()
    except Exception as e:
        return [{"error": f"device_count failed: {e}"}]

    for i in range(n):
        entry: dict = {"index": i}
        try:
            p = torch.cuda.get_device_properties(i)
            entry.update({
                "name": p.name,
                "arch": getattr(p, "gcnArchName", None),
                "total_memory_gib": round(p.total_memory / 2**30, 1),
                "compute_units": p.multi_processor_count,
            })
        except Exception as e:
            entry["error"] = str(e)
        entry.update(smi_fields(i))
        gpus.append(entry)
    return gpus


def _num(v):
    """amdsmi returns the string 'N/A' for unsupported fields."""
    return v if isinstance(v, (int, float)) else None


def smi_fields(idx: int) -> dict:
    """Power cap, idle power, temperature, clock range. None + reason if absent.

    Measured against MI355X / amdsmi 26.2.1, which differs from the initial
    assumptions in two ways that matter:
      * power_limit is reported in MICROwatts (1400000000), not watts.
      * the EDGE sensor is NOT SUPPORTED; HOTSPOT (== junction, what rocm-smi
        prints) is the one that reads.
    """
    try:
        import amdsmi
        from gpu_map import amdsmi_handle
        amdsmi.amdsmi_init()
        # By PCI identity, not by position: the amdsmi and torch orderings are
        # scrambled relative to each other (see scripts/gpu_map.py), so
        # handles[idx] would attribute one GPU's telemetry to another.
        h = amdsmi_handle(idx)
        power = amdsmi.amdsmi_get_power_info(h)

        cap_uw = _num(power.get("power_limit"))
        temp = None
        temp_sensor = None
        for sensor in ("HOTSPOT", "JUNCTION", "EDGE"):
            st = getattr(amdsmi.AmdSmiTemperatureType, sensor, None)
            if st is None:
                continue
            try:
                temp = amdsmi.amdsmi_get_temp_metric(
                    h, st, amdsmi.AmdSmiTemperatureMetric.CURRENT)
                temp_sensor = sensor
                break
            except Exception:
                continue

        out = {
            "power_cap_w": cap_uw / 1e6 if cap_uw else None,
            "power_now_w": _num(power.get("current_socket_power")),
            "temp_c": _num(temp),
            "temp_sensor": temp_sensor,
            "smi_source": "amdsmi",
        }
        try:
            gfx = amdsmi.amdsmi_get_clock_info(h, amdsmi.AmdSmiClkType.GFX)
            out["sclk_max_mhz"] = _num(gfx.get("max_clk"))
            out["sclk_min_mhz"] = _num(gfx.get("min_clk"))
            out["sclk_now_mhz"] = _num(gfx.get("clk"))
        except Exception as e:
            out["clock_error"] = str(e)
        return out
    except Exception as e:
        return {"power_cap_w": None, "power_now_w": None, "temp_c": None,
                "smi_source": None, "smi_error": str(e)}


def rocm_version() -> str | None:
    for p in ("/opt/rocm/.info/version", "/opt/rocm/.info/version-dev"):
        try:
            return Path(p).read_text().strip()
        except Exception:
            continue
    return None


def driver_version() -> str | None:
    try:
        return Path("/sys/module/amdgpu/version").read_text().strip()
    except Exception:
        return None


def dataset_census(data_dir: Path) -> dict:
    expected = {"L1": 94, "L2": 82, "Quant": 33, "FlashInfer-Bench": 26}
    if not data_dir.exists():
        return {"present": False, "expected": expected}
    found = {}
    for cat in expected:
        found[cat] = len([
            p for p in data_dir.rglob(f"{cat}/*/definition.json")
        ])
    return {"present": True, "found": found, "expected": expected,
            "matches_expected": found == expected}


def tests_green(root: Path) -> bool | None:
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"],
                           cwd=root, capture_output=True, text=True, timeout=600)
        return r.returncode == 0
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/00/node-report.json")
    a = ap.parse_args()
    root = Path(__file__).resolve().parent.parent

    gpus = gpu_inventory()
    payload = {
        "gpus": gpus,
        "gpu_count": len(gpus),
        "rocm_version": rocm_version(),
        "driver_version": driver_version(),
        "dataset": dataset_census(root / "data"),
        "tests_green": tests_green(root),
    }
    p = write_artifact(a.out, "00-node-acceptance", payload)
    print(f"wrote {p}")

    # Loud, immediate feedback on the two things that matter most.
    if len(gpus) != 8:
        print(f"WARNING: expected 8 GPUs, found {len(gpus)}. Record why in "
              f"STATE.md before proceeding.", file=sys.stderr)
    # torch reports the arch with its feature flags appended
    # ("gfx950:sramecc+:xnack-"), so match on the base target, not equality.
    archs = {g.get("arch") for g in gpus if g.get("arch")}
    if archs and not all("gfx950" in a for a in archs):
        print(f"WARNING: unexpected arch(s): {archs}", file=sys.stderr)

    for field, label in (("power_cap_w", "power cap"),
                         ("sclk_max_mhz", "max GFX clock")):
        vals = [g.get(field) for g in gpus if g.get(field)]
        if vals and len(set(vals)) > 1:
            print(f"WARNING: {label} differs across GPUs: {sorted(set(vals))} "
                  f"— a non-uniform node produces non-comparable timings",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
