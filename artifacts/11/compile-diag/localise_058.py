#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage-by-stage eager-vs-compiled localisation for L2__058_mamba2_selective_scan.

Not a measurement runner; produces no timing. The reference is transcribed
verbatim from data/SOL-ExecBench/benchmark/L2/058_mamba2_selective_scan/
reference.py with `_cap(name, tensor)` calls inserted after every intermediate.
The SAME instrumented function is run eager and under torch.compile, so the
per-stage diffs are apples to apples.

Caveat recorded up front: capturing every intermediate as a graph output
forbids some fusions Inductor would otherwise make in the uninstrumented
function, so a stage that diverges here is necessary but the magnitude may not
be identical to the uninstrumented run. The end-to-end output diff is printed
last and is compared against the uninstrumented figure from
scripts/diag_compile_divergence.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/work/scripts/runners")
sys.path.insert(0, "/work/src")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from _common import load_problem, prepare_inputs, exec_reference  # noqa: E402

CAPS: dict[str, torch.Tensor] = {}
ORDER: list[str] = []


def _cap(name, t):
    if name not in CAPS:
        ORDER.append(name)
    CAPS[name] = t
    return t


@torch.no_grad()
def run(
    hidden_states, in_proj_weight, conv1d_weight, conv1d_bias, dt_bias, A_log,
    D, norm_weight, out_proj_weight,
    time_step_limit_min, time_step_limit_max, layer_norm_epsilon,
):
    hidden_size = 8192
    num_heads = 256
    head_dim = 64
    intermediate_size = 16384
    ssm_state_size = 256
    conv_kernel_size = 4
    n_groups = 8
    chunk_size = 128
    groups_time_state_size = n_groups * ssm_state_size
    conv_dim = intermediate_size + 2 * groups_time_state_size

    batch_size, seq_len, _ = hidden_states.shape
    dtype = hidden_states.dtype
    device = hidden_states.device

    projected = _cap("01_projected", torch.matmul(hidden_states, in_proj_weight.t()))

    gate_start = projected.shape[-1] - intermediate_size - conv_dim - num_heads
    gate = _cap("02_gate", projected[..., gate_start:gate_start + intermediate_size])
    hidden_B_C = _cap("03_hBC_slice", projected[..., gate_start + intermediate_size:gate_start + intermediate_size + conv_dim])
    dt = _cap("04_dt_raw", projected[..., -num_heads:])

    hidden_B_C_t = hidden_B_C.transpose(1, 2)
    conv_out = _cap("05_conv_out", F.conv1d(hidden_B_C_t, conv1d_weight, conv1d_bias,
                                            padding=conv_kernel_size - 1, groups=conv_dim)[..., :seq_len])
    hidden_B_C = _cap("06_conv_silu", (conv_out * torch.sigmoid(conv_out)).transpose(1, 2))

    hidden_states_ssm = hidden_B_C[..., :intermediate_size]
    B = hidden_B_C[..., intermediate_size:intermediate_size + groups_time_state_size]
    C = hidden_B_C[..., intermediate_size + groups_time_state_size:]

    dt = _cap("07_dt_softplus_clamp", torch.clamp(F.softplus(dt + dt_bias), time_step_limit_min, time_step_limit_max))

    hidden_states_ssm = _cap("08_x_f32", hidden_states_ssm.view(batch_size, seq_len, num_heads, head_dim).float())
    B = B.view(batch_size, seq_len, n_groups, ssm_state_size).float()
    C = C.view(batch_size, seq_len, n_groups, ssm_state_size).float()
    heads_per_group = num_heads // n_groups
    B = _cap("09_B_f32", B.repeat(1, 1, heads_per_group, 1))
    C = _cap("10_C_f32", C.repeat(1, 1, heads_per_group, 1))

    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size

    def pad_tensor_4d(x, p):
        return F.pad(x, (0, 0, 0, 0, 0, p, 0, 0)) if p > 0 else x

    def pad_tensor_3d(x, p):
        return F.pad(x, (0, 0, 0, p, 0, 0)) if p > 0 else x

    D_residual = _cap("11_D_residual", D.float()[..., None] * pad_tensor_4d(hidden_states_ssm, pad_size))
    hidden_states_ssm = _cap("12_x_times_dt", hidden_states_ssm * dt[..., None])
    A = _cap("13_A", -torch.exp(A_log.float()) * dt)

    hidden_states_ssm_padded = pad_tensor_4d(hidden_states_ssm, pad_size)
    A_padded = pad_tensor_3d(A, pad_size)
    B_padded = pad_tensor_4d(B, pad_size)
    C_padded = pad_tensor_4d(C, pad_size)

    padded_seq_len = hidden_states_ssm_padded.shape[1]
    num_chunks = padded_seq_len // chunk_size

    hidden_states_ssm_chunked = hidden_states_ssm_padded.reshape(batch_size, num_chunks, chunk_size, num_heads, head_dim)
    A_chunked = A_padded.reshape(batch_size, num_chunks, chunk_size, num_heads)
    B_chunked = B_padded.reshape(batch_size, num_chunks, chunk_size, num_heads, ssm_state_size)
    C_chunked = C_padded.reshape(batch_size, num_chunks, chunk_size, num_heads, ssm_state_size)

    A_perm = A_chunked.permute(0, 3, 1, 2)
    A_cumsum = _cap("14_A_cumsum", torch.cumsum(A_perm, dim=-1))

    def segment_sum(x):
        cs = x.size(-1)
        x_expanded = x[..., None].expand(*x.size(), cs)
        mask = torch.tril(torch.ones(cs, cs, device=device, dtype=torch.bool), diagonal=-1)
        x_masked = x_expanded.masked_fill(~mask, 0)
        tensor_segsum = torch.cumsum(x_masked, dim=-2)
        mask_diag = torch.tril(torch.ones(cs, cs, device=device, dtype=torch.bool), diagonal=0)
        return tensor_segsum.masked_fill(~mask_diag, -torch.inf)

    segA = _cap("15_segsum_A", segment_sum(A_perm))
    L = _cap("16_L", torch.exp(segA))
    G = _cap("17_G", torch.einsum('bclhn,bcshn->bclsh', C_chunked, B_chunked))
    L_perm = L.permute(0, 2, 3, 4, 1)
    M = _cap("18_M", G * L_perm)
    Y_diag = _cap("19_Y_diag", torch.einsum('bclsh,bcshd->bclhd', M, hidden_states_ssm_chunked))

    decay_states = _cap("20_decay_states", torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum))
    decay_states_perm = decay_states.permute(0, 2, 3, 1)
    B_decay = _cap("21_B_decay", B_chunked * decay_states_perm[..., None])
    states = _cap("22_states", torch.einsum('bcshd,bcshn->bchdn', hidden_states_ssm_chunked, B_decay))

    previous_states = torch.zeros_like(states[:, :1])
    states_with_prev = torch.cat([previous_states, states], dim=1)
    A_chunk_ends = A_cumsum[:, :, :, -1]
    A_chunk_ends_padded = F.pad(A_chunk_ends, (1, 0))

    def segment_sum_1d(x):
        cs = x.size(-1)
        x_expanded = x[..., None].expand(*x.size(), cs)
        mask = torch.tril(torch.ones(cs, cs, device=device, dtype=torch.bool), diagonal=-1)
        x_masked = x_expanded.masked_fill(~mask, 0)
        tensor_segsum = torch.cumsum(x_masked, dim=-2)
        mask_diag = torch.tril(torch.ones(cs, cs, device=device, dtype=torch.bool), diagonal=0)
        return tensor_segsum.masked_fill(~mask_diag, -torch.inf)

    decay_chunk = _cap("23_decay_chunk", torch.exp(segment_sum_1d(A_chunk_ends_padded)))
    states_with_prev_perm = states_with_prev.permute(0, 2, 1, 3, 4)
    new_states = _cap("24_new_states", torch.einsum('bhcd,bhdin->bhcin', decay_chunk, states_with_prev_perm))
    new_states = new_states[:, :, :-1, :, :]
    states_final = new_states.permute(0, 2, 1, 3, 4)

    state_decay_out = _cap("25_state_decay_out", torch.exp(A_cumsum))
    state_decay_out_perm = state_decay_out.permute(0, 2, 3, 1)
    Y_off = _cap("26_Y_off", torch.einsum('bcshn,bchdn,bcsh->bcshd', C_chunked, states_final, state_decay_out_perm))

    y = _cap("27_y_diag_plus_off", Y_diag + Y_off)
    y = y.reshape(batch_size, padded_seq_len, num_heads, head_dim)
    y = _cap("28_y_plus_Dres", y + D_residual)
    if pad_size > 0:
        y = y[:, :seq_len]
    y = _cap("29_y_bf16", y.reshape(batch_size, seq_len, intermediate_size).to(dtype))

    group_size = intermediate_size // n_groups
    y_grouped = y.view(batch_size, seq_len, n_groups, group_size)
    variance = _cap("30_variance", y_grouped.float().pow(2).mean(dim=-1, keepdim=True))
    y_normed = y_grouped * torch.rsqrt(variance + layer_norm_epsilon)
    y_normed = _cap("31_y_normed", y_normed.view(batch_size, seq_len, intermediate_size).to(dtype))
    y_normed = _cap("32_y_normed_w", y_normed * norm_weight)
    y = _cap("33_y_gated", y_normed * (gate * torch.sigmoid(gate)))
    output = _cap("34_output", torch.matmul(y, out_proj_weight.t()))
    return output


def _stats(a, b):
    x = a.detach().to(torch.float64)
    y = b.detach().to(torch.float64)
    fin = torch.isfinite(x) & torch.isfinite(y)
    d = torch.where(fin, (x - y).abs(), torch.zeros_like(x))
    scale = float(torch.where(fin, y.abs(), torch.zeros_like(y)).max())
    mx = float(d.max())
    nz = int((d > 0).sum())
    return {"max_abs": mx, "ref_absmax": scale,
            "rel_to_scale": (mx / scale) if scale > 0 else 0.0,
            "n_differing": nz, "numel": int(d.numel()),
            "frac_differing": nz / d.numel(), "dtype": str(a.dtype),
            "shape": list(a.shape)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uuid", required=True)
    ap.add_argument("--json-out", required=True)
    a = ap.parse_args()

    prob = Path("/work/data/SOL-ExecBench/benchmark/L2/058_mamba2_selective_scan")
    definition, workloads = load_problem(prob)
    w = [x for x in workloads if x.uuid == a.uuid][0]
    _, ns = exec_reference(definition)

    torch.manual_seed(0)
    ins = prepare_inputs(definition, w, ns, device="cuda:0")

    CAPS.clear(); ORDER.clear()
    out_e = run(*ins)
    torch.cuda.synchronize()
    eager = {k: v.detach().clone() for k, v in CAPS.items()}
    order = list(ORDER)

    CAPS.clear(); ORDER.clear()
    compiled = torch.compile(run, dynamic=False)
    out_c = compiled(*ins)
    torch.cuda.synchronize()
    comp = {k: v.detach().clone() for k, v in CAPS.items()}

    rows = []
    for k in order:
        if k not in comp:
            rows.append({"stage": k, "missing_in_compiled": True})
            continue
        s = _stats(comp[k], eager[k])
        s["stage"] = k
        rows.append(s)
        print(f"{k:24s} {str(s['shape'])[:28]:28s} {s['dtype']:15s} "
              f"maxabs={s['max_abs']:.6e} scale={s['ref_absmax']:.4e} "
              f"rel={s['rel_to_scale']:.3e} frac_diff={s['frac_differing']:.4f}", flush=True)

    doc = {"problem": "L2__058_mamba2_selective_scan", "uuid": a.uuid,
           "axes": dict(w.axes), "torch": torch.__version__,
           "note": "instrumented reference; every intermediate is a graph output",
           "stages": rows,
           "final_output_max_abs": _stats(out_c, out_e)}
    Path(a.json_out).write_text(json.dumps(doc, indent=2))


if __name__ == "__main__":
    main()
