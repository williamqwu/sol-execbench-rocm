#!/usr/bin/env python3
"""Stage-by-stage eager vs compiled divergence for L1__067."""
import sys, json
from pathlib import Path
sys.path.insert(0, "/work/scripts/runners")
sys.path.insert(0, "/work/src")
import torch
from _common import exec_reference, load_problem, prepare_inputs

PROB = Path("/work/data/SOL-ExecBench/benchmark/L1/067_flash_attention_gqa_ultralong")
UUID = sys.argv[1] if len(sys.argv) > 1 else "b0c05812-9ac0-5ecb-a7a9-73edaf552dde"

SRC = '''
import torch
import torch.nn.functional as F

@torch.no_grad()
def run(hidden_states, cos, sin, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight):
    num_attention_heads = 32
    num_key_value_heads = 8
    head_dim = 128
    num_key_value_groups = 4
    scaling = head_dim ** -0.5
    batch_size, seq_len, _ = hidden_states.shape

    query_states = F.linear(hidden_states, q_proj_weight)
    key_states = F.linear(hidden_states, k_proj_weight)
    value_states = F.linear(hidden_states, v_proj_weight)
    out = {}
    out["01_q_proj"] = query_states
    out["02_k_proj"] = key_states
    out["03_v_proj"] = value_states

    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)

    cos_expanded = cos.unsqueeze(1)
    sin_expanded = sin.unsqueeze(1)

    q1 = query_states[..., : head_dim // 2]
    q2 = query_states[..., head_dim // 2 :]
    query_rotated = torch.cat((-q2, q1), dim=-1)
    query_states = (query_states * cos_expanded) + (query_rotated * sin_expanded)
    out["04_q_rope"] = query_states

    k1 = key_states[..., : head_dim // 2]
    k2 = key_states[..., head_dim // 2 :]
    key_rotated = torch.cat((-k2, k1), dim=-1)
    key_states = (key_states * cos_expanded) + (key_rotated * sin_expanded)
    out["05_k_rope"] = key_states

    key_states = key_states[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
    ).reshape(batch_size, num_attention_heads, seq_len, head_dim)
    value_states = value_states[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
    ).reshape(batch_size, num_attention_heads, seq_len, head_dim)
    out["06_k_rep"] = key_states
    out["07_v_rep"] = value_states

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    out["08_qk_scaled"] = attn_weights

    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float('-inf'), device=hidden_states.device, dtype=hidden_states.dtype),
        diagonal=1)
    attn_weights = attn_weights + causal_mask
    out["09_masked"] = attn_weights

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    out["10_softmax"] = attn_weights

    attn_output = torch.matmul(attn_weights, value_states)
    out["11_av"] = attn_output

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_len, num_attention_heads * head_dim)
    out["12_reshape"] = attn_output

    output = F.linear(attn_output, o_proj_weight)
    out["13_out_proj"] = output
    return out
'''

definition, workloads = load_problem(PROB)
w = [x for x in workloads if x.uuid == UUID][0]
_, ref_ns = exec_reference(definition)

ns = {}
exec(compile(SRC, "<instr>", "exec"), ns)
fn = ns["run"]
cfn = torch.compile(fn, dynamic=False)

torch.manual_seed(0)
ins = prepare_inputs(definition, w, ref_ns, device="cuda:0")
eager = fn(*ins)
torch.cuda.synchronize()
torch.manual_seed(0)
ins2 = prepare_inputs(definition, w, ref_ns, device="cuda:0")
comp = cfn(*ins2)
torch.cuda.synchronize()

# sanity: inputs identical
for i, (a, b) in enumerate(zip(ins, ins2)):
    assert torch.equal(a, b), f"input {i} differs"

EPS = torch.finfo(torch.float32).eps
print(f"uuid={UUID} axes={dict(w.axes)}  fp32 eps={EPS:.6e}")
print(f"{'stage':14s} {'shape':26s} {'max_abs':>12s} {'max_rel':>12s} {'n_diff':>10s} {'frac_diff':>10s}")
for k in sorted(eager):
    a = eager[k].float()
    b = comp[k].float()
    if a.shape != b.shape:
        print(k, "SHAPE MISMATCH", a.shape, b.shape); continue
    d = (a - b).abs()
    finite = torch.isfinite(a) & torch.isfinite(b)
    d = torch.where(finite, d, torch.zeros_like(d))
    denom = a.abs().clamp_min(1e-30)
    rel = (d / denom)
    nd = int((d > 0).sum().item())
    print(f"{k:14s} {str(tuple(a.shape)):26s} {d.max().item():12.4e} "
          f"{rel.max().item():12.4e} {nd:10d} {nd/a.numel():10.4f}")
