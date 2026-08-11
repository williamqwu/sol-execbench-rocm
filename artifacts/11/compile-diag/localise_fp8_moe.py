import sys, json, torch
sys.path.insert(0, "/work/src")
sys.path.insert(0, "/work/scripts/runners")

SRC = open("/work/data/SOL-ExecBench/benchmark/Quant/004_fp8_moe_expert_linear/reference.py").read()

# Instrumented copy of `run` that returns every intermediate.
INSTR = SRC + '''

@torch.no_grad()
def run_instr(hidden_states, routing_weight, gate_up_weight, down_weight):
    out = {}
    activation_scaler = BlockwiseScaler(ScalingType.BlockWise1x128)
    weight_scaler = BlockwiseScaler(ScalingType.BlockWise128x128)
    gemm_ref = CuBLASRefBlockwiseGemm()

    hidden_fp32 = hidden_states.to(torch.float32);            out["01_hidden_fp32"] = hidden_fp32
    scale_hidden = activation_scaler.compute_scales(hidden_fp32); out["02_scale_hidden"] = scale_hidden
    gate_up_weight_fp32 = gate_up_weight.to(torch.float32)
    gate_up_weight_t = gate_up_weight_fp32.T
    scale_gate_up = weight_scaler.compute_scales(gate_up_weight_t); out["03_scale_gate_up"] = scale_gate_up
    hidden_scaled = activation_scaler.apply_scaling(hidden_fp32, scale_hidden, inverse=False, clamp_to_fp8_range=True)
    out["04_hidden_scaled"] = hidden_scaled
    gate_up_scaled = weight_scaler.apply_scaling(gate_up_weight_t, scale_gate_up, inverse=False, clamp_to_fp8_range=True)
    out["05_gate_up_scaled"] = gate_up_scaled
    hidden_fp8 = hidden_scaled.to(torch.float8_e4m3fn);       out["06_hidden_fp8_bits"] = hidden_fp8.view(torch.uint8)
    gate_up_fp8 = gate_up_scaled.T.to(torch.float8_e4m3fn);   out["07_gate_up_fp8_bits"] = gate_up_fp8.view(torch.uint8)
    scale_gate_up_cublas = scale_gate_up.T.contiguous()

    sa = BlockwiseScaler(ScalingType.BlockWise1x128)
    sb = BlockwiseScaler(ScalingType.BlockWise128x128)
    a_f32 = sa.apply_scaling(hidden_fp8.to(torch.float32), scale_hidden, inverse=True); out["08_a_dequant"] = a_f32
    b_f32 = sb.apply_scaling(gate_up_fp8.to(torch.float32), scale_gate_up_cublas, inverse=True); out["09_b_dequant"] = b_f32
    y = a_f32 @ b_f32.T;                                      out["10_gemm1_f32"] = y
    gate_up_output = y.to(torch.bfloat16);                    out["11_gate_up_bf16"] = gate_up_output

    gate, up = gate_up_output.chunk(2, dim=-1)
    gated_output = F.silu(gate) * up;                         out["12_gated"] = gated_output
    gated_fp32 = gated_output.to(torch.float32)
    scale_gated = activation_scaler.compute_scales(gated_fp32); out["13_scale_gated"] = scale_gated
    down_weight_fp32 = down_weight.to(torch.float32)
    down_weight_t = down_weight_fp32.T
    scale_down = weight_scaler.compute_scales(down_weight_t)
    gated_scaled = activation_scaler.apply_scaling(gated_fp32, scale_gated, inverse=False, clamp_to_fp8_range=True)
    out["14_gated_scaled"] = gated_scaled
    down_scaled = weight_scaler.apply_scaling(down_weight_t, scale_down, inverse=False, clamp_to_fp8_range=True)
    gated_fp8 = gated_scaled.to(torch.float8_e4m3fn);         out["15_gated_fp8_bits"] = gated_fp8.view(torch.uint8)
    down_fp8 = down_scaled.T.to(torch.float8_e4m3fn);         out["16_down_fp8_bits"] = down_fp8.view(torch.uint8)
    scale_down_cublas = scale_down.T.contiguous()
    a2 = sa.apply_scaling(gated_fp8.to(torch.float32), scale_gated, inverse=True)
    b2 = sb.apply_scaling(down_fp8.to(torch.float32), scale_down_cublas, inverse=True)
    y2 = a2 @ b2.T;                                           out["17_gemm2_f32"] = y2
    output = y2.to(torch.bfloat16)
    out["18_final"] = output * routing_weight
    return out
'''

ns = {}
exec(compile(INSTR, "<instr>", "exec"), ns)
eager = ns["run_instr"]
comp = torch.compile(eager, dynamic=False)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
dev = "cuda:0"

def mk():
    torch.manual_seed(0)
    hidden_size, inter = 3584, 2048
    hs = torch.randn(N, hidden_size, dtype=torch.bfloat16, device=dev)
    rw = torch.randn(N, 1, dtype=torch.bfloat16, device=dev)
    gu = torch.randn(inter*2, hidden_size, dtype=torch.bfloat16, device=dev) * (hidden_size ** -0.5)
    dw = torch.randn(hidden_size, inter, dtype=torch.bfloat16, device=dev) * (inter ** -0.5)
    return hs, rw, gu, dw

ins = mk()
oe = eager(*ins); torch.cuda.synchronize()
oc = comp(*ins);  torch.cuda.synchronize()

print(f"{'stage':22s} {'dtype':10s} {'n_diff':>10s} {'frac_diff':>10s} {'max_abs':>12s} {'max_rel':>10s} {'mean_signed':>13s}")
for k in sorted(oe):
    a, b = oe[k], oc[k]
    if a.dtype == torch.uint8:
        d = (a.int() - b.int())
        nd = int((d != 0).sum())
        print(f"{k:22s} {'uint8':10s} {nd:10d} {nd/a.numel():10.6f} {int(d.abs().max()):12d} {'-':>10s} {float(d.float().mean()):13.6f}")
    else:
        x, y = a.float(), b.float()
        d = x - y
        nd = int((d != 0).sum())
        rel = (d.abs() / x.abs().clamp(min=1e-30)).max()
        print(f"{k:22s} {str(a.dtype).replace('torch.',''):10s} {nd:10d} {nd/a.numel():10.6f} {float(d.abs().max()):12.6e} {float(rel):10.3e} {float(d.mean()):13.6e}")
