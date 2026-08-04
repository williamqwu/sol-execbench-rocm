import torch, math, sys
import torch.nn.functional as F
sys.path.insert(0, '/var/tmp/solbench/agent/pilot8/L2__069_joint_transformer_block_residual_path')
import reference as R

dev = torch.device('cuda')


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def run_half(hidden_states, encoder_hidden_states, temb, norm1_weight, norm1_bias,
             norm1_context_weight, norm1_context_bias, to_q_weight, to_q_bias,
             to_k_weight, to_k_bias, to_v_weight, to_v_bias, add_q_proj_weight,
             add_q_proj_bias, add_k_proj_weight, add_k_proj_bias, add_v_proj_weight,
             add_v_proj_bias, to_out_weight, to_out_bias, to_add_out_weight,
             to_add_out_bias, ff_linear1_weight, ff_linear1_bias, ff_linear2_weight,
             ff_linear2_bias, ff_context_linear1_weight, ff_context_linear1_bias,
             ff_context_linear2_weight, ff_context_linear2_bias, DT=torch.float16):
    batch_size = hidden_states.shape[0]
    S = hidden_states.shape[1]
    C = encoder_hidden_states.shape[1]
    dim = 1536; context_dim = 1152; num_heads = 24; head_dim = 64
    scale = head_dim ** -0.5

    nh = F.layer_norm(hidden_states, (dim,), eps=1e-6)
    ne = F.layer_norm(encoder_hidden_states, (context_dim,), eps=1e-6)
    temb_silu = F.silu(temb)
    mod = F.linear(temb_silu, norm1_weight, norm1_bias)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)
    modc = F.linear(temb_silu, norm1_context_weight, norm1_context_bias)
    c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = modc.chunk(6, dim=-1)

    nh = (nh * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)).to(DT)
    ne = (ne * (1 + c_scale_msa.unsqueeze(1)) + c_shift_msa.unsqueeze(1)).to(DT)

    qkvw = torch.cat([to_q_weight, to_k_weight, to_v_weight], 0).to(DT)
    qkvb = torch.cat([to_q_bias, to_k_bias, to_v_bias], 0).to(DT)
    aw = torch.cat([add_q_proj_weight, add_k_proj_weight, add_v_proj_weight], 0).to(DT)
    ab = torch.cat([add_q_proj_bias, add_k_proj_bias, add_v_proj_bias], 0).to(DT)

    qkv = F.linear(nh, qkvw, qkvb)
    cqkv = F.linear(ne, aw, ab)
    T = S + C
    joint = torch.cat([qkv, cqkv], 1).view(batch_size, T, 3, num_heads, head_dim)
    q = joint[:, :, 0].transpose(1, 2)
    k = joint[:, :, 1].transpose(1, 2)
    v = joint[:, :, 2].transpose(1, 2)
    attn = F.scaled_dot_product_attention(q, k, v, scale=scale)
    attn = attn.transpose(1, 2).reshape(batch_size, T, dim)
    ai, ac = attn.split([S, C], dim=1)
    ai = F.linear(ai, to_out_weight.to(DT), to_out_bias.to(DT)).float()
    ac = F.linear(ac, to_add_out_weight.to(DT), to_add_out_bias.to(DT)).float()

    hidden_states = hidden_states + gate_msa.unsqueeze(1) * ai
    encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * ac

    nh = F.layer_norm(hidden_states, (dim,), eps=1e-6)
    ne = F.layer_norm(encoder_hidden_states, (context_dim,), eps=1e-6)
    nh = (nh * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)).to(DT)
    ne = (ne * (1 + c_scale_mlp.unsqueeze(1)) + c_shift_mlp.unsqueeze(1)).to(DT)

    h = F.linear(nh, ff_linear1_weight.to(DT), ff_linear1_bias.to(DT))
    h = F.gelu(h.float(), approximate='tanh').to(DT)
    ff = F.linear(h, ff_linear2_weight.to(DT), ff_linear2_bias.to(DT)).float()
    hc = F.linear(ne, ff_context_linear1_weight.to(DT), ff_context_linear1_bias.to(DT))
    hc = F.gelu(hc.float(), approximate='tanh').to(DT)
    ffc = F.linear(hc, ff_context_linear2_weight.to(DT), ff_context_linear2_bias.to(DT)).float()

    hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff
    encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * ffc
    return encoder_hidden_states, hidden_states


shapes = [(1, 1024, 77), (1, 8192, 77), (32, 256, 77), (4, 2048, 77)]
tols = {1024: 0.015, 8192: 0.03, 256: 0.021, 2048: 0.024}
for b, s_, c in shapes:
    inp = R.get_inputs({'batch_size': b, 'seq_len': s_, 'context_len': c}, dev)
    ref = R.run(**inp)
    for DT in (torch.float16, torch.bfloat16):
        out = run_half(**inp, DT=DT)
        atol = 0.015
        for i, (o, r) in enumerate(zip(out, ref)):
            err = (o - r).abs()
            frac = (err > atol).float().mean().item()
            print(f"  b{b} s{s_} {DT} out{i}: maxerr {err.max().item():.5f} frac>0.015 {frac:.5f}")
    tr = bench(lambda: R.run(**inp))
    th = bench(lambda: run_half(**inp))
    print(f"b={b} s={s_}: ref {tr:.3f} half {th:.3f}  speedup {tr/th:.2f}x")
