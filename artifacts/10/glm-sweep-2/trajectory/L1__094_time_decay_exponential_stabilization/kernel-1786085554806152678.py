import torch
import triton
import triton.language as tl


@triton.jit
def _timestep_kernel(
    # output write
    out_ptr,  # [B, S, H]
    # state ptrs (read+write), [B, H]
    max_state_ptr,
    num_state_ptr,
    den_state_ptr,
    # per-timestep inputs
    key_ptr,   # [B, S, H]
    value_ptr, # [B, S, H]
    # per-channel params [H]
    time_decay_exp_ptr,
    time_first_ptr,
    # strides
    ob_s,  # output stride over seq dim
    kb_s,  # key stride over batch
    ks_s,  # key stride over seq
    sb_h,  # state stride over hidden (==1, contiguous)
    sb_b,  # state stride over batch
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)  # block index over H
    pid_b = tl.program_id(1)  # batch index

    offs_h = pid * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H

    # load per-channel params
    tde = tl.load(time_decay_exp_ptr + offs_h, mask=mask_h, other=0.0)
    tf = tl.load(time_first_ptr + offs_h, mask=mask_h, other=0.0)

    # load states for this batch
    s_off = pid_b * sb_b + offs_h
    ms = tl.load(max_state_ptr + s_off, mask=mask_h, other=0.0)
    ns = tl.load(num_state_ptr + s_off, mask=mask_h, other=0.0)
    ds = tl.load(den_state_ptr + s_off, mask=mask_h, other=0.0)

    # load key/value for this timestep and batch
    kv_off = pid_b * kb_s + ks_s + offs_h
    ck = tl.load(key_ptr + kv_off, mask=mask_h, other=0.0)
    cv = tl.load(value_ptr + kv_off, mask=mask_h, other=0.0)

    # === Output computation ===
    ck_tf = ck + tf
    max_out = tl.maximum(ms, ck_tf)
    e1o = tl.exp(ms - max_out)
    e2o = tl.exp(ck_tf - max_out)
    num = e1o * ns + e2o * cv
    den = e1o * ds + e2o
    out_val = num / den

    # write output
    o_off = pid_b * ob_s + ks_s + offs_h
    tl.store(out_ptr + o_off, out_val, mask=mask_h)

    # === State update ===
    ms_tde = ms + tde
    max_st = tl.maximum(ms_tde, ck)
    e1s = tl.exp(ms_tde - max_st)
    e2s = tl.exp(ck - max_st)
    ns_new = e1s * ns + e2s * cv
    ds_new = e1s * ds + e2s

    # write states
    tl.store(max_state_ptr + s_off, max_st, mask=mask_h)
    tl.store(num_state_ptr + s_off, ns_new, mask=mask_h)
    tl.store(den_state_ptr + s_off, ds_new, mask=mask_h)


@torch.no_grad()
def run(
    time_decay: torch.Tensor,
    key: torch.Tensor,
    time_first: torch.Tensor,
    value: torch.Tensor,
    max_state: torch.Tensor,
    num_state: torch.Tensor,
    den_state: torch.Tensor,
):
    batch_size, seq_len, hidden_size = key.size()

    max_state = max_state.clone().float()
    num_state = num_state.clone().float()
    den_state = den_state.clone().float()

    time_decay_exp = -torch.exp(time_decay.float())

    output = torch.zeros_like(key, dtype=torch.float32)

    BLOCK_H = 2048
    grid_h = triton.cdiv(hidden_size, BLOCK_H)

    for t in range(seq_len):
        _timestep_kernel[(grid_h, batch_size)](
            output,
            max_state,
            num_state,
            den_state,
            key,
            value,
            time_decay_exp,
            time_first,
            output.stride(1),
            key.stride(0),
            key.stride(1) * t,
            1,
            hidden_size,
            H=hidden_size,
            BLOCK_H=BLOCK_H,
        )

    return output, max_state, num_state, den_state
