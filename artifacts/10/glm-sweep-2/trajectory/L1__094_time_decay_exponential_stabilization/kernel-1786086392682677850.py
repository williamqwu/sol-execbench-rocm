import torch
import triton
import triton.language as tl


@triton.jit
def _full_kernel(
    out_ptr,          # [B, S, H]
    max_state_ptr,    # [B, H]
    num_state_ptr,    # [B, H]
    den_state_ptr,    # [B, H]
    key_ptr,          # [B, S, H]
    value_ptr,        # [B, S, H]
    time_decay_exp_ptr,  # [H]
    time_first_ptr,      # [H]
    ob_b,   # output batch stride = S*H
    kb_b,   # key batch stride = S*H
    ks_step,# key/value stride per timestep = H
    sb_b,   # state batch stride = H
    S,      # seq_len
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)   # block over H
    pid_b = tl.program_id(1)  # batch index

    offs_h = pid * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H

    tde = tl.load(time_decay_exp_ptr + offs_h, mask=mask_h, other=0.0)
    tf = tl.load(time_first_ptr + offs_h, mask=mask_h, other=0.0)

    s_off = pid_b * sb_b + offs_h
    ms = tl.load(max_state_ptr + s_off, mask=mask_h, other=0.0)
    ns = tl.load(num_state_ptr + s_off, mask=mask_h, other=0.0)
    ds = tl.load(den_state_ptr + s_off, mask=mask_h, other=0.0)

    kv_base = pid_b * kb_b
    o_base = pid_b * ob_b

    for t in range(S):
        kv_off = kv_base + t * ks_step + offs_h
        ck = tl.load(key_ptr + kv_off, mask=mask_h, other=0.0)
        cv = tl.load(value_ptr + kv_off, mask=mask_h, other=0.0)

        ck_tf = ck + tf
        max_out = tl.maximum(ms, ck_tf)
        e1o = tl.exp(ms - max_out)
        e2o = tl.exp(ck_tf - max_out)
        num = e1o * ns + e2o * cv
        den = e1o * ds + e2o
        out_val = num / den

        o_off = o_base + t * ks_step + offs_h
        tl.store(out_ptr + o_off, out_val, mask=mask_h)

        ms_tde = ms + tde
        max_st = tl.maximum(ms_tde, ck)
        e1s = tl.exp(ms_tde - max_st)
        e2s = tl.exp(ck - max_st)
        ns = e1s * ns + e2s * cv
        ds = e1s * ds + e2s
        ms = max_st

    tl.store(max_state_ptr + s_off, ms, mask=mask_h)
    tl.store(num_state_ptr + s_off, ns, mask=mask_h)
    tl.store(den_state_ptr + s_off, ds, mask=mask_h)


def _choose_block_h(batch_size, hidden_size):
    # Target ~2 waves of blocks to fill 256 CUs.
    target_blocks = 512
    blocks_per_batch = max(1, target_blocks // batch_size)
    block_h = max(32, hidden_size // blocks_per_batch)
    # round down to power of 2
    bh = 1
    while bh * 2 <= block_h:
        bh *= 2
    # clamp
    bh = max(32, min(bh, 512))
    return bh


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

    BLOCK_H = _choose_block_h(batch_size, hidden_size)
    grid_h = triton.cdiv(hidden_size, BLOCK_H)

    _full_kernel[(grid_h, batch_size)](
        output,
        max_state,
        num_state,
        den_state,
        key,
        value,
        time_decay_exp,
        time_first,
        output.stride(0),
        key.stride(0),
        hidden_size,
        hidden_size,
        seq_len,
        H=hidden_size,
        BLOCK_H=BLOCK_H,
    )

    return output, max_state, num_state, den_state
