import torch
import triton
import triton.language as tl


@triton.jit
def _attn_bwd_kernel(Q, K, V, dO, CU, GQKV, OUT, T: tl.constexpr, BLOCK: tl.constexpr):
    chunk = tl.program_id(0)
    head = tl.program_id(1)
    start = tl.load(CU + chunk)
    end = tl.load(CU + chunk + 1)
    length = end - start

    offs = tl.arange(0, BLOCK)
    d = tl.arange(0, 64)
    token = start + offs
    mask_td = (offs[:, None] < length) & (d[None, :] < 64)
    head_base = head * T * 64

    q = tl.load(Q + head_base + token[:, None] * 64 + d[None, :], mask=mask_td, other=0.0).to(tl.float32)
    k = tl.load(K + head_base + token[:, None] * 64 + d[None, :], mask=mask_td, other=0.0).to(tl.float32)
    v = tl.load(V + head_base + token[:, None] * 64 + d[None, :], mask=mask_td, other=0.0).to(tl.float32)
    grad_out = tl.load(dO + token[:, None] * 1280 + head * 64 + d[None, :], mask=mask_td, other=0.0).to(tl.float32)

    scores = tl.dot(q, tl.trans(k), input_precision="ieee") * 0.125
    cols = tl.arange(0, BLOCK)
    valid = (offs[:, None] < length) & (cols[None, :] < length)
    scores = tl.where(valid, scores, -3.4028234663852886e38)
    row_max = tl.max(scores, axis=1)
    probs = tl.exp(scores - row_max[:, None])
    probs = probs / tl.sum(probs, axis=1)[:, None]

    attn_out = tl.dot(probs, v, input_precision="ieee")
    grad_v = tl.dot(tl.trans(probs), grad_out, input_precision="ieee")
    grad_probs = tl.dot(grad_out, tl.trans(v), input_precision="ieee")
    softmax_delta = tl.sum(probs * grad_probs, axis=1)
    grad_scores = probs * (grad_probs - softmax_delta[:, None])
    grad_q = tl.dot(grad_scores, k, input_precision="ieee") * 0.125
    grad_k = tl.dot(tl.trans(grad_scores), q, input_precision="ieee") * 0.125

    out_offsets = token[:, None] * 1280 + head * 64 + d[None, :]
    qkv_offsets = token[:, None] * 3840 + head * 64 + d[None, :]
    tl.store(OUT + out_offsets, attn_out, mask=mask_td)
    tl.store(GQKV + qkv_offsets, grad_q, mask=mask_td)
    tl.store(GQKV + qkv_offsets + 1280, grad_k, mask=mask_td)
    tl.store(GQKV + qkv_offsets + 2560, grad_v, mask=mask_td)


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    hidden_states: torch.Tensor,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    out_weight: torch.Tensor,
):
    total_seq_len = hidden_states.shape[0]
    grad_output_f32 = grad_output.to(torch.float32)
    grad_attn_output = torch.matmul(grad_output_f32, out_weight.to(torch.float32))

    grad_qkv = torch.empty((total_seq_len, 3840), device=hidden_states.device, dtype=torch.float32)
    attn_output = torch.empty((total_seq_len, 1280), device=hidden_states.device, dtype=torch.bfloat16)
    num_chunks = cu_seqlens.numel() - 1
    if total_seq_len == num_chunks * 64:
        _attn_bwd_kernel[(num_chunks, 20)](
            query_states,
            key_states,
            value_states,
            grad_attn_output,
            cu_seqlens,
            grad_qkv,
            attn_output,
            total_seq_len,
            BLOCK=64,
            num_warps=4,
        )
    elif total_seq_len <= 900:
        _attn_bwd_kernel[(num_chunks, 20)](
            query_states,
            key_states,
            value_states,
            grad_attn_output,
            cu_seqlens,
            grad_qkv,
            attn_output,
            total_seq_len,
            BLOCK=128,
            num_warps=4,
        )
    else:
        _attn_bwd_kernel[(num_chunks, 20)](
            query_states,
            key_states,
            value_states,
            grad_attn_output,
            cu_seqlens,
            grad_qkv,
            attn_output,
            total_seq_len,
            BLOCK=128,
            num_warps=8,
        )

    grad_qkv_weight = torch.matmul(grad_qkv.to(torch.bfloat16).t(), hidden_states)
    grad_qkv_bias = grad_qkv.sum(dim=0)
    qkv_weight_f32 = torch.cat((q_weight, k_weight, v_weight), dim=0).to(torch.float32)
    grad_hidden_states = torch.matmul(grad_qkv, qkv_weight_f32)
    grad_out_weight = torch.matmul(grad_output.t(), attn_output)
    grad_out_bias = grad_output_f32.sum(dim=0)

    grad_q_weight, grad_k_weight, grad_v_weight = grad_qkv_weight.split(1280, dim=0)
    grad_q_bias, grad_k_bias, grad_v_bias = grad_qkv_bias.split(1280, dim=0)

    return (
        grad_hidden_states.to(torch.bfloat16),
        grad_q_weight.to(torch.bfloat16),
        grad_q_bias.to(torch.bfloat16),
        grad_k_weight.to(torch.bfloat16),
        grad_k_bias.to(torch.bfloat16),
        grad_v_weight.to(torch.bfloat16),
        grad_v_bias.to(torch.bfloat16),
        grad_out_weight.to(torch.bfloat16),
        grad_out_bias.to(torch.bfloat16),
    )
