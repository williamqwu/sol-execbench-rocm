import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _rotary_qk_kernel(
    qkv_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    qk_idx = tl.program_id(1)
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    token_idx = offsets // 1280
    feature_idx = offsets % 1280
    dim_idx = feature_idx % 80
    other_feature = tl.where(dim_idx < 40, feature_idx + 40, feature_idx - 40)

    base = token_idx * 3840 + qk_idx * 1280
    x = tl.load(qkv_ptr + base + feature_idx, mask=mask)
    other = tl.load(qkv_ptr + base + other_feature, mask=mask)
    rotated = tl.where(dim_idx < 40, -other, other)
    c = tl.load(cos_ptr + token_idx * 80 + dim_idx, mask=mask)
    s = tl.load(sin_ptr + token_idx * 80 + dim_idx, mask=mask)
    tl.store(out_ptr + qk_idx * n_elements + offsets, x * c + rotated * s, mask=mask)


def _rotary_qk(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    total_seq_len = qkv.shape[1]
    n_elements = total_seq_len * 1280
    out = torch.empty(
        (2, total_seq_len, 16, 80), device=qkv.device, dtype=qkv.dtype
    )
    _rotary_qk_kernel[(triton.cdiv(n_elements, 256), 2)](
        qkv, cos, sin, out, n_elements=n_elements, BLOCK_SIZE=256, num_warps=4
    )
    return out


_BOUNDARY_STREAM = None
_BOUNDARY_HOST = {}


def _copy_boundaries_async(cu_seqlens):
    global _BOUNDARY_STREAM
    if _BOUNDARY_STREAM is None:
        _BOUNDARY_STREAM = torch.cuda.Stream(device=cu_seqlens.device)
    count = cu_seqlens.numel()
    host = _BOUNDARY_HOST.get(count)
    if host is None:
        host = torch.empty(count, dtype=torch.int64, device="cpu", pin_memory=True)
        _BOUNDARY_HOST[count] = host
    current = torch.cuda.current_stream(cu_seqlens.device)
    _BOUNDARY_STREAM.wait_stream(current)
    with torch.cuda.stream(_BOUNDARY_STREAM):
        host.copy_(cu_seqlens, non_blocking=True)
    return host


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
    proj_weight: torch.Tensor,
    proj_bias: torch.Tensor,
):
    total_seq_len = hidden_states.shape[0]
    num_heads = 16
    head_dim = 80

    bounds_host = _copy_boundaries_async(cu_seqlens)
    qkv = F.linear(hidden_states, qkv_weight, qkv_bias)
    qkv = qkv.reshape(total_seq_len, 3, num_heads, head_dim).permute(1, 0, 2, 3)
    qk_states = _rotary_qk(qkv, cos, sin)
    value_states = qkv[2]
    query_states, key_states = qk_states.unbind(0)

    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)

    _BOUNDARY_STREAM.synchronize()
    bounds = bounds_host.tolist()
    segments = [(start, end) for start, end in zip(bounds, bounds[1:]) if end != start]
    attn_outputs = []
    for start, end in segments:
        q_seq = query_states[:, :, start:end, :]
        k_seq = key_states[:, :, start:end, :]
        v_seq = value_states[:, :, start:end, :]
        weights = torch.matmul(q_seq, k_seq.transpose(2, 3))
        weights.mul_(head_dim ** -0.5)
        weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(hidden_states.dtype)
        attn_outputs.append(torch.matmul(weights, v_seq).transpose(1, 2))

    if attn_outputs:
        attn_output = torch.cat(attn_outputs, dim=1)
    else:
        attn_output = torch.zeros(
            1, 0, num_heads, head_dim,
            device=hidden_states.device, dtype=hidden_states.dtype,
        )
    attn_output = attn_output.reshape(total_seq_len, 1280).contiguous()
    return F.linear(attn_output, proj_weight, proj_bias)
