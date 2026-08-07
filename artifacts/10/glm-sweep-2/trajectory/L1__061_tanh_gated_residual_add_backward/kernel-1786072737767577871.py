import torch
import triton
import triton.language as tl


@triton.jit
def _tanh_gated_bwd_kernel(
    grad_output_ptr,
    hidden_states_ptr,
    mask_ptr,
    grad_residual_ptr,
    grad_hidden_states_ptr,
    partial_sum_ptr,
    gate_value: tl.constexpr,
    n_rows,
    hidden_size,
    BLOCK_ROW: tl.constexpr,
    BLOCK_COL: tl.constexpr,
):
    pid = tl.program_id(0)
    num_col_tiles = tl.cdiv(hidden_size, BLOCK_COL)

    row_start = (pid // num_col_tiles) * BLOCK_ROW
    col_start = (pid % num_col_tiles) * BLOCK_COL

    row_offs = row_start + tl.arange(0, BLOCK_ROW)
    col_offs = col_start + tl.arange(0, BLOCK_COL)

    row_mask = row_offs < n_rows
    col_mask = col_offs < hidden_size

    # 2D index grid [BLOCK_ROW, BLOCK_COL]
    ptr_base = row_offs[:, None] * hidden_size + col_offs[None, :]
    valid = row_mask[:, None] & col_mask[None, :]

    go = tl.load(grad_output_ptr + ptr_base, mask=valid, other=0.0)
    hs = tl.load(hidden_states_ptr + ptr_base, mask=valid, other=0.0)
    m = tl.load(mask_ptr + row_offs, mask=row_mask, other=0.0)  # [BLOCK_ROW]

    out_ptr = grad_residual_ptr + ptr_base
    outh_ptr = grad_hidden_states_ptr + ptr_base

    # grad_residual = grad_output
    tl.store(out_ptr, go, mask=valid)

    # grad_hidden_states = grad_output * tanh(gate) * mask
    gh = go.to(tl.float32) * gate_value * m.to(tl.float32)[:, None]
    tl.store(outh_ptr, gh.to(tl.bfloat16), mask=valid)

    # grad_gate partial: masked_hidden_states = hidden_states * mask (bf16), then go.to(f32) * that.to(f32)
    masked_hs = (hs * m[:, None]).to(tl.float32)
    partial = tl.sum(go.to(tl.float32) * masked_hs)
    tl.atomic_add(partial_sum_ptr, partial)


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    gate: torch.Tensor,
    hidden_states: torch.Tensor,
    mask: torch.Tensor,
):
    batch_size, seq_len, hidden_size = grad_output.shape[0], grad_output.shape[1], grad_output.shape[2]
    n_rows = batch_size * seq_len

    gate_float = gate.to(torch.float32)
    gate_value_t = torch.tanh(gate_float)
    sech_squared = 1.0 - gate_value_t * gate_value_t

    grad_residual = torch.empty_like(grad_output)
    grad_hidden_states = torch.empty_like(grad_output)
    partial = torch.zeros(1, dtype=torch.float32, device=grad_output.device)

    gate_value = gate_value_t.item()

    BLOCK_COL = 1024
    BLOCK_ROW = 4
    num_col_tiles = triton.cdiv(hidden_size, BLOCK_COL)
    grid = (triton.cdiv(n_rows, BLOCK_ROW) * num_col_tiles,)

    _tanh_gated_bwd_kernel[grid](
        grad_output,
        hidden_states,
        mask,
        grad_residual,
        grad_hidden_states,
        partial,
        gate_value,
        n_rows,
        hidden_size,
        BLOCK_ROW=BLOCK_ROW,
        BLOCK_COL=BLOCK_COL,
    )

    grad_gate = partial * sech_squared
    return grad_residual, grad_hidden_states, grad_gate.to(torch.bfloat16)
