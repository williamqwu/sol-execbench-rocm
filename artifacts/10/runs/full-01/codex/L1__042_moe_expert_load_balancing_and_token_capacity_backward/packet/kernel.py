import torch
import triton
import triton.language as tl


@triton.jit
def _moe_backward_kernel(
    grad_tokens_per_expert,
    grad_expert_mask,
    grad_load_balance_loss,
    training,
    output,
    n_elements: tl.constexpr,
    batch_seq_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < n_elements
    expert = offsets & 255

    mask_grad = tl.load(grad_expert_mask + offsets, mask=valid)
    token_grad = tl.load(grad_tokens_per_expert + expert, mask=valid)
    loss_grad = tl.load(grad_load_balance_loss)
    is_training = tl.load(training)
    loss_contribution = tl.where(
        is_training,
        loss_grad / (batch_seq_len * 8.0),
        0.0,
    )
    accumulated_token_grad = token_grad + loss_contribution
    result = mask_grad + accumulated_token_grad
    tl.store(output + offsets, result, mask=valid)


def run(
    grad_tokens_per_expert: torch.Tensor,
    grad_expert_mask: torch.Tensor,
    grad_load_balance_loss: torch.Tensor,
    topk_idx: torch.Tensor,
    expert_mask: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    training: torch.Tensor,
):
    batch_seq_len = grad_expert_mask.shape[0]
    n_elements = grad_expert_mask.numel()
    output = torch.empty_like(grad_expert_mask)
    _moe_backward_kernel[(triton.cdiv(n_elements, 4096),)](
        grad_tokens_per_expert,
        grad_expert_mask,
        grad_load_balance_loss,
        training,
        output,
        n_elements=n_elements,
        batch_seq_len=batch_seq_len,
        BLOCK_SIZE=4096,
        num_warps=8,
    )
    return output
