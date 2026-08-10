import torch
import triton
import triton.language as tl


@triton.jit
def _moe_backward_kernel(
    grad_mask,
    grad_tpe,
    grad_loss,
    training,
    output,
    n_elements: tl.constexpr,
    n_tokens: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < n_elements
    cols = offsets & 255
    base = tl.load(grad_mask + offsets, mask=valid)
    expert = tl.load(grad_tpe + cols, mask=valid)
    loss = tl.load(grad_loss) * tl.load(training).to(tl.float32)
    result = base + expert + loss / (n_tokens * 8.0)
    tl.store(output + offsets, result, mask=valid)


@torch.no_grad()
def run(
    grad_tokens_per_expert: torch.Tensor,
    grad_expert_mask: torch.Tensor,
    grad_load_balance_loss: torch.Tensor,
    topk_idx: torch.Tensor,
    expert_mask: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    training: torch.Tensor,
):
    n_tokens = grad_expert_mask.shape[0]
    n_elements = grad_expert_mask.numel()
    output = torch.empty_like(grad_expert_mask)
    _moe_backward_kernel[(triton.cdiv(n_elements, 4096),)](
        grad_expert_mask,
        grad_tokens_per_expert,
        grad_load_balance_loss,
        training,
        output,
        n_elements=n_elements,
        n_tokens=n_tokens,
        BLOCK=4096,
        num_warps=1,
    )
    return output
