import torch
import triton
import triton.language as tl


@triton.jit
def _gate_inplace_kernel(x, saved, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < n_elements
    value = tl.load(x + offsets, mask=valid)
    active = tl.load(saved + offsets, mask=valid, other=0.0) > 0.0
    tl.store(x + offsets, value * active, mask=valid)


@triton.jit
def _gate4_inplace_kernel(
    x, saved0, saved1, saved2, saved3, n_per_mask: tl.constexpr,
    BLOCK: tl.constexpr, INTERLEAVED: tl.constexpr,
):
    which = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < n_per_mask
    if INTERLEAVED:
        row = offsets // 256
        col = offsets - row * 256
        x_offsets = row * 1024 + which * 256 + col
    else:
        x_offsets = which * n_per_mask + offsets
    value = tl.load(x + x_offsets, mask=valid)
    # Only one of these four masked loads is live for a given program.  Keeping
    # the saved activations as separate pointers avoids materializing a stack.
    saved = tl.load(saved0 + offsets, mask=valid & (which == 0), other=0.0)
    saved += tl.load(saved1 + offsets, mask=valid & (which == 1), other=0.0)
    saved += tl.load(saved2 + offsets, mask=valid & (which == 2), other=0.0)
    saved += tl.load(saved3 + offsets, mask=valid & (which == 3), other=0.0)
    tl.store(x + x_offsets, value * (saved > 0.0), mask=valid)


def _gate_inplace(x, saved):
    n = x.numel()
    _gate_inplace_kernel[(triton.cdiv(n, 256),)](x, saved, n, BLOCK=256)


def _gate4_inplace(x, saved):
    n = x.shape[1] * x.shape[2]
    _gate4_inplace_kernel[(4, triton.cdiv(n, 256))](
        x, saved[0], saved[1], saved[2], saved[3], n, BLOCK=256,
        INTERLEAVED=(x.stride(0) == 256),
    )


@torch.no_grad()
def run(
    grad_iou_scores,
    grad_hyper_weights,
    iou_token_out,
    mask_tokens_out,
    iou_proj_in_weight,
    iou_proj_in_bias,
    iou_hidden_weight,
    iou_hidden_bias,
    iou_proj_out_weight,
    iou_proj_out_bias,
    iou_hidden1,
    iou_hidden1_relu,
    iou_hidden2,
    iou_hidden2_relu,
    hyper_proj_in_weights,
    hyper_proj_in_biases,
    hyper_hidden_weights,
    hyper_hidden_biases,
    hyper_proj_out_weights,
    hyper_proj_out_biases,
    hyper_hidden1_0,
    hyper_hidden1_1,
    hyper_hidden1_2,
    hyper_hidden1_3,
    hyper_hidden1_relu_0,
    hyper_hidden1_relu_1,
    hyper_hidden1_relu_2,
    hyper_hidden1_relu_3,
    hyper_hidden2_0,
    hyper_hidden2_1,
    hyper_hidden2_2,
    hyper_hidden2_3,
    hyper_hidden2_relu_0,
    hyper_hidden2_relu_1,
    hyper_hidden2_relu_2,
    hyper_hidden2_relu_3,
):
    batch_size, point_batch_size, _ = iou_token_out.shape
    m = batch_size * point_batch_size

    grad_iou_scores_flat = grad_iou_scores.reshape(m, 4)
    iou_hidden2_relu_flat = iou_hidden2_relu.reshape(m, 1024)
    grad_iou_proj_out_weight = grad_iou_scores_flat.t() @ iou_hidden2_relu_flat
    grad_iou_proj_out_bias = grad_iou_scores_flat.sum(dim=0)
    grad_iou_hidden2 = (grad_iou_scores_flat @ iou_proj_out_weight).reshape(
        batch_size, point_batch_size, 1024
    )
    _gate_inplace(grad_iou_hidden2, iou_hidden2)

    grad_iou_hidden2_flat = grad_iou_hidden2.reshape(m, 1024)
    iou_hidden1_relu_flat = iou_hidden1_relu.reshape(m, 1024)
    grad_iou_hidden_weight = grad_iou_hidden2_flat.t() @ iou_hidden1_relu_flat
    grad_iou_hidden_bias = grad_iou_hidden2_flat.sum(dim=0)
    grad_iou_hidden1 = (grad_iou_hidden2_flat @ iou_hidden_weight).reshape(
        batch_size, point_batch_size, 1024
    )
    _gate_inplace(grad_iou_hidden1, iou_hidden1)

    grad_iou_hidden1_flat = grad_iou_hidden1.reshape(m, 1024)
    iou_token_out_flat = iou_token_out.reshape(m, 256)
    grad_iou_proj_in_weight = grad_iou_hidden1_flat.t() @ iou_token_out_flat
    grad_iou_proj_in_bias = grad_iou_hidden1_flat.sum(dim=0)
    grad_iou_token_out = (grad_iou_hidden1_flat @ iou_proj_in_weight).reshape(
        batch_size, point_batch_size, 256
    )

    hidden1 = (hyper_hidden1_0, hyper_hidden1_1, hyper_hidden1_2, hyper_hidden1_3)
    hidden1_relu = (
        hyper_hidden1_relu_0,
        hyper_hidden1_relu_1,
        hyper_hidden1_relu_2,
        hyper_hidden1_relu_3,
    )
    hidden2 = (hyper_hidden2_0, hyper_hidden2_1, hyper_hidden2_2, hyper_hidden2_3)
    hidden2_relu = (
        hyper_hidden2_relu_0,
        hyper_hidden2_relu_1,
        hyper_hidden2_relu_2,
        hyper_hidden2_relu_3,
    )

    grad_mask_base = torch.empty(
        (4, m, 256), device=mask_tokens_out.device, dtype=mask_tokens_out.dtype
    )
    grad_hyper_proj_in_weights = torch.empty_like(hyper_proj_in_weights)
    grad_hyper_proj_in_biases = torch.empty_like(hyper_proj_in_biases)
    grad_hyper_hidden_weights = torch.empty_like(hyper_hidden_weights)
    grad_hyper_hidden_biases = torch.empty_like(hyper_hidden_biases)
    grad_hyper_proj_out_weights = torch.empty_like(hyper_proj_out_weights)
    grad_hyper_proj_out_biases = torch.empty_like(hyper_proj_out_biases)
    if m >= 512:
        grad_hidden2_base = torch.empty(
            (m, 4, 256), device=mask_tokens_out.device, dtype=mask_tokens_out.dtype
        )
        grad_hidden2 = grad_hidden2_base.permute(1, 0, 2)
    else:
        grad_hidden2 = torch.empty(
            (4, m, 256), device=mask_tokens_out.device, dtype=mask_tokens_out.dtype
        )
    grad_weights_batch = grad_hyper_weights.reshape(m, 4, 32).permute(1, 0, 2)

    # Output layer backward.  GEMMs write directly into their final output
    # slices, avoiding the reference's four copy kernels per returned family.
    for mask_idx in range(4):
        grad_weights = grad_hyper_weights[:, :, mask_idx, :].reshape(m, 32)
        h2_relu = hidden2_relu[mask_idx].reshape(m, 256)
        torch.mm(
            grad_weights.t(), h2_relu,
            out=grad_hyper_proj_out_weights[mask_idx],
        )
        torch.sum(
            grad_weights, dim=0, out=grad_hyper_proj_out_biases[mask_idx]
        )
    # rocBLAS uses the same reduction order for this K=32 case in batched and
    # unbatched GEMM, so these four activation gradients can share one launch.
    torch.bmm(grad_weights_batch, hyper_proj_out_weights, out=grad_hidden2)
    _gate4_inplace(grad_hidden2, hidden2)

    # Hidden layer backward.
    if m >= 512:
        grad_hidden1_base = torch.empty_like(grad_hidden2_base)
        grad_hidden1 = grad_hidden1_base.permute(1, 0, 2)
    else:
        grad_hidden1 = torch.empty_like(grad_hidden2)
    for mask_idx in range(4):
        h1_relu = hidden1_relu[mask_idx].reshape(m, 256)
        torch.mm(
            grad_hidden2[mask_idx].t(), h1_relu,
            out=grad_hyper_hidden_weights[mask_idx],
        )
        torch.mm(
            grad_hidden2[mask_idx], hyper_hidden_weights[mask_idx],
            out=grad_hidden1[mask_idx],
        )
    _gate4_inplace(grad_hidden1, hidden1)
    # Interleaving masks on the column axis lets one 2-D reduction retain the
    # exact per-column reduction tree used by four separate sums.
    if m >= 512:
        torch.sum(
            grad_hidden2_base.reshape(m, 1024), dim=0,
            out=grad_hyper_hidden_biases.reshape(1024),
        )
    else:
        torch.sum(grad_hidden2, dim=1, out=grad_hyper_hidden_biases)

    # Input layer backward.
    for mask_idx in range(4):
        token = mask_tokens_out[:, :, mask_idx, :].reshape(m, 256)
        torch.mm(
            grad_hidden1[mask_idx].t(), token,
            out=grad_hyper_proj_in_weights[mask_idx],
        )
        torch.mm(
            grad_hidden1[mask_idx], hyper_proj_in_weights[mask_idx],
            out=grad_mask_base[mask_idx],
        )
    if m >= 512:
        torch.sum(
            grad_hidden1_base.reshape(m, 1024), dim=0,
            out=grad_hyper_proj_in_biases.reshape(1024),
        )
    else:
        torch.sum(grad_hidden1, dim=1, out=grad_hyper_proj_in_biases)
    grad_mask_tokens_out = grad_mask_base.permute(1, 0, 2).reshape(
        batch_size, point_batch_size, 4, 256
    )

    return (
        grad_iou_token_out,
        grad_mask_tokens_out,
        grad_iou_proj_in_weight,
        grad_iou_proj_in_bias,
        grad_iou_hidden_weight,
        grad_iou_hidden_bias,
        grad_iou_proj_out_weight,
        grad_iou_proj_out_bias,
        grad_hyper_proj_in_weights,
        grad_hyper_proj_in_biases,
        grad_hyper_hidden_weights,
        grad_hyper_hidden_biases,
        grad_hyper_proj_out_weights,
        grad_hyper_proj_out_biases,
    )
