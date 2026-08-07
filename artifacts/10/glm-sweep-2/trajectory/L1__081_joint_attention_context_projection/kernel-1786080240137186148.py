import torch

@torch.no_grad()
def run(
    image_attention_output: torch.Tensor,
    context_attention_output: torch.Tensor,
    to_out_weight: torch.Tensor,
    to_out_bias: torch.Tensor,
):
    """
    Joint attention context projection.

    Key insight: the reference concatenates [img, ctx] along seq, then
    duplicates along feature dim to get [x, x], then projects with
    W = [W_L | W_R]. Since the input is duplicated, [x,x] @ W^T == x @ (W_L+W_R)^T.
    This halves the reduction dimension and removes both cat ops.
    """
    batch_size = image_attention_output.shape[0]
    image_seq_len = image_attention_output.shape[1]
    context_seq_len = context_attention_output.shape[1]
    inner_dim = image_attention_output.shape[2]

    # Split weight into left and right halves and sum them
    # to_out_weight: (inner_dim, 2*inner_dim)
    w_l = to_out_weight[:, :inner_dim]       # (inner_dim, inner_dim)
    w_r = to_out_weight[:, inner_dim:]       # (inner_dim, inner_dim)
    w_sum = w_l + w_r                         # (inner_dim, inner_dim)
    w_sum_t = w_sum.t()                       # (inner_dim, inner_dim)

    # Project image stream
    projected_image = torch.matmul(image_attention_output, w_sum_t) + to_out_bias

    # Project context stream
    projected_context = torch.matmul(context_attention_output, w_sum_t) + to_out_bias

    return projected_image, projected_context
