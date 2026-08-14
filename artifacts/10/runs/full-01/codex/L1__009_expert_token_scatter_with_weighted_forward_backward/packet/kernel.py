import torch


def _run(
    grad_output,
    token_indices,
    selected_tokens,
    w1_output,
    gate_output,
    up_output,
    gated_output,
    expert_output,
    selected_weights,
    w1_weight,
    w2_weight,
    w3_weight,
    combine_grad_weights: bool,
    transpose_intermediates: bool,
):
    batch_seq_len = grad_output.shape[0]
    go = grad_output[token_indices]

    # Keep the matrix operands in BF16 so all six contractions use the native
    # MFMA path.  Each contraction still accumulates in FP32 internally.
    grad_expert = go * selected_weights[:, None]
    # Producing the transposed matrix is faster for the tall output shape; the
    # returned view restores the specified [hidden, ffn] layout at no cost.
    grad_w2 = (gated_output.t() @ grad_expert).t()

    if transpose_intermediates:
        # With very small token counts, keep the long FFN axis outermost.  This
        # is the favorable hipBLAS layout for every contraction in the branch
        # and lets the compiler materialize the two derivatives as one tile.
        grad_gated_t = w2_weight.t() @ grad_expert.t()
        z_t = w1_output.t().float()
        sigmoid_t = torch.sigmoid(z_t)
        grad_w1o_t = (
            grad_gated_t.float()
            * up_output.t().float()
            * (sigmoid_t * (1.0 + z_t * (1.0 - sigmoid_t)))
        ).bfloat16()
        grad_up_t = grad_gated_t * gate_output.t()
        grad_both = torch.stack((grad_w1o_t, grad_up_t), dim=0)
        ffn_dim = w1_output.shape[1]
        num_tokens = token_indices.shape[0]
        grad_pair = grad_both.reshape(2 * ffn_dim, num_tokens) @ selected_tokens
        grad_w1 = grad_pair[:ffn_dim]
        grad_w3 = grad_pair[ffn_dim:]
        grad_x = torch.addmm(
            grad_both[1].t() @ w3_weight, grad_both[0].t(), w1_weight
        )
    else:
        grad_gated = grad_expert @ w2_weight
        grad_up = grad_gated * gate_output

        # SiLU's derivative is sensitive enough to retain FP32 pointwise math;
        # round only at the matrix boundary, as the returned gradients do.
        z = w1_output.float()
        sigmoid = torch.sigmoid(z)
        grad_w1o = (
            grad_gated.float()
            * up_output.float()
            * (sigmoid * (1.0 + z * (1.0 - sigmoid)))
        ).bfloat16()

        if combine_grad_weights:
            # Both outer-product gradients share selected_tokens.  Joining
            # their left operands lets hipBLAS produce both outputs in one
            # larger GEMM.  This wins until token-level compute dominates.
            grad_pair = torch.cat((grad_w1o, grad_up), dim=1).t() @ selected_tokens
            ffn_dim = w1_output.shape[1]
            grad_w1 = grad_pair[:ffn_dim]
            grad_w3 = grad_pair[ffn_dim:]
        else:
            grad_w1 = grad_w1o.t() @ selected_tokens
            grad_w3 = grad_up.t() @ selected_tokens
        grad_x = torch.addmm(grad_up @ w3_weight, grad_w1o, w1_weight)

    grad_hidden = torch.zeros_like(grad_output)
    grad_hidden[token_indices] = grad_x
    grad_route = torch.zeros(
        batch_seq_len, dtype=torch.bfloat16, device=grad_output.device
    )
    grad_route[token_indices] = (
        go.float() * expert_output.float()
    ).sum(-1).bfloat16()
    return (
        grad_hidden,
        grad_route,
        grad_w1,
        grad_w2,
        grad_w3,
    )


# Dynamic compilation keeps one graph per contraction strategy and fuses the
# gather/pointwise/scatter chains around the library GEMMs.
_compiled_run = torch.compile(_run, fullgraph=True, dynamic=True)


def run(
    grad_output,
    token_indices,
    selected_tokens,
    w1_output,
    gate_output,
    up_output,
    gated_output,
    expert_output,
    selected_weights,
    w1_weight,
    w2_weight,
    w3_weight,
):
    combine = token_indices.shape[0] <= 1024
    transpose_intermediates = token_indices.shape[0] <= 128
    return _compiled_run(
        grad_output,
        token_indices,
        selected_tokens,
        w1_output,
        gate_output,
        up_output,
        gated_output,
        expert_output,
        selected_weights,
        w1_weight,
        w2_weight,
        w3_weight,
        combine,
        transpose_intermediates,
    )
