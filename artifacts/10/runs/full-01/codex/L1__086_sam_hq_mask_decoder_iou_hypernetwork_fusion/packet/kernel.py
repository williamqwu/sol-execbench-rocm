import torch


def _linear_relu(x, weight, bias):
    """rocBLAS linear with its native fused ReLU epilogue."""
    shape = x.shape
    x2 = x.view(-1, shape[-1])
    y = torch.ops.aten._addmm_activation.default(
        bias, x2, weight.t(), beta=1, alpha=1, use_gelu=False
    )
    return y.view(*shape[:-1], weight.shape[0])


def _body(
    iou_token_out,
    mask_tokens_out,
    iou_proj_in_weight,
    iou_proj_in_bias,
    iou_hidden_weight,
    iou_hidden_bias,
    iou_proj_out_weight,
    iou_proj_out_bias,
    hyper0_proj_in_weight,
    hyper0_proj_in_bias,
    hyper0_hidden_weight,
    hyper0_hidden_bias,
    hyper0_proj_out_weight,
    hyper0_proj_out_bias,
    hyper1_proj_in_weight,
    hyper1_proj_in_bias,
    hyper1_hidden_weight,
    hyper1_hidden_bias,
    hyper1_proj_out_weight,
    hyper1_proj_out_bias,
    hyper2_proj_in_weight,
    hyper2_proj_in_bias,
    hyper2_hidden_weight,
    hyper2_hidden_bias,
    hyper2_proj_out_weight,
    hyper2_proj_out_bias,
    hyper3_proj_in_weight,
    hyper3_proj_in_bias,
    hyper3_hidden_weight,
    hyper3_hidden_bias,
    hyper3_proj_out_weight,
    hyper3_proj_out_bias,
):
    x = _linear_relu(iou_token_out, iou_proj_in_weight, iou_proj_in_bias)
    x = _linear_relu(x, iou_hidden_weight, iou_hidden_bias)
    iou_scores = torch.nn.functional.linear(x, iou_proj_out_weight, iou_proj_out_bias)

    params = (
        (hyper0_proj_in_weight, hyper0_proj_in_bias, hyper0_hidden_weight,
         hyper0_hidden_bias, hyper0_proj_out_weight, hyper0_proj_out_bias),
        (hyper1_proj_in_weight, hyper1_proj_in_bias, hyper1_hidden_weight,
         hyper1_hidden_bias, hyper1_proj_out_weight, hyper1_proj_out_bias),
        (hyper2_proj_in_weight, hyper2_proj_in_bias, hyper2_hidden_weight,
         hyper2_hidden_bias, hyper2_proj_out_weight, hyper2_proj_out_bias),
        (hyper3_proj_in_weight, hyper3_proj_in_bias, hyper3_hidden_weight,
         hyper3_hidden_bias, hyper3_proj_out_weight, hyper3_proj_out_bias),
    )
    outputs = []
    for token, (w1, b1, w2, b2, w3, b3) in enumerate(params):
        h = _linear_relu(mask_tokens_out[:, :, token, :], w1, b1)
        h = _linear_relu(h, w2, b2)
        outputs.append(torch.nn.functional.linear(h, w3, b3))
    return iou_scores, torch.stack(outputs, dim=2)


# Keeping GEMMs on ATen/rocBLAS is required for the reference's FP32 reduction
# order.  The fused activation op removes ten standalone ReLU kernels, while
# Inductor's graph runner safely handles the harness's changing input storage.
_compiled = torch.compile(
    _body,
    fullgraph=True,
    options={
        "max_autotune_gemm_backends": "ATEN",
        "triton.cudagraphs": True,
    },
)


@torch.no_grad()
def run(*args):
    return _compiled(*args)
