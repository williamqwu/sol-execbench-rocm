import torch


@torch.compile(fullgraph=True, dynamic=True)
def _compiled(grad_output, dt_with_bias, dt_activated, time_step_min, time_step_max):
    grad = grad_output.float()
    mask = (dt_activated.float() > time_step_min) & (dt_activated.float() < time_step_max)
    grad = grad * mask * torch.sigmoid(dt_with_bias.float())
    return grad.bfloat16(), grad.sum(dim=(0, 1)).bfloat16()


@torch.compile(fullgraph=True, dynamic=True)
def _pointwise_fp32(grad_output, dt_with_bias, dt_activated, time_step_min, time_step_max):
    grad = grad_output.float()
    mask = (dt_activated.float() > time_step_min) & (dt_activated.float() < time_step_max)
    grad = grad * mask * torch.sigmoid(dt_with_bias.float())
    return grad, grad.bfloat16()


@torch.compile(fullgraph=True, dynamic=True)
def _reduce_bias(grad):
    return grad.sum(dim=(0, 1)).bfloat16()


@torch.no_grad()
def run(grad_output, dt_with_bias, dt_activated, time_step_min, time_step_max):
    # Inductor's fused reduction is launch-efficient for small tensors, while
    # PyTorch's multi-stage reduction has much better CU occupancy when large.
    if grad_output.numel() >= 50_000:
        grad, grad_dt = _pointwise_fp32(grad_output, dt_with_bias, dt_activated,
                                       time_step_min, time_step_max)
        return grad_dt, _reduce_bias(grad)
    return _compiled(grad_output, dt_with_bias, dt_activated, time_step_min, time_step_max)
