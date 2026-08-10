import torch


_input_stream = torch.cuda.Stream()
_weight_stream = torch.cuda.Stream()


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    reshaped: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len, hidden_size = grad_output.shape
    grad_output_2d = grad_output.reshape(-1, hidden_size)
    reshaped_2d = reshaped.reshape(-1, hidden_size)

    caller_stream = torch.cuda.current_stream()
    _input_stream.wait_stream(caller_stream)
    _weight_stream.wait_stream(caller_stream)
    with torch.cuda.stream(_input_stream):
        grad_reshaped = grad_output_2d.mm(weight)
    with torch.cuda.stream(_weight_stream):
        grad_weight = grad_output_2d.t().mm(reshaped_2d)
    caller_stream.wait_stream(_input_stream)
    caller_stream.wait_stream(_weight_stream)
    grad_attn_output = (
        grad_reshaped.reshape(batch_size, seq_len, 32, 64)
        .transpose(1, 2)
    )
    return grad_attn_output, grad_weight
