import torch


@torch.compile(
    fullgraph=True,
    dynamic=True,
    options={"coordinate_descent_tuning": True},
)
def _rmsnorm(hidden_states, weight):
    return torch.nn.functional.rms_norm(
        hidden_states.float(), (4096,), weight.float(), eps=1e-5
    ).to(hidden_states.dtype)


@torch.no_grad()
def run(hidden_states, weight):
    return _rmsnorm(hidden_states, weight)
