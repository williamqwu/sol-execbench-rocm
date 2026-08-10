import torch


@torch.compile(
    fullgraph=True,
    dynamic=True,
    options={"coordinate_descent_tuning": True, "max_autotune": True},
)
def _rmsnorm(hidden_states, weight):
    return torch.nn.functional.rms_norm(
        hidden_states, (4096,), weight, eps=1e-5
    )


@torch.no_grad()
def run(hidden_states, weight):
    return _rmsnorm(hidden_states, weight)
