import torch
import aiter


# The extension owns the hipBLASLt handle used by the explicitly selected
# solutions below.  Initializing it at module load keeps setup out of run().
aiter.hipb_create_extension()


_SOLUTIONS = {
    128: 477124,
    512: 477091,
    2048: 477479,
    4096: 477480,
    7184: 477479,
    8192: 477480,
    17312: 477479,
    39232: 477480,
}

_PADDED_SOLUTIONS = {
    422: (512, 477091),
    1172: (1280, 477479),
    7184: (7424, 477479),
    17312: (17408, 477479),
}


@torch.no_grad()
def run(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    m = hidden_states.numel() // 1280
    if m < 128:
        return torch.matmul(hidden_states, weight.t())

    hidden_2d = hidden_states.reshape(m, 1280)
    padded_config = _PADDED_SOLUTIONS.get(m)
    if padded_config is not None:
        # Complete 256-row tiles are faster than the backend's ragged choices,
        # even after copying this comparatively small operand.
        padded_m, padded_solution = padded_config
        padded = torch.empty(
            (padded_m, 1280),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        padded[:m].copy_(hidden_2d)
        logits_2d = aiter.hipb_mm(weight, padded.t(), padded_solution).t()[:m]
        return logits_2d.view(*hidden_states.shape[:-1], 51866)

    solution = _SOLUTIONS.get(m)
    if solution is None:
        logits_2d = torch.mm(weight, hidden_2d.t()).t()
    else:
        logits_2d = aiter.hipb_mm(weight, hidden_2d.t(), solution).t()
    return logits_2d.view(*hidden_states.shape[:-1], 51866)
