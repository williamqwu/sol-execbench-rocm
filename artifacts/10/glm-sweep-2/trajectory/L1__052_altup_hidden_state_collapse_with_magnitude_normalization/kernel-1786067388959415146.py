import torch

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    unembed_proj_1: torch.Tensor,
    unembed_proj_2: torch.Tensor,
    epsilon: float,
):
    dev = hidden_states.device
    # stacked inputs [2, B*S, H] and weights [2, H, H] -> single bmm
    hs1 = hidden_states[1]
    hs2 = hidden_states[2]
    inp = torch.stack([hs1, hs2], dim=0)                  # [2, B, S, H]
    inp2d = inp.reshape(2, -1, hs1.shape[-1])              # [2, B*S, H]
    w = torch.stack([unembed_proj_1, unembed_proj_2], dim=0)  # [2, H, H]
    proj = torch.bmm(inp2d, w)                             # [2, B*S, H]
    p1 = proj[0].to(torch.float32)
    p2 = proj[1].to(torch.float32)

    first = hidden_states[0]
    first_f = first.to(torch.float32)
    target_mag = torch.sqrt(torch.mean(first_f * first_f, dim=-1, keepdim=True))

    out = _fuse(first_f, p1, p2, target_mag, epsilon)
    return out.to(torch.bfloat16)


@torch.compile(mode="max-autotune", dynamic=True)
def _fuse(first_f, p1, p2, target_mag, epsilon):
    mag1 = torch.sqrt(torch.clamp(torch.mean(p1 * p1, dim=-1, keepdim=True), min=epsilon))
    mag2 = torch.sqrt(torch.clamp(torch.mean(p2 * p2, dim=-1, keepdim=True), min=epsilon))
    n1 = p1 * (target_mag / mag1)
    n2 = p2 * (target_mag / mag2)
    return (first_f + n1 + n2) * (1.0 / 3.0)
