import torch


@torch.no_grad()
def run(hidden_states, q_weight, k_weight, v_weight):
    bsz, q_len, _ = hidden_states.shape
    w = torch.cat([q_weight, k_weight, v_weight], dim=0)
    out = torch.matmul(hidden_states, w.t())
    o = out.view(bsz, q_len, 24, 128)
    return (
        o[:, :, 0:16].transpose(1, 2),
        o[:, :, 16:20].transpose(1, 2),
        o[:, :, 20:24].transpose(1, 2),
    )
