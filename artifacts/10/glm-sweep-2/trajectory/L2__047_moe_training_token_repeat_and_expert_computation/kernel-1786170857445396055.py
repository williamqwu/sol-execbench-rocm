import torch
import torch.nn.functional as F


def get_inputs(
    axes_and_scalars: dict[str, ...], device: torch.device
) -> dict[str, torch.Tensor]:
    batch_seq_len = axes_and_scalars["batch_seq_len"]
    hidden_size = axes_and_scalars["hidden_size"]
    moe_intermediate_size = axes_and_scalars["moe_intermediate_size"]
    num_experts = axes_and_scalars["num_experts"]
    num_experts_per_tok = axes_and_scalars["num_experts_per_tok"]

    hidden_states = torch.randn(batch_seq_len, hidden_size, device=device, dtype=torch.float16)
    topk_idx = torch.stack([
        torch.randperm(num_experts, device=device)[:num_experts_per_tok]
        for _ in range(batch_seq_len)
    ]).to(torch.int64)
    topk_weight_raw = torch.rand(batch_seq_len, num_experts_per_tok, device=device, dtype=torch.float16)
    topk_weight = topk_weight_raw / topk_weight_raw.sum(dim=-1, keepdim=True)
    expert_gate_projs = torch.randn(
        num_experts, moe_intermediate_size, hidden_size, device=device, dtype=torch.float16
    ) * 0.02
    expert_up_projs = torch.randn(
        num_experts, moe_intermediate_size, hidden_size, device=device, dtype=torch.float16
    ) * 0.02
    expert_down_projs = torch.randn(
        num_experts, hidden_size, moe_intermediate_size, device=device, dtype=torch.float16
    ) * 0.02

    return {
        "hidden_states": hidden_states,
        "topk_idx": topk_idx,
        "topk_weight": topk_weight,
        "expert_gate_projs": expert_gate_projs,
        "expert_up_projs": expert_up_projs,
        "expert_down_projs": expert_down_projs,
    }


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weight: torch.Tensor,
    expert_gate_projs: torch.Tensor,
    expert_up_projs: torch.Tensor,
    expert_down_projs: torch.Tensor,
) -> torch.Tensor:
    batch_seq_len, hidden_size = hidden_states.shape
    num_experts = expert_gate_projs.shape[0]
    num_experts_per_tok = topk_idx.shape[1]
    I = expert_gate_projs.shape[1]
    N = batch_seq_len * num_experts_per_tok
    dev = hidden_states.device

    hidden_states_repeated = hidden_states.unsqueeze(1).expand(-1, num_experts_per_tok, -1).reshape(N, hidden_size)
    flat_topk_idx = topk_idx.view(-1)
    flat_topk_weight = topk_weight.view(-1)

    sorted_idx = torch.argsort(flat_topk_idx, stable=True)
    sorted_expert = flat_topk_idx[sorted_idx]
    sorted_hidden = hidden_states_repeated[sorted_idx]
    sorted_weight = flat_topk_weight[sorted_idx]

    counts = torch.bincount(sorted_expert, minlength=num_experts).to(torch.int64)
    max_tokens = int(counts.max().item())

    offsets = torch.zeros(num_experts + 1, dtype=torch.int64, device=dev)
    offsets[1:] = counts.cumsum(0)
    arange_N = torch.arange(N, device=dev, dtype=torch.int64)
    exp_off = torch.repeat_interleave(offsets[:-1], counts)
    local_pos = arange_N - exp_off
    flat_idx = sorted_expert * max_tokens + local_pos

    padded = torch.zeros(num_experts, max_tokens, hidden_size, device=dev, dtype=hidden_states.dtype)
    padded.view(num_experts * max_tokens, hidden_size).index_copy_(0, flat_idx, sorted_hidden)

    gate = torch.bmm(padded, expert_gate_projs.transpose(1, 2))
    up = torch.bmm(padded, expert_up_projs.transpose(1, 2))
    act = F.silu(gate) * up
    out = torch.bmm(act, expert_down_projs.transpose(1, 2))

    out_flat = out.view(num_experts * max_tokens, hidden_size)
    sorted_out = out_flat[flat_idx]
    sorted_out = sorted_out * sorted_weight.unsqueeze(-1)

    y = torch.empty_like(hidden_states_repeated)
    y[sorted_idx] = sorted_out
    y = y.view(batch_seq_len, num_experts_per_tok, hidden_size)
    output = y.sum(dim=1)
    return output
