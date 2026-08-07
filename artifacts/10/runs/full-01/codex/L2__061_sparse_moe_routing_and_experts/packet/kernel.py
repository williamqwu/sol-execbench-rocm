import torch
import torch.nn.functional as F
import triton.language as tl

from aiter.ops.triton.moe.moe_align_block_size import moe_align_block_size_triton
from aiter.ops.triton.moe.moe_op import fused_moe


_TOP_K = 10
_NUM_EXPERTS = 512
_BLOCK_M = 32
_MOE_CONFIG = {
    "BLOCK_SIZE_M": _BLOCK_M,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 8,
    "num_warps": 4,
    "num_stages": 2,
}


@torch.no_grad()
def run(
    hidden_states,
    router_weight,
    expert_gate_proj,
    expert_up_proj,
    expert_down_proj,
    shared_gate_proj,
    shared_up_proj,
    shared_down_proj,
    shared_expert_gate_weight,
):
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states_flat = hidden_states.view(-1, hidden_dim)

    router_logits = torch.matmul(hidden_states_flat, router_weight.t())
    routing_weights = F.softmax(router_logits.float(), dim=1)
    routing_weights_topk, selected_experts = torch.topk(
        routing_weights, _TOP_K, dim=-1
    )
    routing_weights_topk = routing_weights_topk / routing_weights_topk.sum(
        dim=-1, keepdim=True
    )
    routing_weights_topk = routing_weights_topk.to(torch.bfloat16)

    # Build the block-aligned expert schedule.  Padding slots must contain the
    # sentinel assignment id so that padded blocks cannot overwrite output 0.
    selected_experts = selected_experts.to(torch.int32)
    num_assignments = selected_experts.numel()
    max_padded = num_assignments + _NUM_EXPERTS * _BLOCK_M
    sorted_token_ids = torch.full(
        (max_padded,),
        num_assignments,
        dtype=torch.int32,
        device=hidden_states.device,
    )
    sorted_expert_ids = torch.empty(
        ((max_padded + _BLOCK_M - 1) // _BLOCK_M,),
        dtype=torch.int32,
        device=hidden_states.device,
    )
    num_tokens_post_pad = torch.empty(
        (1,), dtype=torch.int32, device=hidden_states.device
    )
    moe_align_block_size_triton(
        selected_experts,
        _NUM_EXPERTS,
        _BLOCK_M,
        sorted_token_ids,
        sorted_expert_ids,
        num_tokens_post_pad,
    )

    # The Triton primitive consumes the model's native [expert, out, in]
    # weights and writes results in original token/top-k order.
    routing_weights_fp32 = routing_weights_topk.float()

    def expert_mm(a, weight, out, top_k):
        fused_moe(
            a,
            weight,
            out,
            None,
            None,
            None,
            routing_weights_fp32,
            selected_experts,
            sorted_token_ids,
            sorted_expert_ids,
            num_tokens_post_pad,
            False,
            top_k,
            tl.float32,
            False,
            False,
            False,
            None,
            _MOE_CONFIG,
        )

    num_tokens = hidden_states_flat.shape[0]
    gate_out = torch.empty(
        (num_tokens, _TOP_K, 512),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    up_out = torch.empty_like(gate_out)
    expert_mm(hidden_states_flat, expert_gate_proj, gate_out, _TOP_K)
    expert_mm(hidden_states_flat, expert_up_proj, up_out, _TOP_K)
    intermediate = gate_out * torch.sigmoid(gate_out)
    intermediate = intermediate * up_out
    expert_output = torch.empty(
        (num_assignments, 1, hidden_dim),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    expert_mm(
        intermediate.view(num_assignments, 512),
        expert_down_proj,
        expert_output,
        1,
    )
    expert_output = expert_output.view(num_tokens, _TOP_K, hidden_dim)
    expert_output = expert_output * routing_weights_topk[:, :, None]
    final_hidden_states = expert_output.sum(dim=1)

    shared_gate_out = torch.matmul(hidden_states_flat, shared_gate_proj.t())
    shared_up_out = torch.matmul(hidden_states_flat, shared_up_proj.t())
    shared_intermediate = (
        shared_gate_out * torch.sigmoid(shared_gate_out) * shared_up_out
    )
    shared_expert_output = torch.matmul(shared_intermediate, shared_down_proj.t())
    shared_gate = torch.sigmoid(
        torch.matmul(hidden_states_flat, shared_expert_gate_weight.t())
    )
    final_hidden_states = final_hidden_states + shared_gate * shared_expert_output

    output = final_hidden_states.view(batch_size, sequence_length, hidden_dim)
    return output, router_logits.to(torch.bfloat16)
