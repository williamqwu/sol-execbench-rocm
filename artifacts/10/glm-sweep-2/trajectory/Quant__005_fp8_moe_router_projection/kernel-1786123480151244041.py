import torch

@torch.compile(dynamic=True)
def _fwd(hidden_states, gate_weight, scale_hidden, scale_weight):
    num_experts = 64
    M, K = hidden_states.shape
    nb = K // 128
    a = (hidden_states.to(torch.float32).reshape(M, nb, 128) * scale_hidden.reshape(M, nb, 1)).reshape(M, K).to(torch.bfloat16)
    w = (gate_weight[:num_experts].to(torch.float32).reshape(num_experts, nb, 128) * scale_weight.reshape(1, nb, 1)).reshape(num_experts, K).to(torch.bfloat16)
    return a @ w.T

@torch.no_grad()
def run(hidden_states, gate_weight, scale_hidden, scale_weight):
    return _fwd(hidden_states, gate_weight, scale_hidden, scale_weight)
