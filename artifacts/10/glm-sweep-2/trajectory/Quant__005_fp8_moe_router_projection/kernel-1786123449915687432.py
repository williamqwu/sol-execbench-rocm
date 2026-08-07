import torch

@torch.no_grad()
def run(hidden_states, gate_weight, scale_hidden, scale_weight):
    num_experts = 64
    M, K = hidden_states.shape
    nb = K // 128
    # dequant hidden (BlockWise1x128): a[m,k]=qx[m,k]*sh[m,k//128]
    a = (hidden_states.to(torch.float32).reshape(M, nb, 128) * scale_hidden.reshape(M, nb, 1)).reshape(M, K).to(torch.bfloat16)
    # dequant weight first 64 experts (BlockWise128x128, shared scale): w[n,k]=qw[n,k]*sw[0,k//128]
    w = (gate_weight[:num_experts].to(torch.float32).reshape(num_experts, nb, 128) * scale_weight.reshape(1, nb, 1)).reshape(num_experts, K).to(torch.bfloat16)
    return a @ w.T
