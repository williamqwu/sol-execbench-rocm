import torch


@torch.no_grad()
def run(grad_output, input_ids, hidden_states_fp32, rstd, norm_weight):
    vocab_size = 65536
    H = grad_output.shape[-1]
    go32 = grad_output.float()
    hn = hidden_states_fp32 * rstd
    gnw = (go32 * hn).sum(dim=(0, 1))
    ghn = go32 * norm_weight.float()
    m = (ghn * hn).mean(dim=-1, keepdim=True)
    gh = rstd * (ghn - m * hn)
    gew = torch.zeros(vocab_size, H, dtype=torch.float32, device=grad_output.device)
    gew.index_add_(0, input_ids.view(-1), gh.view(-1, H))
    return gew.to(torch.bfloat16), gnw.to(torch.bfloat16)
