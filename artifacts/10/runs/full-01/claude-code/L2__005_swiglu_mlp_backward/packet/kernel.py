import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_bwd_elem(
    GGO, UP, AG, GO,          # inputs  [N] flat  (N = M*I)
    GATED, GG, GU,            # outputs [N] flat
    N,
    BLOCK: tl.constexpr,
):
    """Fused elementwise stage of the SwiGLU MLP backward.

    gated_output        = bf16(activated_gate * up_output)
    grad_up_output      = bf16(grad_gated_output * activated_gate)
    grad_activated_gate = bf16(grad_gated_output * up_output)
    grad_gate_output    = bf16(f32(grad_activated_gate) * silu'(gate_output))
    """
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    ggo = tl.load(GGO + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(UP + offs, mask=mask, other=0.0).to(tl.float32)
    ag = tl.load(AG + offs, mask=mask, other=0.0).to(tl.float32)
    go = tl.load(GO + offs, mask=mask, other=0.0).to(tl.float32)

    gated = (ag * up).to(tl.bfloat16)
    grad_up = (ggo * ag).to(tl.bfloat16)
    gag = (ggo * up).to(tl.bfloat16).to(tl.float32)

    s = 1.0 / (1.0 + tl.exp(-go))
    silu_grad = s * (1.0 + go * (1.0 - s))
    grad_gate = (gag * silu_grad).to(tl.bfloat16)

    tl.store(GATED + offs, gated, mask=mask)
    tl.store(GG + offs, grad_gate, mask=mask)
    tl.store(GU + offs, grad_up, mask=mask)


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    gate_output: torch.Tensor,
    up_output: torch.Tensor,
    activated_gate: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
):
    batch_size, seq_len, hidden_size = grad_output.shape
    intermediate_size = gate_output.shape[-1]
    M = batch_size * seq_len
    H = hidden_size
    I = intermediate_size

    go2 = grad_output.reshape(M, H)
    x2 = x.reshape(M, H)

    # ---- 1. grad_gated_output = grad_output @ down_weight        [M, I]
    ggo = torch.matmul(go2, down_weight)

    # ---- 2. fused elementwise stage
    gated = torch.empty((M, I), dtype=torch.bfloat16, device=go2.device)
    gg = torch.empty((M, I), dtype=torch.bfloat16, device=go2.device)
    gu = torch.empty((M, I), dtype=torch.bfloat16, device=go2.device)

    N = M * I
    BLOCK = 4096
    _swiglu_bwd_elem[(triton.cdiv(N, BLOCK),)](
        ggo, up_output, activated_gate, gate_output,
        gated, gg, gu,
        N, BLOCK=BLOCK, num_warps=8, num_stages=1,
    )

    # ---- 3. grad_down_weight = grad_output^T @ gated_output      [H, I]
    grad_down_weight = torch.matmul(go2.t(), gated)

    # ---- 4. weight gradients                                     [I, H]
    grad_gate_weight = torch.matmul(gg.t(), x2)
    grad_up_weight = torch.matmul(gu.t(), x2)

    # ---- 5. grad_x: two *separately rounded* bf16 matmuls, then a bf16 add.
    #        Fusing them with an fp32 accumulator is NOT equivalent -- the two
    #        terms cancel heavily and the reference's intermediate rounding is
    #        part of the spec.
    grad_x = torch.matmul(gg, gate_weight)
    grad_x += torch.matmul(gu, up_weight)

    return (
        grad_x.view(batch_size, seq_len, H),
        grad_gate_weight,
        grad_up_weight,
        grad_down_weight,
    )
