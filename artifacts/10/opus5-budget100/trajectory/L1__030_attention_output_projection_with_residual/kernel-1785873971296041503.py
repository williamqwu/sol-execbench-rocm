import torch, os
from torch.utils.cpp_extension import load
_D='/var/tmp/solbench/agent/opus5-budget100/L1__030_attention_output_projection_with_residual/work'
_mod = load(name='hl', sources=[_D+'/ext/hl.cpp'], extra_include_paths=['/opt/rocm/include'],
            extra_ldflags=['-L/opt/rocm/lib','-lhipblaslt'], extra_cflags=['-O3'], verbose=False)
@torch.no_grad()
def run(attn_output, residual, o_proj_weight):
    b,s,h = attn_output.shape
    return _mod.gemm_res(attn_output.view(-1,h), residual.view(-1,h), o_proj_weight).view(b,s,h)
