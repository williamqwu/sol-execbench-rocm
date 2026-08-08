import torch, sys
sys.path.insert(0,'/var/tmp/solbench/agent/opus5-budget100/L2__050_vae_decoder_mid_block_attention_resnet')
from harness import make, TOL
import reference
for shape in TOL:
    B,H,W = shape
    args = make(B,H,W)
    a = reference.run(*args); b = reference.run(*args)
    atol,rtol = TOL[shape]
    tight = rtol <= 1.2e-7
    print(f"B{B} {H}x{W}: run-to-run identical={torch.equal(a,b)} maxdiff={(a-b).abs().max().item():.3e}  tol_tight={tight}")
