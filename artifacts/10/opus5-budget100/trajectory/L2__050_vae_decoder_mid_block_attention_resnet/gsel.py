import torch, sys
sys.path.insert(0,'/var/tmp/solbench/agent/opus5-budget100/L2__050_vae_decoder_mid_block_attention_resnet')
from harness import make, TOL
import kernel_q as K
for shape in TOL:
    args=make(*shape); t=list(args[:32]); eps=args[32]
    def eager(): return K._body(*t, eps, *K._fused(t))
    ent=K._build(t,eps)
    te=K._time(eager)
    if ent is not None:
        static,g,out=ent
        def gp():
            torch._foreach_copy_(static,t); g.replay(); return out.clone()
        tg=K._time(gp)
    else:
        # build graph manually to measure anyway
        tg=float('nan')
    print(f"{shape}: graph={'YES' if ent else 'no '} eager={te:.4f} graph={tg:.4f}")
