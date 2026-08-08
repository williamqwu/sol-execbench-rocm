import torch, sys
sys.path.insert(0,'/var/tmp/solbench/agent/opus5-budget100/L2__050_vae_decoder_mid_block_attention_resnet')
from harness import make, TOL
import kernel_q as K
for shape in TOL:
    args=make(*shape); t=list(args[:32]); eps=args[32]
    def eager(): return K._body(*t, eps, *K._fused(t))
    static=[x.clone() for x in t]
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): K._body(*static, eps, *K._fused(static))
    torch.cuda.current_stream().wait_stream(s)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out=K._body(*static, eps, *K._fused(static))
    def gp():
        torch._foreach_copy_(static,t); g.replay(); return out.clone()
    te=min(K._time(eager),K._time(eager)); tg=min(K._time(gp),K._time(gp))
    tcp=K._time(lambda: torch._foreach_copy_(static,t))
    tcl=K._time(lambda: out.clone())
    print(f"{shape}: eager={te:.4f} graph={tg:.4f} ratio={tg/te:.3f}  copy={tcp:.4f} clone={tcl:.4f}")
