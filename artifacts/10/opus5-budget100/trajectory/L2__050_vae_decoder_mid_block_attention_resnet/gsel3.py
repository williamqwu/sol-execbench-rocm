import torch, sys
sys.path.insert(0,'/var/tmp/solbench/agent/opus5-budget100/L2__050_vae_decoder_mid_block_attention_resnet')
from harness import make, TOL
import kernel_q as K
for shape in TOL:
    args=make(*shape); t=list(args[:32]); eps=args[32]
    def eager(): return K._body(*t, eps, *K._fused(t))
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): K._body(*t, eps, *K._fused(t))
    torch.cuda.current_stream().wait_stream(s)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out=K._body(*t, eps, *K._fused(t))
    def gp():
        g.replay(); return out.clone()
    ref=eager(); o=gp()
    te=min(K._time(eager),K._time(eager)); tg=min(K._time(gp),K._time(gp))
    print(f"{shape}: eager={te:.4f} graph_nocopy={tg:.4f} ratio={tg/te:.3f} exact={(o==ref).all().item()}")
