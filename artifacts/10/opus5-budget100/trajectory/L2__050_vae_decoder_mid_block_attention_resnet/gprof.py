import torch, sys, time
import torch.nn.functional as F
sys.path.insert(0,'/var/tmp/solbench/agent/opus5-budget100/L2__050_vae_decoder_mid_block_attention_resnet')
from harness import make, bench
import reference
from kernel_g import _body

for shape in [(1,16,16),(1,32,32),(4,16,16),(1,48,48),(2,64,64),(32,32,32)]:
    B,H,W = shape
    args = make(B,H,W)
    tensors = [a for a in args if isinstance(a, torch.Tensor)]
    eps = args[-1]
    static = [t.clone() for t in tensors]
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): _body(*static, eps)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = _body(*static, eps)

    t_ref = bench(lambda: reference.run(*args))
    t_replay = bench(lambda: g.replay())
    t_copy = bench(lambda: torch._foreach_copy_(static, tensors))
    t_clone = bench(lambda: out.clone())
    t_full = bench(lambda: (torch._foreach_copy_(static, tensors), g.replay(), out.clone()))
    nbytes = sum(t.numel()*4 for t in tensors)/1e6
    print(f"{str(shape):12s} ref={t_ref:7.4f} replay={t_replay:7.4f} copy={t_copy:7.4f} clone={t_clone:7.4f} full={t_full:7.4f} inMB={nbytes:.0f}")
