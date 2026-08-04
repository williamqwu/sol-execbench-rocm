import torch, sys, os, time, triton
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference, impl
torch.manual_seed(0)
def bench(f, n=50):
    for _ in range(10): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3
for M in [384, 1024, 2048, 4096]:
    ins = reference.get_inputs({'num_tokens':M}, torch.device('cuda'))
    hs, rw, w1, w2 = ins['hidden_states'], ins['routing_weight'], ins['gate_up_weight'], ins['down_weight']
    tq_a = bench(lambda: impl.quantize_act(hs))
    tq_w1 = bench(lambda: impl.quantize_w(w1))
    tq_w2 = bench(lambda: impl.quantize_w(w2))
    aq, asc = impl.quantize_act(hs); w1q,w1s = impl.quantize_w(w1); w2q,w2s = impl.quantize_w(w2)
    NH = w1.shape[0]//2; K=hs.shape[1]; H=w2.shape[0]
    gq = torch.empty((M,NH),dtype=torch.float8_e4m3fn,device='cuda')
    gs = torch.empty((M,NH//128),dtype=torch.float32,device='cuda')
    out = torch.empty((M,H),dtype=torch.bfloat16,device='cuda')
    c1=impl.CFG1; c2=impl.CFG2
    def g1():
        impl._gemm1_silu_quant[(triton.cdiv(M,c1['BLOCK_M'])*(NH//128),)](aq,asc,w1q,w1s,gq,gs,M,K//128,NH//128,
          aq.stride(0),asc.stride(0),w1q.stride(0),w1s.stride(0),gq.stride(0),gs.stride(0),
          BLOCK_M=c1['BLOCK_M'],GROUP_M=c1['GROUP_M'],EVEN_M=(M%c1['BLOCK_M']==0),num_warps=c1['num_warps'],num_stages=c1['num_stages'])
    def g2():
        impl._gemm2[(triton.cdiv(M,c2['BLOCK_M'])*triton.cdiv(H,c2['BLOCK_N']),)](gq,gs,w2q,w2s,rw,out,M,NH//128,H//128,
          gq.stride(0),gs.stride(0),w2q.stride(0),w2s.stride(0),out.stride(0),
          BLOCK_M=c2['BLOCK_M'],BLOCK_N=c2['BLOCK_N'],GROUP_M=c2['GROUP_M'],EVEN_M=(M%c2['BLOCK_M']==0),num_warps=c2['num_warps'],num_stages=c2['num_stages'])
    t1=bench(g1); t2=bench(g2)
    tot = bench(lambda: impl.moe(hs,rw,w1,w2))
    print(f"M={M}: qa={tq_a*1e3:.0f}us qw1={tq_w1*1e3:.0f} qw2={tq_w2*1e3:.0f} gemm1={t1*1e3:.0f} gemm2={t2*1e3:.0f} sum={(tq_a+tq_w1+tq_w2+t1+t2)*1e3:.0f} total={tot*1e3:.0f}us")
