import torch, json, time
import reference as R, kernel as K
names = ["grad_output","token_indices","selected_tokens","w1_output","gate_output","up_output","gated_output","expert_output","selected_weights","w1_weight","w2_weight","w3_weight"]
dev=torch.device("cuda:0")
def bench(f,args,n=20):
    for _ in range(5): f(*args)
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f(*args)
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3
for line in open("workload.jsonl"):
    w=json.loads(line); ax=dict(w["axes"]); ax["hidden_dim"]=4096; ax["ffn_dim"]=14336
    torch.manual_seed(1); inp=R.get_inputs(ax,dev); args=[inp[n] for n in names]
    tr=bench(R.run,args); tk=bench(K.run,args)
    print(f"{ax['batch_seq_len']:5d} {ax['num_tokens']:5d}  ref={tr:8.3f}ms  mine={tk:7.3f}ms  {tr/tk:5.2f}x")
