import torch, json
import reference as R, kernel as K
names = ["grad_output","token_indices","selected_tokens","w1_output","gate_output","up_output","gated_output","expert_output","selected_weights","w1_weight","w2_weight","w3_weight"]
outn = ["ghs","grw","gw1","gw2","gw3"]
dev=torch.device("cuda:0")
for line in open("workload.jsonl"):
    w=json.loads(line); ax=dict(w["axes"]); ax["hidden_dim"]=4096; ax["ffn_dim"]=14336
    torch.manual_seed(1)
    inp=R.get_inputs(ax,dev); args=[inp[n] for n in names]
    ref=R.run(*args); got=K.run(*args); tol=w["tolerance"]
    s=[]
    for nm,a,b in zip(outn,got,ref):
        af=a.float(); bf=b.float()
        thr=tol["max_atol"]+tol["max_rtol"]*bf.abs()
        bad=((af-bf).abs()>thr).float().mean().item()
        s.append(f"{nm}:badfrac={bad:.5f}")
    print(ax["batch_seq_len"],ax["num_tokens"], " ".join(s))
