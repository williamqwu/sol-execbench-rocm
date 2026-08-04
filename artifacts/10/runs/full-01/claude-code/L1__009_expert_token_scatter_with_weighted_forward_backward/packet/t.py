import torch, json, importlib, sys, time
import reference as R
import kernel as K

names = ["grad_output","token_indices","selected_tokens","w1_output","gate_output","up_output","gated_output","expert_output","selected_weights","w1_weight","w2_weight","w3_weight"]
outn = ["ghs","grw","gw1","gw2","gw3"]
dev = torch.device("cuda:0")
ok=True
for line in open("workload.jsonl"):
    w = json.loads(line)
    ax = dict(w["axes"]); ax["hidden_dim"]=4096; ax["ffn_dim"]=14336
    torch.manual_seed(0)
    inp = R.get_inputs(ax, dev)
    args = [inp[n] for n in names]
    ref = R.run(*args)
    got = K.run(*args)
    tol = w["tolerance"]
    line_ok=True
    for nm, a, b in zip(outn, got, ref):
        af=a.float(); bf=b.float()
        thr = tol["max_atol"] + tol["max_rtol"]*bf.abs()
        bad = (af-bf).abs() > thr
        ratio = 1 - bad.float().mean().item()
        if ratio < tol["required_matched_ratio"]:
            line_ok=False
            print(f"  FAIL {nm} ratio={ratio:.5f} maxerr={(af-bf).abs().max().item():.4f} atol={tol['max_atol']:.4f}")
    print(("PASS " if line_ok else "FAIL "), ax["batch_seq_len"], ax["num_tokens"])
    ok = ok and line_ok
print("ALL OK" if ok else "SOME FAILED")
