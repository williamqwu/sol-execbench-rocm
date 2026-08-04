"""Benchmark kernel modules against the real 38 workloads (real indptr/indices)."""
import json
import math
import os
import sys
import importlib

import torch
from safetensors.torch import load_file

ROOT = "/var/tmp/solbench/agent/pilot8/FlashInfer-Bench__019_mla_paged_prefill_causal_h16_ckv512_kpe64_ps1"
sys.path.insert(0, ROOT)
PROB = "/work/data/SOL-ExecBench/benchmark/FlashInfer-Bench/019_mla_paged_prefill_causal_h16_ckv512_kpe64_ps1"
BLOB = "/work/data"

dev = "cuda"
H, D, DP = 16, 512, 64


def load_workloads():
    out = []
    with open(f"{PROB}/workload.jsonl") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


_cache = {}


def get_meta(path):
    if path not in _cache:
        _cache[path] = load_file(os.path.join(BLOB, path))
    return _cache[path]


def build(w, npages_cap=None):
    ax = w["axes"]
    total_q = ax["total_q"]
    num_pages = ax["num_pages"]
    ins = w["inputs"]
    m = get_meta(ins["qo_indptr"]["path"])
    qo = m["qo_indptr"].to(torch.int32).to(dev)
    kvp = m["kv_indptr"].to(torch.int32).to(dev)
    kvi = m["kv_indices"].to(torch.int32).to(dev)
    g = torch.Generator(device=dev).manual_seed(1234)
    q_nope = torch.randn(total_q, H, D, generator=g, device=dev, dtype=torch.float32).bfloat16()
    q_pe = torch.randn(total_q, H, DP, generator=g, device=dev, dtype=torch.float32).bfloat16()
    np_use = num_pages if npages_cap is None else min(num_pages, npages_cap)
    ckv = torch.randn(np_use, 1, D, generator=g, device=dev, dtype=torch.float32).bfloat16()
    kpe = torch.randn(np_use, 1, DP, generator=g, device=dev, dtype=torch.float32).bfloat16()
    if np_use < num_pages:
        kvi = kvi % np_use
    return (q_nope, q_pe, ckv, kpe, qo, kvp, kvi, float(ins["sm_scale"]["value"]))


def bench(mod, args, iters=20, warmup=5):
    for _ in range(warmup):
        mod.run(*args)
    torch.cuda.synchronize()
    st = torch.cuda.Event(True)
    en = torch.cuda.Event(True)
    st.record()
    for _ in range(iters):
        mod.run(*args)
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters


if __name__ == "__main__":
    mods = sys.argv[1:] or ["kernel"]
    loaded = [importlib.import_module(m) for m in mods]
    ws = load_workloads()
    ARGS = [build(w) for w in ws]
    tots = {m: [] for m in mods}
    print(f"{'uuid':10s} {'total_q':>8s} {'batch':>6s} {'kvidx':>8s} " + " ".join(f"{m:>12s}" for m in mods))
    for w, a in zip(ws, ARGS):
        ax = w["axes"]
        row = []
        for name, mod in zip(mods, loaded):
            t = bench(mod, a)
            tots[name].append(t)
            row.append(t)
        print(f"{w['uuid'][:8]:10s} {ax['total_q']:8d} {ax['len_indptr']-1:6d} {ax['num_kv_indices']:8d} " + " ".join(f"{t:12.4f}" for t in row))
    print()
    for name in mods:
        v = tots[name]
        print(f"{name}: sum={sum(v):9.3f} ms  geo={math.exp(sum(map(math.log, v))/len(v)):8.4f} ms")
