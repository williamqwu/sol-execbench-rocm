import argparse, importlib, json, sys, time
import torch

sys.path.insert(0, ".")

import enum
if not hasattr(enum, "StrEnum"):
    class StrEnum(str, enum.Enum):
        def __str__(self):
            return str(self.value)
    enum.StrEnum = StrEnum

import reference


def load_workloads():
    wls = []
    with open("workload.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                wls.append(json.loads(line))
    return wls


def check(out, ref, tol):
    out = out.float()
    ref = ref.float()
    diff = (out - ref).abs()
    allowed = tol["max_atol"] + tol["max_rtol"] * ref.abs()
    matched = (diff <= allowed).float().mean().item()
    return matched, diff.max().item(), (diff / (tol["max_atol"] + tol["max_rtol"] * ref.abs())).max().item()


def bench(fn, args, iters=20, warmup=5):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    st = torch.cuda.Event(True); en = torch.cuda.Event(True)
    st.record()
    for _ in range(iters):
        fn(*args)
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters * 1000.0  # us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="kernel")
    ap.add_argument("--only", type=int, default=None)
    ap.add_argument("--noref", action="store_true")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    mod = importlib.import_module(args.module)
    importlib.reload(mod)

    dev = torch.device("cuda:0")
    wls = load_workloads()
    torch.manual_seed(0)

    tot_ref = 0.0
    tot_mine = 0.0
    allok = True
    for i, w in enumerate(wls):
        if args.only is not None and i != args.only:
            continue
        axes = w["axes"]
        inp = reference.get_inputs(axes, dev)
        a = (inp["hidden_states"], inp["weight"], inp["scale_x"], inp["scale_w"])
        out = mod.run(*a)
        if not args.noref:
            ref = reference.run(*a)
            assert out.shape == ref.shape, (out.shape, ref.shape)
            assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
            m, mx, worst = check(out, ref, w["tolerance"])
            ok = m >= w["tolerance"]["required_matched_ratio"]
        else:
            m, mx, worst, ok = -1, -1, -1, True
        allok &= ok
        t_mine = bench(mod.run, a, iters=args.iters)
        t_ref = bench(reference.run, a, iters=3) if not args.noref else 0.0
        tot_ref += t_ref; tot_mine += t_mine
        print(f"[{i:2d}] B={axes['batch_size']:5d} L={axes['seq_len']:5d} M={axes['batch_size']*axes['seq_len']:6d} "
              f"{'PASS' if ok else 'FAIL'} match={m:.5f} maxdiff={mx:.4f} worst_ratio={worst:.3f} "
              f"mine={t_mine:9.1f}us ref={t_ref:10.1f}us speedup={t_ref/max(t_mine,1e-9):6.1f}x")
        del inp, a
        torch.cuda.empty_cache()
    print(f"ALL {'PASS' if allok else 'FAIL'}  total mine={tot_mine:.1f}us ref={tot_ref:.1f}us")


if __name__ == "__main__":
    main()
