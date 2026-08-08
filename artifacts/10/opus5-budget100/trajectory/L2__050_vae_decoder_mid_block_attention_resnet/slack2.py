import torch, sys
import torch.nn.functional as F
sys.path.insert(0,'/var/tmp/solbench/agent/opus5-budget100/L2__050_vae_decoder_mid_block_attention_resnet')
from harness import make, ORDER, TOL, check
import reference

dev='cuda'
C=512

# Which conv variants stay within tight tolerance?
for shape in [(1,32,32),(32,32,32),(2,41,41)]:
    B,H,W = shape
    args = make(B,H,W)
    ref = reference.run(*args)
    atol,rtol = TOL[shape]

    # variant A: channels_last convs
    import copy
    def run_cl(*a):
        import kernel_variants as kv
        return kv.run_cl(*a)
    # variant: run reference again (determinism check)
    r2 = reference.run(*args)
    ratio,mx = check(r2, ref, atol, rtol)
    print(f"{shape} rerun-identical: ratio={ratio:.5f} max={mx:.2e}")

    # channels_last
    import kernel_variants as kv
    for name, fn in [('channels_last', kv.run_cl), ('unfold_gemm', kv.run_unfold), ('fp16x2conv', kv.run_fp16x2)]:
        try:
            o = fn(*args)
            ratio,mx = check(o, ref, atol, rtol)
            bound_typ = atol + rtol*ref.abs()
            frac_bad = ((o-ref).abs()>bound_typ).float().mean().item()
            print(f"  {name}: ratio={ratio:.5f} max={mx:.2e} {'PASS' if ratio>=0.99 else 'FAIL'}")
        except Exception as e:
            print(f"  {name}: ERR {type(e).__name__} {e}")
