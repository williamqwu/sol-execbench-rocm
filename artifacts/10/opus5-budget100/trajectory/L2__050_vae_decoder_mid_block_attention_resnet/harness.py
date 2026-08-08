import torch, math, time, sys, importlib
import torch.nn.functional as F
BASE='/var/tmp/solbench/agent/opus5-budget100/L2__050_vae_decoder_mid_block_attention_resnet'
sys.path.insert(0, BASE)
dev='cuda'
C=512

ORDER = ['hidden_states','temb','resnet1_norm1_weight','resnet1_norm1_bias','resnet1_conv1_weight','resnet1_conv1_bias','resnet1_time_emb_proj_weight','resnet1_time_emb_proj_bias','resnet1_norm2_weight','resnet1_norm2_bias','resnet1_conv2_weight','resnet1_conv2_bias','attn_group_norm_weight','attn_group_norm_bias','attn_to_q_weight','attn_to_q_bias','attn_to_k_weight','attn_to_k_bias','attn_to_v_weight','attn_to_v_bias','attn_to_out_weight','attn_to_out_bias','resnet2_norm1_weight','resnet2_norm1_bias','resnet2_conv1_weight','resnet2_conv1_bias','resnet2_time_emb_proj_weight','resnet2_time_emb_proj_bias','resnet2_norm2_weight','resnet2_norm2_bias','resnet2_conv2_weight','resnet2_conv2_bias','eps']

TOL = {(1,32,32):(3.8822859424124205e-06,1.1920928955078125e-07),
       (1,61,61):(3.900360110448521e-06,1.1920928955078125e-07),
       (16,32,32):(3.931760339537081e-06,1.1920928955078125e-07),
       (32,32,32):(3.964399394002429e-06,1.1920928955078125e-07),
       (2,41,41):(3.898631173405141e-06,1.1920928955078125e-07),
       (2,64,64):(1.430511474609375e-04,0.3166666666666667),
       (1,48,48):(1.049041748046875e-04,0.29545454545454547),
       (4,16,16):(1.1444091796875e-04,0.22208121827411167),
       (8,32,32):(1.2874603271484375e-04,0.37037037037037035),
       (4,48,48):(1.3828277587890625e-04,0.4655172413793104),
       (1,16,16):(9.5367431640625e-05,0.20394736842105265),
       }
SHAPES = list(TOL.keys())

def make(B,H,W, seed=1234):
    g = torch.Generator(device=dev).manual_seed(seed)
    def rn(*s): return torch.randn(*s, device=dev, generator=g)
    def wm(*s):
        fan = s[-1]
        return torch.randn(*s, device=dev, generator=g)/math.sqrt(fan)
    a = {}
    a['hidden_states']=rn(B,C,H,W); a['temb']=rn(B,C)
    for p in ['resnet1','resnet2']:
        a[f'{p}_norm1_weight']=torch.ones(C,device=dev); a[f'{p}_norm1_bias']=torch.zeros(C,device=dev)
        a[f'{p}_conv1_weight']=wm(C,C,3,3); a[f'{p}_conv1_bias']=rn(C)
        a[f'{p}_time_emb_proj_weight']=wm(C,C); a[f'{p}_time_emb_proj_bias']=rn(C)
        a[f'{p}_norm2_weight']=torch.ones(C,device=dev); a[f'{p}_norm2_bias']=torch.zeros(C,device=dev)
        a[f'{p}_conv2_weight']=wm(C,C,3,3); a[f'{p}_conv2_bias']=rn(C)
    a['attn_group_norm_weight']=torch.ones(C,device=dev); a['attn_group_norm_bias']=torch.zeros(C,device=dev)
    for n in ['q','k','v','out']:
        a[f'attn_to_{n}_weight']=wm(C,C); a[f'attn_to_{n}_bias']=rn(C)
    a['eps']=1e-6
    return [a[k] for k in ORDER]

def check(out, ref, atol, rtol):
    ae=(out.float()-ref.float()).abs()
    bound = atol + rtol*ref.float().abs()
    ratio = 1.0 - (ae>bound).float().mean().item()
    return ratio, ae.max().item()

def bench(fn, iters=20, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/iters*1000

def main(modname='kernel'):
    import reference
    mod = importlib.import_module(modname)
    tot_s=0.0; n=0; allpass=True
    for shape in SHAPES:
        B,H,W = shape
        args = make(B,H,W)
        ref = reference.run(*args)
        out = mod.run(*args)
        atol,rtol = TOL[shape]
        r,mx = check(out, ref, atol, rtol)
        ok = r>=0.99
        allpass &= ok
        tr = bench(lambda: reference.run(*args))
        tm = bench(lambda: mod.run(*args))
        sp = tr/tm
        tot_s += math.log(sp); n+=1
        print(f"B{B} {H}x{W}: {'PASS' if ok else 'FAIL'} ratio={r:.5f} maxabs={mx:.3e} refmax={ref.abs().max().item():.2f} | mine={tm:.4f} ref={tr:.4f} sp={sp:.2f}x")
    print(f"geomean {math.exp(tot_s/n):.3f}x  allpass={allpass}")

if __name__=='__main__':
    main(sys.argv[1] if len(sys.argv)>1 else 'kernel')
