import torch, math, sys
import torch.nn.functional as F
sys.path.insert(0,'/var/tmp/solbench/agent/opus5-budget100/L2__050_vae_decoder_mid_block_attention_resnet')
import reference

dev='cuda'
torch.manual_seed(0)
C=512

def make(B,H,W):
    g = torch.Generator(device=dev).manual_seed(1234)
    def rn(*s): return torch.randn(*s, device=dev, generator=g)
    def wm(o,i,*r):
        fan = r[-1] if r else i
        return torch.randn(o,i,*r, device=dev, generator=g)/math.sqrt(fan)
    args = dict(
        hidden_states=rn(B,C,H,W), temb=rn(B,C),
    )
    for p in ['resnet1','resnet2']:
        args[f'{p}_norm1_weight']=torch.ones(C,device=dev); args[f'{p}_norm1_bias']=torch.zeros(C,device=dev)
        args[f'{p}_conv1_weight']=wm(C,C,3,3); args[f'{p}_conv1_bias']=rn(C)
        args[f'{p}_time_emb_proj_weight']=wm(C,C); args[f'{p}_time_emb_proj_bias']=rn(C)
        args[f'{p}_norm2_weight']=torch.ones(C,device=dev); args[f'{p}_norm2_bias']=torch.zeros(C,device=dev)
        args[f'{p}_conv2_weight']=wm(C,C,3,3); args[f'{p}_conv2_bias']=rn(C)
    args['attn_group_norm_weight']=torch.ones(C,device=dev); args['attn_group_norm_bias']=torch.zeros(C,device=dev)
    for n in ['q','k','v','out']:
        args[f'attn_to_{n}_weight']=wm(C,C); args[f'attn_to_{n}_bias']=rn(C)
    args['eps']=1e-6
    return args

ORDER = ['hidden_states','temb','resnet1_norm1_weight','resnet1_norm1_bias','resnet1_conv1_weight','resnet1_conv1_bias','resnet1_time_emb_proj_weight','resnet1_time_emb_proj_bias','resnet1_norm2_weight','resnet1_norm2_bias','resnet1_conv2_weight','resnet1_conv2_bias','attn_group_norm_weight','attn_group_norm_bias','attn_to_q_weight','attn_to_q_bias','attn_to_k_weight','attn_to_k_bias','attn_to_v_weight','attn_to_v_bias','attn_to_out_weight','attn_to_out_bias','resnet2_norm1_weight','resnet2_norm1_bias','resnet2_conv1_weight','resnet2_conv1_bias','resnet2_time_emb_proj_weight','resnet2_time_emb_proj_bias','resnet2_norm2_weight','resnet2_norm2_bias','resnet2_conv2_weight','resnet2_conv2_bias','eps']

def check(out, ref, atol, rtol):
    ae=(out.float()-ref.float()).abs()
    bound = atol + rtol*ref.float().abs()
    bad = (ae>bound).float().mean().item()
    return 1-bad, ae.max().item()

TOL = {(1,32,32):(3.8822859424124205e-06,1.1920928955078125e-07),
       (1,61,61):(3.900360110448521e-06,1.1920928955078125e-07),
       (16,32,32):(3.931760339537081e-06,1.1920928955078125e-07),
       (32,32,32):(3.964399394002429e-06,1.1920928955078125e-07),
       (2,41,41):(3.898631173405141e-06,1.1920928955078125e-07),
       (2,64,64):(0.0001430511474609375,0.3166666666666667),
       (1,16,16):(9.5367431640625e-05,0.20394736842105265),
       }

if __name__=='__main__':
    for shape in [(1,32,32),(1,61,61),(32,32,32),(2,64,64)]:
        B,H,W = shape
        a = make(B,H,W)
        args = [a[k] for k in ORDER]
        ref = reference.run(*args)
        print(shape, 'ref out absmax', ref.abs().max().item(), 'mean', ref.abs().mean().item())
        atol,rtol = TOL[shape]
        # variant: conv via unfold-matmul fp32
        import kernel_test
        out = kernel_test.run(*args)
        r,mx = check(out, ref, atol, rtol)
        print('   matched_ratio', r, 'maxabs', mx, 'PASS' if r>=0.99 else 'FAIL')
