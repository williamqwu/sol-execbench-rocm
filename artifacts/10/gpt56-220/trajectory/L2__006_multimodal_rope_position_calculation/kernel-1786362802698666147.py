import torch
from typing import Tuple
import triton
import triton.language as tl

@triton.jit
def _rope(ids, ig, vg, secs, mask, out, delta, S:tl.constexpr,
          B:tl.constexpr, NI:tl.constexpr, NV:tl.constexpr, NOBJ:tl.constexpr, BS:tl.constexpr):
    b=tl.program_id(0)
    p=tl.arange(0,BS)
    active=p<S
    x=tl.load(ids+b*S+p,mask=active,other=-1)
    valid=tl.load(mask+b*S+p,mask=active,other=0)==1
    # Compact index. Generated inputs are dense; this also handles padding
    # preceding a token for ordinary 1-D positions.
    cp=tl.cumsum(valid.to(tl.int32),axis=0)-1
    base=cp
    o0=base; o1=base; o2=base
    shift=tl.zeros((BS,),tl.int32)
    # Metadata index equals the number of same-type markers in earlier rows,
    # plus the local marker rank. Counts are tiny, so scanning prior rows is cheap.
    img_prior=tl.zeros((),tl.int32); vid_prior=tl.zeros((),tl.int32)
    for q in tl.static_range(0,B):
        qx=tl.load(ids+q*S+p,mask=(p<S) & (q<b),other=-1)
        img_prior += tl.sum((qx==151655).to(tl.int32),axis=0)
        vid_prior += tl.sum((qx==151656).to(tl.int32),axis=0)
    im=(x==151655)&valid
    vm=(x==151656)&valid
    ir=tl.cumsum(im.to(tl.int32),axis=0)-1+img_prior
    vr=tl.cumsum(vm.to(tl.int32),axis=0)-1+vid_prior
    # At most eight objects occur in the specified domain. Each iteration
    # selects one marker by its rank and applies that segment's coordinate map.
    mr=tl.cumsum((im|vm).to(tl.int32),axis=0)-1
    for r in tl.static_range(0,NOBJ):
        sel=mr==r
        has_i=tl.sum((im & sel).to(tl.int32),axis=0)>0
        has_v=tl.sum((vm & sel).to(tl.int32),axis=0)>0
        ei=tl.max(tl.where(im & sel,cp,-1),axis=0)
        ev=tl.max(tl.where(vm & sel,cp,-1),axis=0)
        gri=tl.max(tl.where(im & sel,ir,0),axis=0)
        grv=tl.max(tl.where(vm & sel,vr,0),axis=0)
        if NI:
            gi=tl.minimum(gri,NI-1)
            it=tl.load(ig+gi*3); ih=tl.load(ig+gi*3+1)//2; iw=tl.load(ig+gi*3+2)//2
        else:
            it=1;ih=1;iw=1
        if NV:
            gv=tl.minimum(grv,NV-1)
            vt=tl.load(vg+gv*3); vh=tl.load(vg+gv*3+1)//2; vw=tl.load(vg+gv*3+2)//2
            vs=tl.load(secs+gv)
        else:
            vt=1;vh=1;vw=1;vs=0.0
        ed=tl.where(has_i,ei,ev); tt=tl.where(has_i,it,vt); hh=tl.where(has_i,ih,vh); ww=tl.where(has_i,iw,vw)
        sec=tl.where(has_i,0.0,vs); cnt=tt*hh*ww
        k=cp-ed; inside=(has_i|has_v)&(k>=0)&(k<cnt)
        tcoord=(k//(hh*ww)).to(tl.float32)*sec*2.0
        rem=k%(hh*ww)
        o0=tl.where(inside,ed+shift+tcoord.to(tl.int32),o0)
        o1=tl.where(inside,ed+shift+rem//ww,o1)
        o2=tl.where(inside,ed+shift+rem%ww,o2)
        mx=tl.maximum(tl.maximum((tt-1).to(tl.float32)*sec*2.0,hh-1.0),ww-1.0).to(tl.int32)
        d=tl.where(has_i|has_v,mx+1-cnt,0)
        after=(has_i|has_v)&(cp>=ed+cnt)
        o0+=tl.where(after,d,0);o1+=tl.where(after,d,0);o2+=tl.where(after,d,0)
        shift+=tl.where(after,d,0)
    fill=active & valid
    tl.store(out+(0*B+b)*S+p,tl.where(fill,o0,1),mask=active)
    tl.store(out+(1*B+b)*S+p,tl.where(fill,o1,1),mask=active)
    tl.store(out+(2*B+b)*S+p,tl.where(fill,o2,1),mask=active)
    if tl.constexpr(NI==0 and NV==0):
        d=tl.max(tl.where(valid,cp,-1),axis=0)+1-S
    else:
        d=tl.max(tl.where(valid,o0,0),axis=0)
        d=tl.maximum(d,tl.max(tl.where(valid,o1,0),axis=0))
        d=tl.maximum(d,tl.max(tl.where(valid,o2,0),axis=0))+1-tl.sum(valid.to(tl.int32),axis=0)
    tl.store(delta+b,d)

@torch.no_grad()
def run(input_ids:torch.Tensor,image_grid_thw:torch.Tensor,video_grid_thw:torch.Tensor,
        second_per_grid_ts:torch.Tensor,attention_mask:torch.Tensor)->Tuple[torch.Tensor,torch.Tensor]:
    B,S=input_ids.shape; BS=triton.next_power_of_2(S)
    storage=torch.empty(3*B*S+B,dtype=input_ids.dtype,device=input_ids.device)
    out=storage[:3*B*S].view(3,B,S)
    delta=storage[3*B*S:].view(B,1)
    _rope[(B,)](input_ids,image_grid_thw,video_grid_thw,second_per_grid_ts,attention_mask,
                out,delta,S=S,B=B,NI=image_grid_thw.shape[0],NV=video_grid_thw.shape[0],
                NOBJ=image_grid_thw.shape[0]+video_grid_thw.shape[0],BS=BS,
                num_warps=4)
    return out,delta
