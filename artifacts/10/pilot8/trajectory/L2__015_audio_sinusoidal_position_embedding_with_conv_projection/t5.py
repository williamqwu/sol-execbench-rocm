import torch, time
import reference as R, kernel as K
dev=torch.device('cuda')
shapes=[(2,1688),(32,4328),(1,1048),(1,1808),(32,920),(64,128),(16,2048),(16,384),(4,2216),(2,256),(2,2528),(8,1976),(8,3256),(32,768),(64,512),(1,3000)]
for (B,T) in shapes:
    d=R.get_inputs({"batch_size":B,"time_dim":T},dev)
    a=R.run(**d)
    b=K.run(**d)
    b2=K.run(**d)
    print(B,T,"err",(a.float()-b.float()).abs().max().item(),"rerun-diff",(b.float()-b2.float()).abs().max().item(),"max",a.abs().max().item())
