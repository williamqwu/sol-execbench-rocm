import torch, time
import reference as R, kernel as K
dev=torch.device('cuda')
for (B,T) in [(2,256),(1,1048),(1,3000),(64,128),(32,4328)]:
    d=R.get_inputs({"batch_size":B,"time_dim":T},dev)
    a=R.run(**d); b=K.run(**d)
    print(B,T,tuple(a.shape),tuple(b.shape),(a.float()-b.float()).abs().max().item(), a.abs().max().item())
