from ..networks.nfmodules import MSConv
import torch
import torch.nn as nn

class ConvNPE(nn.Module):
    def __init__(self, n_lay, base, emb_net, module_args):
        super().__init__()
        self.convmod = nn.ModuleList([MSConv(**module_args) for _ in range(n_lay)])
        self.embed = emb_net
        self.base_dist = base

    def forward(self, x, y, t):
        y_t = self.embed(y,t)
        ladj = x.new_zeros(x.shape[0])
        for mc in self.convmod:
            x, ladj_i = mc(x, y_t)
            ladj += ladj_i
        
        return x, ladj

    def inverse(self, z, y, t):
        y_t = self.embed(y,t)
        for mc in self.convmod:
            z, _ = mc(z, y_t)
        
        return x
    
    def sample(self, y, t, n, max_samp = None):
        assert y.shape[0] == 1, "Can only condition on a single observation for sampling"
        x_s = []
        while n > 0:
            ns = int(np.minimum(n, max_samp if max_samp is not None else np.inf)) 
            y = y.expand(ns, -1, -1, -1)
            z = self.base_dist().sample((ns,))
            s_dim = self.x_dim
            s_dim[0] = ns
            z = z.reshape(tuple(s_dim))
            x_s.append(self.inverse(z, y, t))
            n -= ns
        
        return torch.cat(x_s, dim = 0)

    def loss(self, x, y, t):
        z, ladj = self.forward(x,y,t)
        z = z.reshape((z.shape[0], -1))
        return -(ladj + self.base_dist().log_prob(z)).mean()