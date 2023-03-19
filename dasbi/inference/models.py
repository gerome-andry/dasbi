from ..networks.nfmodules import MSConv
import torch
import torch.nn as nn
import numpy as np
from tqdm import trange

class ConvNSE(nn.Module):
    pass


class ConvNPE(nn.Module):
    def __init__(self, n_lay, base, emb_net, module_args, roll = False, ar = False):
        super().__init__()
        self.convmod = nn.ModuleList([MSConv(**module_args) for _ in range(n_lay)])
        self.x_dim = module_args["x_dim"]
        self.embed = emb_net
        self.base_dist = base
        self.roll = roll
        self.autoreg = ar

    def forward(self, x, y, t, x_ar = None):
        y_t = self.embed(y, t)
        if self.autoreg:
            y_t = torch.cat((y_t, x_ar), dim = 1)

        ladj = x.new_zeros(x.shape[0])
        for mc in self.convmod:
            x, ladj_i = mc(x, y_t)
            ladj += ladj_i
            if self.roll:
                if self.convmod[0].type == '1D':
                    dim = x.shape[-2] 
                    x = x.roll(shifts = dim//8, dims = -2)
                else:
                    dim = x.shape[-2:] 
                    x = x.roll(shifts = (dim[0]//4, dim[1]//4), dims = (-2, -1))
                    
        return x, ladj

    def inverse(self, z, y, t, x_ar = None):
        y_t = self.embed(y, t)
        if self.autoreg:
            y_t = torch.cat((y_t, x_ar), dim = 1)

        for mc in reversed(self.convmod):
            if self.roll:
                if self.convmod[0].type == '1D':
                    dim = z.shape[-2] 
                    z = z.roll(shifts = -dim//8, dims = -2)
                else:
                    dim = z.shape[-2:] 
                    z = z.roll(shifts = (-dim[0]//4, -dim[1]//4), dims = (-2, -1))

            z, _ = mc.inverse(z, y_t)
            
        return z

    def sample(self, y, t, n, max_samp=None, x_ar = None):
        y_dim = y.shape
        t_dim = t.shape
        x_s = []
        n_iter = int(np.ceil(n/max_samp))
        for _ in trange(n_iter):
            ns = int(np.minimum(n, max_samp if max_samp is not None else np.inf))
            y_t = y.unsqueeze(0).expand(ns, -1, -1, -1, -1).reshape((-1,) + y_dim[1:])
            t_t = t.unsqueeze(0).expand(ns, -1, -1).reshape((-1,) + t_dim[1:])
            z = self.base_dist().sample((ns * y_dim[0],))
            s_dim = self.x_dim
            s_dim[0] = ns * y_dim[0]
            z = z.reshape(tuple(s_dim))
            x_art = None
            if self.autoreg:
                xar_dim = x_ar.shape
                x_art = x_ar.unsqueeze(0).expand(ns, -1, -1, -1, -1).reshape((-1,) + xar_dim[1:])
            x_s.append(self.inverse(z, y_t, t_t, x_ar=x_art).reshape((ns, -1) + tuple(s_dim[1:])))
            n -= ns

        return torch.cat(x_s, dim=0)

    def loss(self, x, y, t, x_ar = None):
        z, ladj = self.forward(x, y, t, x_ar=x_ar)
        z = z.reshape((z.shape[0], -1))
        return -(ladj + self.base_dist().log_prob(z)).mean()
