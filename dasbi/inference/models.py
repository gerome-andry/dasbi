from ..networks.nfmodules import MSConv
import torch
import torch.nn as nn
import numpy as np


class ConvNSE(nn.Module):
    pass


class ConvNPE(nn.Module):
    def __init__(self, n_lay, base, emb_net, module_args):
        super().__init__()
        self.convmod = nn.ModuleList([MSConv(**module_args) for _ in range(n_lay)])
        self.x_dim = module_args["x_dim"]
        self.embed = emb_net
        self.base_dist = base

    def forward(self, x, y, t):
        y_t = self.embed(y, t)
        ladj = x.new_zeros(x.shape[0])
        for mc in self.convmod:
            x_init = x.clone()
            x, ladj_i = mc(x, y_t)
            ladj += ladj_i

        return x, ladj

    def inverse(self, z, y, t):
        y_t = self.embed(y, t)
        for mc in reversed(self.convmod):
            z, _ = mc.inverse(z, y_t)

        return z

    def sample(self, y, t, n, max_samp=None):
        # assert y.shape[0] == 1, "Can only condition on a single observation for sampling"
        y_dim = y.shape
        t_dim = t.shape
        x_s = []
        while n > 0:
            ns = int(np.minimum(n, max_samp if max_samp is not None else np.inf))
            y_t = y.unsqueeze(0).expand(ns, -1, -1, -1, -1).reshape((-1,) + y_dim[1:])
            t_t = t.unsqueeze(0).expand(ns, -1, -1).reshape((-1,) + t_dim[1:])
            z = self.base_dist().sample((ns * y_dim[0],))
            s_dim = self.x_dim
            s_dim[0] = ns * y_dim[0]
            z = z.reshape(tuple(s_dim))
            x_s.append(self.inverse(z, y_t, t_t).reshape((ns, -1) + tuple(s_dim[1:])))
            n -= ns

        return torch.cat(x_s, dim=0)

    def loss(self, x, y, t):
        z, ladj = self.forward(x, y, t)
        z = z.reshape((z.shape[0], -1))
        return -(ladj + self.base_dist().log_prob(z)).mean()
