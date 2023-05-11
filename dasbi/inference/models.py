from ..networks.nfmodules import MSConv
from ..networks.score import ScoreAttUNet
from ..diagnostic.classifier import TimeEmb, pos_embed
import torch
import torch.nn as nn
import numpy as np
from tqdm import trange, tqdm
import math
import matplotlib.pyplot as plt
import os               

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class VPScorePosterior(nn.Module):
    def __init__(self, emb_net, state_dim, targ_c, eps = 1e-3, **score_args):
        super().__init__()
        self.score = ScoreAttUNet(**score_args) # condition in the score input
        # self.score = MLP(**score_args)
        self.alpha = lambda t : torch.cos(math.acos(math.sqrt(eps))*t)**2
        self.embed = emb_net
        self.epsilon = eps 
        self.x_dim = state_dim
        self.x_imp = nn.Conv2d(self.x_dim[1], targ_c, 1)
        # t = torch.linspace(1,0,65)
        # plt.plot(t[:-1], self.mu(t[:-1]))
        # plt.plot(t[:-1], self.sigma(t[:-1]))
        # plt.plot(t[:-1], self.mu(t[:-1]))
        # plt.plot(t[:-1], self.mu(t[:-1] - 1/64)/self.mu(t[:-1]))
        # plt.show()

    def mu(self, t):
        return self.alpha(t)

    def sigma(self, t):
        return (1 - self.alpha(t) ** 2 + self.epsilon**2).sqrt()
    
    def forward(self, x, t):# x -> shape of x
        e = torch.randn_like(x)
        x = self.mu(t[...,None,None,None])*x + self.sigma(t[...,None,None,None])*e

        return x, e #sample x_t from p(x_t|x), rescaled target N(0,I)
    
    def loss(self, x, y, t):
        noise_t = torch.rand((x.shape[0])).to(x)
        y_emb = self.embed(y, t)
        x, scaled_target = self(x, noise_t)
        # dims = x.shape
        # print(x.shape, y_emb.shape)

        return (scaled_target - 
                self.score(torch.cat((y_emb, self.x_imp(x)), dim = 1), noise_t)).square().mean()
        # return (scaled_target - 
        #         self.score(torch.cat((y_emb, x), dim = 1).flatten(start_dim = 1), noise_t).reshape(dims)).square().mean()

    def sample(self, y, t, n, steps = 100, x_ref = None):
        denoise_time = torch.linspace(1,0,steps + 1).to(y)
        y_emb = self.embed(y, t)
        y_shapes = y_emb.shape
        y_emb = y_emb[None,...].expand(n,-1,-1,-1,-1).reshape((-1,) + y_shapes[-3:])
        dt = 1/steps 

        sample_sz = list(self.x_dim)
        sample_sz[0] = n*y_shapes[0]
        x = torch.randn(sample_sz).to(y)

        for t_n in denoise_time[:-1]:
            score_tn = t_n.unsqueeze(0).repeat(n*y_shapes[0])
            ratio = self.mu(t_n-dt)/self.mu(t_n)
            s = self.score(torch.cat((y_emb, self.x_imp(x)), dim = 1), score_tn)

            x = ratio*x + (self.sigma(t_n - dt) - 
                           ratio*self.sigma(t_n))*s
            
        return x.reshape((n,-1,) + self.x_dim[1:])


class VPScoreLinear(nn.Module):
    def __init__(self, state_dim, targ_c, observer, noise, gamma = 1e-2, eps = 1e-3, **score_args):
        super().__init__()
        self.score = ScoreAttUNet(**score_args) # condition in the score input
        self.alpha = lambda t : torch.cos(math.acos(math.sqrt(eps))*t)**2
        self.epsilon = eps 
        self.x_dim = state_dim
        self.x_imp = nn.Conv2d(self.x_dim[1], targ_c, 1)
        self.obs = observer 
        self.noise_sig = noise #spatial dim of y
        self.embed = TimeEmb(5)
        self.gam = gamma
        self.register_buffer('s_emb', pos_embed(self.x_dim[-2:]))

    def mu(self, t):
        return self.alpha(t)

    def sigma(self, t):
        return (1 - self.alpha(t) ** 2 + self.epsilon**2).sqrt()
    
    def forward(self, x, t):# x -> shape of x
        e = torch.randn_like(x)
        x = self.mu(t[...,None,None,None])*x + self.sigma(t[...,None,None,None])*e

        return x, e #sample x_t from p(x_t|x), rescaled target N(0,I)
    
    def loss(self, x, t):
        noise_t = torch.rand((x.shape[0])).to(x)
        t_emb = self.embed(t, x.shape[-2:])
        s_emb = self.s_emb.expand(x.shape[0],-1,-1,-1)
        x, scaled_target = self(x, noise_t)

        return (scaled_target - 
                self.score(torch.cat((t_emb, s_emb, self.x_imp(x)), dim = 1), noise_t)).square().mean()
    
    def composed_rscore(self, st, x, y, noise_t, scales):
        # t_emb = self.embed(t, x.shape[-2:])
        s_m = self.score(torch.cat((st,self.x_imp(x)), dim = 1), noise_t)
        mu, sigma = self.mu(noise_t[0]), self.sigma(noise_t[0])

        if sigma / mu > 2:
            return s_m
        
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)

            x_hat = (x - sigma*s_m)/mu

            device = x_hat.device
            x_hat = self.obs.observe(x_hat.cpu()).to(device)
            
            var = (self.noise_sig/scales)**2 + self.gam*(sigma/mu)**2
            mlogp = (((y - x_hat)**2)/var).sum()/2

        s_l, = torch.autograd.grad(mlogp, x)
        return s_m + sigma*s_l


    def sample(self, y, t, n, scales, steps = 100, corr = 3, tau = .25):
        denoise_time = torch.linspace(1,0,steps + 1).to(y)
        t_emb = self.embed(t, self.x_dim[-2:])
        s_emb = self.s_emb.expand(y.shape[0],-1,-1,-1)#deal with dim
        st_emb = torch.cat((t_emb, s_emb), dim = 1)

        st_shapes = st_emb.shape
        st_emb = st_emb[None,...].expand(n,-1,-1,-1,-1).reshape((-1,) + st_shapes[-3:]) 

        y = y[None,...].expand(n,-1,-1,-1,-1).reshape((-1,) + y.shape[-3:])
        
        dt = 1/steps 

        sample_sz = list(self.x_dim)
        sample_sz[0] = n*st_shapes[0]
        x = torch.randn(sample_sz).to(y)

        for t_n in tqdm(denoise_time[:-1]):
            score_tn = t_n.unsqueeze(0).repeat(n*st_shapes[0])
            ratio = self.mu(t_n-dt)/self.mu(t_n)
            s = self.composed_rscore(st_emb, x, y, score_tn, scales) #USE COMPOSED SCORE !!

            x = ratio*x + (self.sigma(t_n - dt) - 
                           ratio*self.sigma(t_n))*s
            
            for _ in range(corr):
                eps = torch.randn_like(x)
                s = -self.composed_rscore(st_emb, x, y, score_tn - dt, scales) / self.sigma(t_n - dt)
                delta = tau / s.square().mean(dim=(1,2,3), keepdim=True)

                x = x + delta * s + torch.sqrt(2 * delta) * eps
            # plt.clf()
            # plt.imshow(x.squeeze().item())
            # plt.show(block = False)
            # plt.pause(.1)

        return x.reshape((n,-1,) + self.x_dim[1:])
    

class MafNPE(nn.Module):
    def __init__(self, emb, NSF):
        super().__init__()
        self.flow = NSF
        self.embed = emb
    
    def forward(self, x, y, t):
        y_t = self.embed(y,t)
        batch = x.shape[0]
        return self.flow(x.reshape((batch, -1)), y_t.reshape(batch, -1))

    def sample(self, y, t, n):
        y_t = self.embed(y,t)
        batch = y_t.shape[0]
        return self.flow.flow(y_t.reshape((batch, -1))).sample((n,))
    
    def loss(self, x, y, t):
        log_p = self(x,y,t)
        return -log_p.mean()


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
        n_iter = 1 if max_samp is None else int(np.ceil(n/max_samp))
        for _ in range(n_iter):
            ns = int(np.minimum(n, (max_samp if max_samp is not None else np.inf)))
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
        z.clamp(-10,10)
        return -(ladj + self.base_dist().log_prob(z)).mean()
