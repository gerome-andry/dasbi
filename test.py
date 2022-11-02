# from kolmogorov_methods import *
# from dynamical_system import KolmogorovFlow

# dyn_sys = KolmogorovFlow(64, 100, 25, 1e-2, 4)
# a = generate_data_kolmogorov(random.PRNGKey(42), dyn_sys, 1, 10, 0)

#python run_generate_training_data.py --config config_files/data_generation/kolmogorov_generate_data.config

# %matplotlib inline
#python run_generate_training_data.py --config config_files/data_generation/kolmogorov_generate_data.config

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import zuko

from itertools import islice
from tqdm import tqdm


from lampe.data import JointLoader
from lampe.inference import NPE, NPELoss
from lampe.plots import nice_rc, corner, mark_point
from lampe.utils import GDStep

from torchode import solve_ivp
import numpy as np

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import time 

# data = torch.load('LZ96_data_5pts.pt')
# print(data.shape)
# N = 40
# B = 2**20
F = 8

# LOWER = -F*torch.ones(N)
# UPPER = F*torch.ones(N)

# prior = zuko.distributions.BoxUniform(LOWER, UPPER)

torch.manual_seed(42)


class LZ96():
    def __init__(self):
        self.data = None
        self.obs = None 

    def Lorenz96(t, X):
        """
        - X of the form (B, N) ; B for the number of samples and N for the number of grid points
        - t current time
        """
        D = torch.zeros(X.shape)
        for i in range(N):
            D[...,i] = (X[...,(i+1)%N] - X[...,i-2])*X[...,i-1] - X[...,i] + F
        
        return D

    def loadData(self, filename):
        self.data = torch.load(filename)
        print(self.data.shape)

    def observe(self, span, noise_amp = .5):
        """
        The states are cyclic then X[-1] is linked to X[0]
        """
        if self.data is None:
            print("No data to observe !")
            return 

        N = self.data.shape[-1]
        data_vol = torch.prod(torch.tensor(self.data.shape[1:]))
        LIM_VOL = np.maximum(data_vol, 2**20)
        data_splitter = np.append(np.arange(0, self.data.shape[0], LIM_VOL//data_vol), self.data.shape[0])
        # print(data_splitter)

        span = np.clip(span, a_min = 1, a_max = N)

        from tqdm import tqdm
        idx = torch.arange(0, N, span)
        idxs = torch.stack([idx - 1, idx, (idx + 1)%N]).T
        for i in tqdm(range(1, len(data_splitter))):
            dat_obs = self.data[data_splitter[i-1]:data_splitter[i],..., idxs] 
            dat_obs += torch.randn_like(dat_obs) + dat_obs*noise_amp

            agg_obs = torch.mean(dat_obs, axis = -1)
            if self.obs is None:
                self.obs = agg_obs
            else:
                self.obs = torch.cat((self.obs, agg_obs), dim = 0)
        
        print(self.obs.shape)


sys = LZ96()
span = 4
sys.loadData('LZ96_data_5pts.pt')
sys.observe(span)

gs = sys.data.shape[2]
T = sys.data.shape[1]

for it in range(T):
    plt.plot(list(range(gs)), sys.data[0, it, :], label = 'state')
    plt.plot(torch.arange(0, gs, span), sys.obs[0, it, :], 'o', label = 'observations')
    plt.legend()

    plt.axis([0, gs - 1, -F - 2, F + 2])
    plt.show(block = False)
    plt.pause(0.1)
    plt.clf()

# X0 = prior.sample((B,))
# # X0 = X0.to('cuda')
# t = time.time()

# n_steps = 50
# t_eval = torch.linspace(0, 5, n_steps)
# sol = solve_ivp(Lorenz96, X0, t_eval)
# print(sol.ys.shape)

# # sol = []
# # for x0 in X0:
# #     sol.append(solve_ivp(Lorenz96, (0,5), x0, vectorized = False))

# print(time.time() - t)

# filepath = './LZ96_data_40pts.pt'

# with open(filepath, 'ab') as torchFile:
#     torch.save(sol.ys, torchFile)    

# for xt in sol.ys[0]:
#     plt.plot(xt)
#     plt.axis([0, N - 1, -F - 2, F + 2])
#     plt.show(block = False)
#     plt.pause(0.1)
#     plt.clf()