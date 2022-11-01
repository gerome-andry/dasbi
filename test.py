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

# data = torch.load('LZ96_data.pt')
# print(data.shape)
N = 40
B = 2**20
F = 8

LOWER = -F*torch.ones(N)
UPPER = F*torch.ones(N)

prior = zuko.distributions.BoxUniform(LOWER, UPPER)

def Lorenz96(t, X):
    """
    - X of the form (B, N) ; B for the number of samples and N for the number of grid points
    - t current time
    """
    D = torch.zeros(X.shape)
    for i in range(N):
        D[...,i] = (X[...,(i+1)%N] - X[...,i-2])*X[...,i-1] - X[...,i] + F
    
    return D

X0 = prior.sample((B,))
# X0 = X0.to('cuda')
t = time.time()

n_steps = 50
t_eval = torch.linspace(0, 5, n_steps)
sol = solve_ivp(Lorenz96, X0, t_eval)
print(sol.ys.shape)

# sol = []
# for x0 in X0:
#     sol.append(solve_ivp(Lorenz96, (0,5), x0, vectorized = False))

print(time.time() - t)

filepath = './LZ96_data_40pts.pt'

with open(filepath, 'ab') as torchFile:
    torch.save(sol.ys, torchFile)    

for xt in sol.ys[0]:
    plt.plot(xt)
    plt.axis([0, N - 1, -F - 2, F + 2])
    plt.show(block = False)
    plt.pause(0.1)
    plt.clf()