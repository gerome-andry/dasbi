from .sim_utils import *
from torchode import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
from .observators.observator2D import *
from tqdm import tqdm
import seaborn as sns

class LZ2D(Simulator):
    def __init__(self, N, M, F=16,noise=0.5):
        super().__init__()
        self.N = N
        self.M = M
        self.F = F
        self.noise_amp = noise

    def generate_steps(self, x0, t_vect, observe=True):
        super().generate_steps(x0, t_vect, observe)

        # print("Generating steps")
        sol = solve_ivp(self.odefun, x0.view(-1, self.N*self.M), t_vect)
        self.data = sol.ys.view(-1, len(t_vect), self.N, self.M)
        self.time = sol.ts
        # print(self.data.shape)

        if observe:
            # print("Starting observations")
            self.obs = self.observe(data=self.data)

    def odefun(self, t, state): #state (B, N, M+1)
        X = state.view(-1, self.N, self.M)
        
        Xm1 = torch.roll(X, (1,1), dims = (-2, -1))
        Xm2 = torch.roll(Xm1, (1,1), dims = (-2,-1))
        Xp1 = torch.roll(X, (-1,-1), dims = (-2,-1))

        dXdt = Xm1*(Xp1 - Xm2) - X + self.F 

        return dXdt.reshape(-1, self.N*self.M)

    def observe(self, data=None):
        if data is None:
            data = self.data
        # print(data.shape)
        observation = self.observer.observe(data)
        # print(self.obs.shape)

        for i in range(observation.shape[0]):
            observation[i] += self.noise_amp * torch.randn_like(observation[i])

        return observation
    
    def vorticity(self, x):# (B,T,N,M)
        b,t,n,m = x.shape

        y = torch.nn.functional.pad(x, (1, 1, 1, 1), mode = 'circular')
        print(y.shape)
        y = y.reshape(-1, n+2, m+2)
        print(y.shape)

        dx, = torch.gradient(y, dim = -2)
        dy, = torch.gradient(y, dim = -1)
        
        y = dx - dy 
        y = y[:,1:-1,1:-1]

        return y.reshape(b,t,n,m)