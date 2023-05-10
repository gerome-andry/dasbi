from .sim_utils import *
from torchode import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
from .observators.observator2D import *
from tqdm import tqdm
import seaborn as sns

class LZ2D(Simulator):
    def __init__(self, N, M, F=16,noise=0.1):
        super().__init__()
        self.N = N
        self.M = M
        self.F = F
        # self.c = c
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
        
        # Xm1, Xm2, Xp1 = [torch.roll(X, i, dims = -2) for i in (1, 2, -1)]
        # Ym1, Ym2, Yp1 = [torch.roll(X, i, dims = -1) for i in (1, 2, -1)]
        Xm1 = torch.roll(X, (1,1), dims = (-2, -1))
        Xm2 = torch.roll(Xm1, (1,1), dims = (-2,-1))
        Xp1 = torch.roll(X, (-1,-1), dims = (-2,-1))

        # dX = Xp1 - Xm1 + Yp1 - Ym1
        # dX2 =  Xp1 + Xm1 + Yp1 + Ym1 - 4*X

        dXdt = Xm1*(Xp1 - Xm2) - X + self.F 

        return dXdt.reshape(-1, self.N*self.M)
        # X = state.view(-1, self.N, self.M)
        
        # Xm1, Xm2, Xp1 = [torch.roll(X, i, dims = -2) for i in (1, 2, -1)]
        # Ym1, Ym2, Yp1 = [torch.roll(X, i, dims = -1) for i in (1, 2, -1)]

        # dX = Xm1*(Xp1 - Xm2) - X + self.Fx \
        #     + self.c * (Ym1*(Ym2 - Yp1) + X - self.Fy)
        
        # return dX.reshape(-1, self.N*self.M)
    
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