from .sim_utils import *
from torchode import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
from .observators.observator2D import *
from tqdm import tqdm

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class LZ96(Simulator):
    def __init__(self, N=5, F=8, noise=0.1):
        super().__init__()
        self.N = N
        self.F = F
        self.noise_amp = noise

    def generate_steps(self, x0, t_vect, observe=True):
        super().generate_steps(x0, t_vect, observe)

        print("Generating steps")
        sol = solve_ivp(self.odefun, x0, t_vect)
        self.data = sol.ys

        if observe:
            print("Starting observations")
            self.observe(data=self.data)

    def odefun(self, t, init_state):
        D = torch.zeros(init_state.shape)
        for i in range(self.N):
            D[..., i] = (
                (init_state[..., (i + 1) % self.N] - init_state[..., i - 2])
                * init_state[..., i - 1]
                - init_state[..., i]
                + self.F
            )

        return D

    def display_sim(self, idx=0, obs=False, filename=None, delay=0.5, show=True):
        data = self.data[idx].T
        plt.imshow(data, cmap="plasma", interpolation="spline16")
        plt.colorbar()

        if show:
            plt.show(block=False)
            plt.pause(delay)
            if filename is not None:
                plt.savefig(filename + ".pdf")

            plt.clf()

            if obs:
                data = self.obs[idx].T
                plt.imshow(data, cmap="plasma", interpolation="spline16")
                plt.colorbar()
                plt.show(block=False)
                plt.pause(delay)
                if filename is not None:
                    plt.savefig(filename + "_obs.pdf")

        plt.close()

    def observe(self, data=None, tspan=1):
        if data is None:
            data = self.data
        print(data.shape)
        self.obs = self.observer.observe(data.unsqueeze(-1)).squeeze(-1)
        print(self.obs.shape)

        for i in tqdm(range(self.obs.shape[0])):
            self.obs[i] += self.obs[i] * self.noise_amp * torch.randn_like(self.obs[i])

    def __str__(self):
        str = "==========\nLZ96 model\n=========="
        return str + super().__str__()


if __name__ == "__main__":
    import torch

    torch.manual_seed(33)

    syst = LZ96(N=10, noise=0)
    X0 = 10 * torch.rand((1000, syst.N))

    n_steps = 100
    t_eval = torch.linspace(0, 5, n_steps)

    o = ObservatorStation2D((10, 1), (2, 1), (1, 0), (2, 0), (0.75, 1))

    syst.init_observer(o)
    syst.generate_steps(X0, t_eval, observe=True)
    syst.display_sim(obs=True, delay=1)
    syst.save_raw("")

    # syst.save_h5_data(filename = "LZ96_data/LZ_10pt_")

    # print(syst)
