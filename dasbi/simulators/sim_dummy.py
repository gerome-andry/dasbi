from .sim_utils import *
import torch 


class Dummy(Simulator):
    def __init__(self, N = 8, noise = .5):
        super().__init__()
        self.N = N   
        self.noise_amp = noise  
        self.observable = True   

    def generate_steps(self, x0, t_vect, observe=True):
        super().generate_steps(x0, t_vect, observe)

        b = x0.shape[0]
        lg = len(t_vect)
        self.data = x0[..., None,:].repeat(1,lg,1)
        self.data += t_vect[None, ..., None]
        self.time = t_vect[None, ...].repeat(b, 1)
        if observe:
            self.obs = torch.zeros((b,self.N//4, lg))
            for i in range(self.N//4):
                self.obs[..., i] = self.data[...,4*i:4*(i+1)].sum(-1)

            self.obs += self.noise_amp*torch.randn_like(self.obs)


if __name__ == "__main__":
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    import matplotlib.pyplot as plt
    
    N = 400
    x0 = torch.randn((1, N))
    t_eval = torch.linspace(0,10,100)

    syst = Dummy(N = N)
    syst.generate_steps(x0, t_eval)
    
    for d in syst.data.squeeze():
        plt.plot(d)
    plt.show()

    for d in syst.obs.squeeze():
        plt.plot(d)
    plt.show()

