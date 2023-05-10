from dasbi.inference.models import VPScorePosterior
import torch
import numpy as np           
from LZ96_POSTS import build
from dasbi.simulators.sim_2D import LZ2D as sim
import os                   
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import matplotlib.pyplot as plt
import seaborn as sns 
import time

if __name__ == "__main__":
    s = sim(N = 32, M = 32)
    t = torch.linspace(0,20,200)
    torch.manual_seed(42)
    x0 = torch.randn((100,32, 32))
    # x0[:,:,:] = 0
    # x0[:,15,15] = torch.rand(1)
    tm = time.time()
    s.generate_steps(x0,t, observe=False)
    # print(s.data.shape)
    print(time.time() - tm)
    # mu = s.data[0,:,:,1:].mean(dim = 0)
    # std = s.data[0,:,:,:1:].std(dim = 0)
    col = sns.color_palette("icefire", as_cmap=True)
    # s.data = s.vorticity(s.data)
    # s.data = s.data.std(dim = 0)
    m, M = s.data.min(), s.data.max()
    # for d in range(8):
    #     plt.plot(s.data[0,:,d,d])
    # plt.show()
    for dat_t in s.data[0]:
        plt.clf()
        plt.imshow(dat_t, interpolation = 'spline16', cmap = col, vmin = m, vmax = M)
        plt.show(block=False)
        plt.pause(.1)

    exit()
    import os

    dp = {
        8 : 2,
        16 : 2,
        32 : 2,
        64 : 3,
        128 : 3,
        256 : 3,
        512 : 4
    }

    for i in range(3,10):
        N = 2**i
        window = 10
        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
        CONFIG = {
            # Architecture
            "embedding": 3,
            "depth": dp[N],
            "input_h": 45 + int(np.log2(N)),
            # Training
            # "epochs": [512]*lN,
            "batch_size": 128,
            "step_per_batch": 512,
            "optimizer": "AdamW",
            "learning_rate": 1e-3,  # np.geomspace(1e-3, 1e-4).tolist(),
            "weight_decay": 1e-4,  # np.geomspace(1e-2, 1e-4).tolist(),
            "scheduler": "linear",  # , 'cosine', 'exponential'],
            # Data
            "points": N,
            "noise": 0.5,
            "train_sim": 2**10,
            "val_sim": 2**8,
            "device": 'cpu',
            # Test with assimilation window
            "x_dim": (1, 1, N, 1),
            "y_dim": (1, window, N//4, 1),
            "y_dim_emb": (1, 11, N, 1),
            'obs_mask': True, #+1 in y_dim
            "observer_fp": f"experiments/observer{N}LZ.pickle",
        }



        m = build(**CONFIG)
        # print(m)
        print('NSE:',sum(param.numel() for param in m.parameters()))
    # print(m)
    # m(torch.zeros(CONFIG["x_dim"]), torch.ones((1,1)))