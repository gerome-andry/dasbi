from dasbi.inference.models import VPScorePosterior
import torch
import numpy as np           
from LZ96_POSTS import build

if __name__ == "__main__":
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
            "input_h": 32,
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