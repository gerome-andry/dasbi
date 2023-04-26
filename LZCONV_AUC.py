# borrowed @ francois-rozet

# import h5py
import json
import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import seaborn
import time
import wandb

from dawgz import job, schedule
from pathlib import Path
from tqdm import trange
from typing import *

from dasbi.simulators.sim_lorenz96 import LZ96 as sim
from dasbi.diagnostic.classifier import SampleCheck
from LZ96_CONV import build as buildSampler

# SCRATCH = os.environ.get("SCRATCH", ".")
# PATH = Path(SCRATCH) / "auc/lz96/conv_npe"
# PATH.mkdir(parents=True, exist_ok=True)

window = 10
N = 8
y_mode = False 

# N_grid = [2**i for i in range(3,10)]
# lN = len(N_grid)
nms_dict = {
    8: 2,
    16: 2,
    32: 2,
    64: 3,
    128: 3,
    256: 4,
    512: 4,
}
ms_mod = {
    8: 1,
    16: 1,
    32: 1,
    64: 2,
    128: 3,
    256: 4,
    512: 4,
}
CONFIG = {
    "observer_fp" : [f"experiments/observer{N}LZ.pickle"],
    "points" : [N],
    "noise" : [.5],
    "train_sim" : [2**10],
    "val_sim" : [2**8],
    "x_dim": [(1, 1, N, 1)],
    "y_dim": [(1, window, N//4, 1)],
    "epochs": [256],
    "batch_size": [128],
    "step_per_batch": [512],
    "optimizer": ["AdamW"],

    "embedding": [3],
    "kernel_size": [2],
    "ms_modules": [ms_mod[N]],
    "num_conv": [2],
    "N_ms": [nms_dict[N]],
    # Data
    "device": ['cuda'],
    "y_dim_emb": [(1, 11, N, 1)],
    'obs_mask': [True], #+1 in y_dim
    'ar': [False], #+1 in y_dim_emb (for modargs not embnet)
    'roll':[True],
}

def process_sim(simulator):
    MUX = simulator.data.mean(dim=(0, 1))
    SIGMAX = simulator.data.std(dim=(0, 1))

    MUY = simulator.obs.mean(dim=(0, 1))
    SIGMAY = simulator.obs.std(dim=(0, 1))

    MUT = simulator.time.mean(dim=(0, 1))
    SIGMAT = simulator.time.std(dim=(0, 1))

    simulator.data = (simulator.data - MUX) / SIGMAX
    simulator.obs = (simulator.obs - MUY) / SIGMAY
    simulator.time = (simulator.time - MUT) / SIGMAT


@job(array=5, cpus=2, gpus=1, ram="32GB", time="2-10:00:00")
def train_class(i: int):
    # config = {key: random.choice(values) for key, values in CONFIG.items()}
    run_idx = i%5
    # recover the good model ... 
    config = {key : values[0] for key,values in CONFIG.items()}

    with open(config["observer_fp"], "rb") as handle:
        observer = pickle.load(handle)

    gr = 'step' if window == 1 else 'assim'
    run = wandb.init(project="dasbi", config=config, group=f"LZ96_diag_{gr}")
    # runpath = PATH / f"runs/{run.name}_{run.id}"
    # runpath.mkdir(parents=True, exist_ok=True)

    # Data
    tmax = 50
    traj_len = 1024 
    times = torch.linspace(0, tmax, traj_len)

    simt = sim(N=config["points"], noise=config["noise"])
    simt.init_observer(observer)
    simt.generate_steps(torch.randn((config["train_sim"], config["points"])), times)
    process_sim(simt)

    simv = sim(N=config["points"], noise=config["noise"])
    simv.init_observer(observer)
    simv.generate_steps(torch.randn((config["val_sim"], config["points"])), times)
    process_sim(simv)

    # Network
    if y_mode:
        state_sz = list(config["y_dim"])
        state_sz[1] = 1
        classifier = SampleCheck(state_sz, config["y_dim"], 
                             reduce = int(np.log2(config["points"]//4)), 
                             type = '1D').to(config['device'])
    else:    
        classifier = SampleCheck(config["x_dim"], config["y_dim"], 
                             reduce = int(np.log2(config["points"])) - 1, 
                             type = '1D').to(config['device'])
    # So we always have 2 features at the end of convolution extraction part
    size = sum(param.numel() for param in classifier.parameters())
    run.config.num_param = size

    # Sampler 
    sampler = buildSampler(**config).to(config['device'])
    pts = config["points"]
    # modelfname = f"../checkpoints/LZ/CONV/{gr}/{pts}/{run_idx}.pth"
    modelfname = "experiments/test/test.pth"
    state = torch.load(f"{modelfname}", map_location=torch.device(config['device']))

    with torch.no_grad():
        sampler(torch.zeros(config["x_dim"]).to(config['device']), 
                torch.zeros(config["y_dim"]).to(config['device']), 
                torch.zeros(1).to(config['device']))
        # forward pass to init parameters
    sampler.load_state_dict(state)
    sampler.eval()

    # Training
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    step_per_batch = config["step_per_batch"]
    best = 0

    ## Optimizer
    if config["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(
            classifier.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
        )
    else:
        raise ValueError()

    lr = lambda t: 1 - (t / epochs)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr)

    ## Loop
    for epoch in trange(epochs, ncols=88):
        losses_train = []
        losses_val = []

        ### Train
        k = np.random.choice(
            len(simt.data),
            size=batch_size,
            replace=False
        )

        start = time.time()

        for xb, yb, tb in zip(
            simt.data[k].cuda(), simt.obs[k].cuda(), simt.time[k].cuda()
        ):
            subset_data = np.random.choice(
                np.arange(window - 1, traj_len),#because window of 10
                size=step_per_batch,
                replace=False,
            )
            sd_pos = subset_data[:step_per_batch//2]

            x, y, t = (
                xb[sd_pos],
                torch.cat([yb[idx - window + 1 : idx + 1].unsqueeze(0) for idx in subset_data], dim=0),
                tb[subset_data],
            )
            # ADD HERE SAMPLES FOR NEG
            # x_fake = model.sample...
            y = y[..., None]
            with torch.no_grad():
                x_fake = sampler.sample(y[step_per_batch//2:], t[step_per_batch//2:], 1).squeeze()
    
            x = torch.cat((x,x_fake), dim = 0)
            if y_mode:
                x = simt.observe(x.cpu()).to(config['device']) # create true and fake observations

            x = x[:, None, ..., None]
            
            labels = torch.zeros((len(subset_data), 2)).to(x)
            labels[:len(subset_data)//2, 0] = 1.
            labels[len(subset_data)//2:, 1] = 1.

            optimizer.zero_grad()
            l = classifier.loss(x, y, t, labels)
            l.backward()
            optimizer.step()

            losses_train.append(l.detach())

        end = time.time()

        ### Valid
        k = np.random.choice(
            len(simv.data),
            size=batch_size // 2,
            replace=False,
        )

        with torch.no_grad():
            for xb, yb, tb in zip(
                simv.data[k].cuda(), simv.obs[k].cuda(), simv.time[k].cuda()
            ):
                subset_data = np.random.choice(
                    np.arange(window - 1, traj_len),
                    size=step_per_batch,
                    replace=False,
                )

                sd_pos = subset_data[:step_per_batch//2]

                x, y, t = (
                    xb[sd_pos],
                    torch.cat([yb[idx - window + 1 : idx + 1].unsqueeze(0) for idx in subset_data], dim=0),
                    tb[subset_data],
                )
                # ADD HERE SAMPLES FOR NEG
                # x_fake = model.sample...
                y = y[..., None]
                x_fake = sampler.sample(y[step_per_batch//2:], t[step_per_batch//2:], 1).squeeze()
                x = torch.cat((x,x_fake), dim = 0)
                if y_mode:
                    x = simt.observe(x.cpu()).to(config['device'])
                x = x[:, None, ..., None]
                
                labels = torch.zeros((len(subset_data), 2)).to(x)
                labels[:len(subset_data)//2, 0] = 1.
                labels[len(subset_data)//2:, 1] = 1.
                
                _, _, auc = classifier.AUC(x, y, t, labels)
                losses_val.append(auc)

            if epoch%16 == 0:
                yb = simv.obs[-1].cuda()
                x,y,t = (simv.data[-1, window - 1:].cuda(), 
                         torch.cat([yb[idx - window + 1 : idx + 1].unsqueeze(0) for idx in range(window-1, traj_len)], dim=0), 
                         simv.time[-1, window - 1:].cuda())
                lg = len(t)
                
                y = y[..., None]
                x_fake = sampler.sample(y[lg//2:], t[lg//2:], 1).squeeze()
                x[lg//2:] = x_fake
                if y_mode:
                    x = simv.observe(x.cpu()).to(config['device'])
                x = x[:, None, ..., None]

                #x_fake are same as real
                labels = torch.zeros((lg, 2)).to(x)
                labels[:lg//2, 0] = 1.
                labels[lg//2:, 1] = 1.
                fpr, tpr, _ = classifier.AUC(x,y,t, labels)
                plt.plot(fpr, tpr)
                plt.title('ROC curve')
                plt.xlabel('FPR')
                plt.ylabel('TPR')
                run.log({"ROC_classif" : wandb.Image(plt)})
                plt.close()


        ### Logs
        loss_train = torch.stack(losses_train).mean().item()
        loss_val = torch.stack(losses_val).mean().item()

        run.log(
            {
                "loss": loss_train,
                "AUC": loss_val,
                "time_epoch": (end - start),
            }
        )

        ### Checkpoint
        if loss_val > best :
            best = loss_val
            # torch.save(
            #     classifier.state_dict(),
            #     runpath / f"checkpoint_{epoch:04d}.pth",
            # )

        scheduler.step()

    run.finish()


if __name__ == "__main__":
    # schedule(
    #     train_class,
    #     name="AUC BASELINE",
    #     backend="slurm",
    #     settings={"export": "ALL"},
    #     env=[
    #         "conda activate DASBI",
    #         "export WANDB_SILENT=true",
    #     ],
    # )
    train_class(0)
