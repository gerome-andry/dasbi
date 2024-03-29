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

from lampe.inference import NPE
from zuko.flows import MAF

from dasbi.networks.embedding import EmbedObs
from dasbi.simulators.sim_lorenz96 import LZ96 as sim
from dasbi.inference.models import MafNPE

SCRATCH = os.environ.get("SCRATCH", ".")
PATH = Path(SCRATCH) / "npe_nsf/lz96"
PATH.mkdir(parents=True, exist_ok=True)

fact = 5
N_grid = [2**i for i in range(3,9)]
Y_grid = [int(np.ceil(x/4)) for x in N_grid]
lN = len(N_grid)
window = 10
max_epochs = 2048
CONFIG = {
    # Architecture
    "embedding": [3]*lN,
    "hf": [[32*int(k**0.5), ]*4 for k in N_grid],
    "tf": [3 + k//256 for k in N_grid],
    # Training
    # "epochs": [512]*lN,
    "batch_size": [128]*lN,
    "step_per_batch": [512]*lN,
    "optimizer": ["AdamW"]*lN,
    "learning_rate": [1e-3]*lN,  
    "weight_decay": [1e-4]*lN,  
    "scheduler": ["linear"]*lN, 
    # Data
    "points": N_grid,
    "noise": [0.5]*lN,
    "train_sim": [2**10]*lN,
    "val_sim": [2**8]*lN,
    "device": ['cuda']*lN,
    # Test with assimilation window
    "x_dim": [(1, 1, sp, 1) for sp in N_grid],
    "y_dim": [(1, window, spy, 1) for spy in Y_grid],
    "y_dim_emb": [(1, 11, sp, 1) for sp in N_grid],
    'obs_mask': [True]*lN,
    "observer_fp": [f"experiments/observer{N}LZ.pickle" for N in N_grid],
}


def build(**config):
    N = config["points"]

    mask = None
    if config['obs_mask']:
        with open(config["observer_fp"], "rb") as handle:
            observer = pickle.load(handle)
        mask = observer.get_mask().to(config['device'])

    emb_out = torch.tensor(config["y_dim_emb"]) 

    emb_net = EmbedObs(
        torch.tensor(config["y_dim"]),
        emb_out,
        conv_lay=config["embedding"],
        observer_mask=mask
    )
    myNSF = NPE(N, emb_out[1]*N, build = MAF, passes = 2, hidden_features = config['hf'], transforms = config['tf'], randperm = True)
    return MafNPE(emb_net, myNSF)


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

    ret_ls = [MUX, SIGMAX, MUY, SIGMAY, MUT, SIGMAT]

    return ret_ls


@job(array=fact*lN, cpus=2, gpus=1, ram="32GB", time="1-12:00:00")
def MAF_train(i: int):
    # config = {key: random.choice(values) for key, values in CONFIG.items()}
    config = {key : values[i%lN] for key,values in CONFIG.items()}

    with open(config["observer_fp"], "rb") as handle:
        observer = pickle.load(handle)

    gr = 'step' if window == 1 else 'assim'
    run = wandb.init(project="dasbi", config=config, group=f"LZ96_scaling_{gr}")
    runpath = PATH / f"runs/{run.name}_{run.id}"
    runpath.mkdir(parents=True, exist_ok=True)

    # with open(runpath / "config.json", "w") as f:
    #     json.dump(config, f)

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
    conv_npe = build(**config).cuda()
    size = sum(param.numel() for param in conv_npe.parameters())
    run.config.num_param = size

    # Training
    # epochs = config["epochs"]
    batch_size = config["batch_size"]
    step_per_batch = config["step_per_batch"]
    best = 1000
    prev_loss = best
    time_buff = 256
    count = 0
    ## Optimizer
    if config["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(
            conv_npe.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
    else:
        raise ValueError()

    if config["scheduler"] == "linear":
        lr = lambda t: 1 - (t / (1.5*max_epochs))
    # elif config["scheduler"] == "cosine":
    #     lr = lambda t: (1 + math.cos(math.pi * t / epochs)) / 2
    # elif config["scheduler"] == "exponential":
    #     lr = lambda t: math.exp(-7 * (t / epochs) ** 2)
    else:
        raise ValueError()

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr)

    ## Loop
    # for epoch in trange(epochs, ncols=88):
    epoch = 0
    while True:
        losses_train = []
        losses_val = []

        ### Train
        i = np.random.choice(
            len(simt.data),
            size=batch_size,
            replace=False
        )

        start = time.time()

        for xb, yb, tb in zip(
            simt.data[i].cuda(), simt.obs[i].cuda(), simt.time[i].cuda()
        ):
            subset_data = np.random.choice(
                np.arange(window - 1, traj_len),#because window of 10
                size=step_per_batch,
                replace=False,
            )

            x, y, t = (
                xb[subset_data],
                torch.cat([yb[i - window + 1 : i + 1].unsqueeze(0) for i in subset_data], dim=0),
                tb[subset_data],
            )
            x = x[:, None, ..., None]
            y = y[..., None]
            
            optimizer.zero_grad()
            l = conv_npe.loss(x, y, t)
            l.backward()
            norm = torch.nn.utils.clip_grad_norm_(conv_npe.embed.parameters(), 1)
            if torch.isfinite(norm):
                optimizer.step()

            losses_train.append(l.detach())

        end = time.time()

        ### Valid
        i = np.random.choice(
            len(simv.data),
            size=batch_size//2,
            replace=False,
        )

        with torch.no_grad():
            for xb, yb, tb in zip(
                simv.data[i].cuda(), simv.obs[i].cuda(), simv.time[i].cuda()
            ):
                subset_data = np.random.choice(
                    np.arange(window - 1, traj_len),
                    size=step_per_batch,
                    replace=False,
                )

                x, y, t = (
                    xb[subset_data],
                    torch.cat([yb[i - window + 1 : i + 1].unsqueeze(0) for i in subset_data], dim = 0),
                    tb[subset_data],
                )
                x = x[:, None, ..., None]
                y = y[..., None]
                
                losses_val.append(conv_npe.loss(x, y, t))

        ### Logs
        loss_train = torch.stack(losses_train).mean().item()
        loss_val = torch.stack(losses_val).mean().item()
        
        run.log(
            {
                "loss": loss_train,
                "loss_val": loss_val,
                "time_epoch": (end - start),
                "lr": optimizer.param_groups[0]["lr"],
                "plateau_buffer": count,
                "epoch": epoch
            }
        )

        ### Checkpoint
        if (prev_loss - loss_val) > 5e-2:
            prev_loss = loss_val
            torch.save(
                conv_npe.state_dict(),
                runpath / f"checkpoint.pth",
            )
            count = 0
        else:
            count += 1

        epoch += 1

        if count == time_buff or epoch == max_epochs:
            break

        scheduler.step()

    run.finish()


if __name__ == "__main__":
    schedule(
        MAF_train,
        name="Scaling LZ",
        backend="slurm",
        settings={"export": "ALL"},
        env=[
            "conda activate DASBI",
            "export WANDB_SILENT=true",
        ],
    )
