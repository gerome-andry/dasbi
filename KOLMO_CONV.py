# borrowed @ francois-rozet

import h5py
import json
import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import seaborn as sns
import time
import wandb

from dawgz import job, schedule
from pathlib import Path
from tqdm import trange
from typing import *

from zuko.distributions import DiagNormal
from zuko.flows import Unconditional
from dasbi.inference.models import ConvNPE as NPE
from dasbi.networks.embedding import EmbedObs
from dasbi.simulators.sim_2D import LZ2D as sim


SCRATCH = os.environ.get("SCRATCH", ".")
PATH = Path(SCRATCH) / "npe_2D/lz96"
PATH.mkdir(parents=True, exist_ok=True)

fact = 5
N_grid = [2**i for i in range(5,6)]
Y_grid = [int(np.ceil(x/6)) for x in N_grid]
lN = len(N_grid)
window = 10
max_epochs = 2048


CONFIG = {
    # Architecture
    "embedding": [3]*lN,
    "kernel_size": [3]*lN,
    "ms_modules": [4]*lN,
    "num_conv": [3]*lN,
    "N_ms": [3],
    # Training
    # "epochs": [512]*lN,
    "batch_size": [512]*lN,
    "step_per_batch": [32]*lN,
    "optimizer": ["AdamW"]*lN,
    "learning_rate": [1e-5]*lN,  # np.geomspace(1e-3, 1e-4).tolist(),
    "weight_decay": [1e-4]*lN,  # np.geomspace(1e-2, 1e-4).tolist(),
    "scheduler": ["linear"]*lN,  # , 'cosine', 'exponential'],
    # Data
    "points": N_grid,
    "noise": [0.5]*lN,
    "train_sim": [819]*lN,
    "val_sim": [102]*lN,
    "device": ['cuda']*lN,
    # Test with assimilation window
    "x_dim": [(1, 2, sp, sp) for sp in N_grid],
    "y_dim": [(1, 2*window, spy, spy) for spy in Y_grid],
    "y_dim_emb": [(1, 11, sp, sp) for sp in N_grid],
    'obs_mask': [True]*lN, #+1 in y_dim
    'ar': [False]*lN, #+1 in y_dim_emb (for modargs not embnet)
    'roll':[True]*lN,
    "observer_fp": [f"experiments/observer2D.pickle" for N in N_grid],
}


def build(**config):
    mod_args = {
        "x_dim": torch.tensor(config["x_dim"]),
        "y_dim": torch.tensor(config["y_dim_emb"]),
        "n_modules": config["ms_modules"],
        "n_c": config["num_conv"],
        "k_sz": torch.tensor((config["kernel_size"], config["kernel_size"])),
        "type": "2D",
    }

    N = config["points"]
    base = Unconditional(
        DiagNormal,
        torch.zeros(2*N*N),
        torch.ones(2*N*N),
        buffer=True,
    )

    mask = None 
    if config['obs_mask']:
        with open(config["observer_fp"], "rb") as handle:
            observer = pickle.load(handle)
        mask = observer.get_mask().to(config['device'])

    # exit()
    emb_out = torch.tensor(config["y_dim_emb"]) 
    if config['ar']:
        emb_out[1] -= 1

    emb_net = EmbedObs(
        torch.tensor(config["y_dim"]),
        emb_out,
        conv_lay=config["embedding"],
        observer_mask=mask
    )
    return NPE(config["N_ms"], base, emb_net, mod_args, roll = config["roll"], ar=config["ar"])


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

def vorticity(x):
    *batch, _, h, w = x.shape

    y = x.reshape(-1, 2, h, w)
    y = torch.nn.functional.pad(y, (1, 1, 1, 1), mode='circular')

    du, = torch.gradient(y[:, 0], dim=-1)
    dv, = torch.gradient(y[:, 1], dim=-2)

    y = du - dv
    y = y[:, 1:-1, 1:-1]
    y = y.reshape(*batch, h, w)

    return y

def coarsen(x, r=2):
    *batch, h, w = x.shape

    x = x.reshape(*batch, h // r, r, w // r, r)
    x = x.mean(axis=(-3, -1))

    return x

def load_data(file):
    filep = Path(SCRATCH) / file
    with h5py.File(filep, mode='r') as f:
        data = f['x'][:]

    data = torch.from_numpy(data)

    data = coarsen(data)

    return data

@job(array=fact*lN, cpus=3, gpus=1, ram="32GB", time="5-12:00:00")
def CONV_train(i: int):
    # config = {key: random.choice(values) for key, values in CONFIG.items()}
    config = {key : values[i%lN] for key,values in CONFIG.items()}

    with open(config["observer_fp"], "rb") as handle:
        observer = pickle.load(handle)

    gr = 'step' if window == 1 else 'assim'
    run = wandb.init(project="dasbi", config=config, group=f"LZ2D_{gr}")
    runpath = PATH / f"runs/{run.name}_{run.id}"
    runpath.mkdir(parents=True, exist_ok=True)

    with open(runpath / "config.json", "w") as f:
        json.dump(config, f)

    # Data
    # tmax = 50
    traj_len = 64 
    times = torch.arange(traj_len).float()

    simt = sim(N=config["points"], M=config["points"], noise=config["noise"])
    simt.init_observer(observer)
    simt.data = load_data('train.h5')
    simt.obs = simt.observe()
    simt.time = times[None,...].repeat(config["train_sim"],1)
    process_sim(simt)

    simv = sim(N=config["points"], M=config["points"], noise=config["noise"])
    simv.init_observer(observer)
    simv.data = load_data('valid.h5')
    simv.obs = simv.observe()
    simv.time = times[None,...].repeat(config["val_sim"],1)
    process_sim(simv)

    col = sns.color_palette("icefire", as_cmap=True)

    gt, obs = simv.data[0,traj_len//2].cuda(),\
                simv.obs[0,traj_len//2].cuda()
    gt = vorticity(gt[None,...]).squeeze()
    plt.imshow(gt.cpu(), cmap=col)
    plt.title('GT')
    run.log({"GT state" : wandb.Image(plt)})
    plt.close()

    obs = vorticity(obs[None,...]).squeeze()
    plt.imshow(obs.cpu(), cmap=col)
    plt.title('GT')
    run.log({"GT obs" : wandb.Image(plt)})
    plt.close()

    # Network
    conv_nse = build(**config).cuda()
    # wandb.watch(conv_npe, log = 'all', log_freq = 128)
    size = sum(param.numel() for param in conv_nse.parameters())
    run.config.num_param = size

    # Training
    # epochs = config["epochs"]
    batch_size = config["batch_size"]
    step_per_batch = config["step_per_batch"]
    best = 1000
    prev_loss = best
    time_buff = 1024
    count = 0
    ## Optimizer
    if config["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(
            conv_nse.parameters(),
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
            sh_y = yb.shape
            x, y, t = (
                xb[subset_data],
                torch.cat([(yb[i - window + 1 : i + 1].reshape(window*2, sh_y[-2], sh_y[-1])).unsqueeze(0) for i in subset_data], dim=0),
                tb[subset_data],
            )
            # x = x[:, None, ...]
            
            optimizer.zero_grad()
            l = conv_nse.loss(x, y, t)
            l.backward()
            norm = torch.nn.utils.clip_grad_norm_(conv_nse.parameters(), 1)
            if torch.isfinite(norm):
                optimizer.step()
            
                losses_train.append(l.detach())

        end = time.time()

        ### Valid
        i = np.random.choice(
            len(simv.data),
            size=batch_size//8,
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

                sh_y = yb.shape
                x, y, t = (
                    xb[subset_data],
                    torch.cat([(yb[i - window + 1 : i + 1].reshape(window*2, sh_y[-2], sh_y[-1])).unsqueeze(0) for i in subset_data], dim=0),
                    tb[subset_data],
                )
                # x = x[:, None, ...]
                
                losses_val.append(conv_nse.loss(x, y, t))

            gt, obs, tm = simv.data[0,traj_len//2].cuda(),\
                            (simv.obs[0,traj_len//2-window+1:traj_len//2 + 1].reshape(2*window, sh_y[-2], sh_y[-1])).cuda(),\
                            simv.time[0,traj_len//2].cuda()
            # gt = vorticity(gt[None,...]).squeeze()
            # plt.imshow(gt.cpu(), cmap=col)
            # plt.title('GT')
            # run.log({"GT state" : wandb.Image(plt)})
            # plt.close()
            # plt.imshow(obs[-1], cmap=col)
            # plt.title('GT obs')
            # run.log({"GT observation" : wandb.Image(plt)})
            # plt.close()
            if epoch %10 == 0:
                samp = conv_nse.sample(obs[None,...], tm[None,...], 1).squeeze(0)
                obs_samp = simv.observe(samp.cpu())
                samp = vorticity(samp).squeeze()
                plt.imshow(samp.cpu(), cmap=col)
                plt.title('SAMPLE')
                run.log({"Sampled state" : wandb.Image(plt)})
                plt.close()

                obs_samp = vorticity(obs_samp).squeeze()
                plt.imshow(obs_samp.cpu(), cmap=col)
                plt.title('SAMPLE')
                run.log({"Sampled obs" : wandb.Image(plt)})
                plt.close()
            # plt.imshow(gt, cmap=col)
            # plt.title('GT')
            # run.log({"GT state" : wandb.Image(plt)})
            # plt.close()

        ### Logs
        if losses_train:
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
            if (prev_loss - loss_val) > 1e-5:
                prev_loss = loss_val
                torch.save(
                    conv_nse.state_dict(),
                    runpath / f"checkpoint.pth",
                )
                count = 0
            else:
                count += 1
        else:
            print('skip this step')
            
        epoch += 1

        if count == time_buff or epoch == max_epochs:
            break

        scheduler.step()

    run.finish()


if __name__ == "__main__":
    schedule(
        CONV_train,
        name="LZ2D_train",
        backend="slurm",
        settings={"export": "ALL"},
        env=[
            "conda activate DASBI",
            "export WANDB_SILENT=true",
        ],
    )
