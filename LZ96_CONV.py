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

from zuko.distributions import DiagNormal
from zuko.flows import Unconditional
from dasbi.inference.models import ConvNPE as NPE
from dasbi.networks.embedding import EmbedObs
from dasbi.simulators.sim_lorenz96 import LZ96 as sim


SCRATCH = os.environ.get("SCRATCH", ".")
PATH = Path(SCRATCH) / "npe_conv/lz96"
PATH.mkdir(parents=True, exist_ok=True)

N_grid = [8, 64, 128, 256, 512]#[2**i for i in range(3,10)]
Y_grid = [int(np.ceil(x/4)) for x in N_grid]
lN = len(N_grid)
window = 1
CONFIG = {
    # Architecture
    "embedding": [4]*lN,
    "kernel_size": [2]*lN,
    "ms_modules": [1 + k//256 for k in N_grid],
    "num_conv": [2]*lN,
    "N_ms": [2 + k//128 for k in N_grid],
    # Training
    "epochs": [512]*lN,
    "batch_size": [64]*lN,
    "step_per_batch": [512]*lN,
    "optimizer": ["AdamW"]*lN,
    "learning_rate": [3e-3]*lN,  # np.geomspace(1e-3, 1e-4).tolist(),
    "weight_decay": [1e-4]*lN,  # np.geomspace(1e-2, 1e-4).tolist(),
    "scheduler": ["linear"]*lN,  # , 'cosine', 'exponential'],
    # Data
    "points": N_grid,
    "noise": [0.5]*lN,
    "train_sim": [2**10]*lN,
    "val_sim": [2**8]*lN,
    "device": ['cuda']*lN,
    # Test with assimilation window
    "x_dim": [(1, 1, sp, 1) for sp in N_grid],
    "y_dim": [(1, window, spy, 1) for spy in Y_grid],
    "y_dim_emb": [(1, 5, sp, 1) for sp in N_grid],
    'obs_mask': [False]*lN, #+1 in y_dim
    'ar': [False]*lN, #+1 in y_dim_emb (for modargs not embnet)
    'roll':[True]*lN,
    "observer_fp": [f"experiments/observer{N}LZ.pickle" for N in N_grid],
}


def build(**config):
    mod_args = {
        "x_dim": torch.tensor(config["x_dim"]),
        "y_dim": torch.tensor(config["y_dim_emb"]),
        "n_modules": config["ms_modules"],
        "n_c": config["num_conv"],
        "k_sz": torch.tensor((config["kernel_size"], 1)),
        "type": "1D",
    }

    N = config["points"]
    base = Unconditional(
        DiagNormal,
        torch.zeros(N),
        torch.ones(N),
        buffer=True,
    )

    mask = None
    if config['obs_mask']:
        with open(config["observer_fp"], "rb") as handle:
            observer = pickle.load(handle)
        mask = observer.get_mask().to(config['device'])

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


@job(array=lN, cpus=2, gpus=1, ram="32GB", time="20:00:00")
def train(i: int):
    # config = {key: random.choice(values) for key, values in CONFIG.items()}
    config = {key : values[i] for key,values in CONFIG.items()}

    with open(config["observer_fp"], "rb") as handle:
        observer = pickle.load(handle)

    run = wandb.init(project="dasbi", config=config, group="LZ96_scaling_assim")
    runpath = PATH / f"runs/{run.name}_{run.id}"
    runpath.mkdir(parents=True, exist_ok=True)

    # with open(runpath / "config.json", "w") as f:
    #     json.dump(config, f)

    # Data
    tmax = 100
    traj_len = tmax*10 
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
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    step_per_batch = config["step_per_batch"]
    best = 1000

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
        lr = lambda t: 1 - (t / epochs)
    elif config["scheduler"] == "cosine":
        lr = lambda t: (1 + math.cos(math.pi * t / epochs)) / 2
    elif config["scheduler"] == "exponential":
        lr = lambda t: math.exp(-7 * (t / epochs) ** 2)
    else:
        raise ValueError()

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr)

    ## Loop
    for epoch in trange(epochs, ncols=88):
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
            x_ar = None
            if config['ar']:
                x_ar = xb[subset_data - 1]
                x_ar = x_ar[:, None, ..., None]
            
            optimizer.zero_grad()
            l = conv_npe.loss(x, y, t, x_ar)
            l.backward()
            optimizer.step()

            losses_train.append(l.detach())

        end = time.time()

        ### Valid
        i = np.random.choice(
            len(simv.data),
            size=batch_size // 4,
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
                x_ar = None
                if config['ar']:
                    x_ar = xb[subset_data - 1]
                    x_ar = x_ar[:, None, ..., None]

                losses_val.append(conv_npe.loss(x, y, t, x_ar))

        ### Logs
        loss_train = torch.stack(losses_train).mean().item()
        loss_val = torch.stack(losses_val).mean().item()

        run.log(
            {
                "loss": loss_train,
                "loss_val": loss_val,
                "time_epoch": (end - start),
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        ### Checkpoint
        if loss_val < best * 0.95:
            best = loss_val
            torch.save(
                conv_npe.state_dict(),
                runpath / f"checkpoint_{epoch:04d}.pth",
            )

        scheduler.step()

    # # Load best checkpoint
    # checkpoints = sorted(runpath.glob("checkpoint_*.pth"))
    # state = torch.load(checkpoints[-1])

    # conv_npe.load_state_dict(state)
    # conv_npe.eval()

    # # Evaluation
    # sime = sim(N=config["points"], noise=config["noise"])
    # sime.init_observer(observer)
    # sime.generate_steps(torch.randn((1, config["points"])), times)

    # x, y, t = sime.data[9], sime.obs[:10], sime.time[9]

    # x = x[:, None, :, None]
    # y = y[None, :, :, None]
    # t = t.unsqueeze(0)

    # traj = []
    # for yt, t in zip(y, t):
    #     samp = conv_npe.sample(yt.unsqueeze(0), t.unsqueeze(1), 3).cpu().numpy()
    #     traj.append(samp.unsqueeze(0))
    # traj = torch.cat(traj, dim=0).squeeze()

    # fig, axs = plt.subplots(4, 1, figsize=(7, 7))
    # figo, axso = plt.subplots(4, 1, figsize=(7, 7))

    # for i, (ax, axo) in enumerate(zip(axs.flat, axso.flat)):
    #     if i == 3:
    #         ax.imshow(x.squeeze().cpu().numpy(), cmap=seaborn.cm.coolwarm)
    #         axo.imshow(y.squeeze().cpu().numpy(), cmap=seaborn.cm.coolwarm)
    #     else:
    #         ax.imshow(traj[..., i], cmap=seaborn.cm.coolwarm)
    #         o = sime.observe(traj[..., i])
    #         axo.imshow(o, cmap=seaborn.cm.coolwarm)

    #     ax.label_outer()
    #     axo.label_outer()

    # fig.tight_layout()
    # figo.tight_layout()

    # run.log({"sample_traj": wandb.Image(fig)})
    # run.log({"sample_obs": wandb.Image(figo)})
    run.finish()


if __name__ == "__main__":
    schedule(
        train,
        name="Scaling LZ",
        backend="slurm",
        settings={"export": "ALL"},
        env=[
            "conda activate DASBI",
            "export WANDB_SILENT=true",
        ],
    )
