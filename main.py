from dasbi.simulators.sim_lorenz96 import LZ96 as sim
from dasbi.simulators.observators.observator2D import ObservatorStation2D

from zuko.distributions import DiagNormal
from zuko.flows import Unconditional

import torch

from tqdm import tqdm

from LZ96_LINP import build

# GENERATE DATA AND OBSERVATIONS
torch.manual_seed(42)

import pickle

from lampe.inference import NPE     
from zuko.flows import NSF 
from dasbi.networks.embedding import EmbedObs
import numpy as np 
import matplotlib.pyplot as plt   
import os          
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

class myMOD(torch.nn.Module):
    def __init__(self, emb, NSF):
        super().__init__()
        self.flow = NSF
        self.emb = emb 
# exit()
for i in range(3,9):
    N = 2**i
# exit()

# n_sim = 2**10
# N = 8
# directory = "test"
# modelfname = f"experiments/{directory}/test.pth"
    observerfname = f"observer{N}LZ.pickle"

# simulator = sim(N=N, noise=0.5)
# observer = ObservatorStation2D((N, 1), (3, 1), (1, 1), (2, 0), (.8, 1))
# with open(f"experiments/{observerfname}", "rb") as handle:
#     observer = pickle.load(handle)
# simulator.init_observer(observer)


# with open('experiments/observer32LZ.pickle', 'wb') as handle:
#     pickle.dump(observer, handle, protocol=pickle.HIGHEST_PROTOCOL)
# observer.visualize()

# tmax = 50
# traj_len = 1024 
# times = torch.linspace(0, tmax, traj_len)

# simulator.generate_steps(torch.randn((n_sim, N)), times)
# # print(simulator.data.shape)

# MUX = simulator.data.mean(dim=(0, 1))
# SIGMAX = simulator.data.std(dim=(0, 1))

# def preprocess_x(x):
#     return (x - MUX) / SIGMAX


# def postprocess_x(x):
#     return x * SIGMAX + MUX


# MUY = simulator.obs.mean(dim=(0, 1))
# SIGMAY = simulator.obs.std(dim=(0, 1))


# def preprocess_y(y):
#     return (y - MUY) / SIGMAY


# def postprocess_y(y):
#     return y * SIGMAY + MUY


# MUT = simulator.time.mean(dim=(0, 1))
# SIGMAT = simulator.time.std(dim=(0, 1))

# def preprocess_t(t):
#     return (t - MUT) / SIGMAT


# def postprocess_t(t):
#     return t * SIGMAT + MUT

# # simulator.data = preprocess_x(simulator.data)
# # simulator.obs = preprocess_y(simulator.obs)
# # simulator.time = preprocess_t(simulator.time)
# # simulator.display_sim(obs=True, delay = 10)

# start = 50
# finish = 100
    window = 1
# # TRAIN A MODEL
# simulator.data = simulator.data[:, start:finish]
# simulator.obs = simulator.obs[:, start - window + 1 : finish]
# simulator.time = simulator.time[:, start:finish]

# simulator.display_sim(obs=True, filename=f"experiments/{directory}/hovGT")
# simulator.data = preprocess_x(simulator.data)
# simulator.obs = preprocess_y(simulator.obs)
# simulator.time = preprocess_t(simulator.time)
# base = Unconditional(
#     DiagNormal,
#     torch.zeros(N),
#     torch.ones(N),
#     buffer=True,
# )

    nms_dict = {
        8: 2,
        16: 2,
        32: 2,
        64: 3,
        128: 3,
        256: 4,
        512: 4,
    }
    dp = {
        8 : 2,
        16 : 2,
        32 : 2,
        64 : 3,
        128 : 3,
        256 : 4
    }

    chan = {
        8 : 60,
        16 : 61,
        32 : 62,
        64 : 53,
        128 : 54,
        256 : 54
    }

    config = {
        "embedding": 3,
        "kernel_size": 2,
        "ms_modules": int(np.log(N)/np.log(4)) if N >= 128 else 1,
        "N_ms": nms_dict[N],
        "ms_modules": 1 + N//256,
        "num_conv": 2,
        "N_ms": 2 + N//128,
        "hf": [32*int(N**0.5), ]*4,
        "tf": 3 + N//256,
        "depth": dp[N],
        "input_h": chan[N],
        # Training
        "epochs": 256,
        "batch_size": 64,
        "step_per_batch": 64,
        "optimizer": "AdamW",
        "learning_rate": 3e-3,  # np.geomspace(1e-3, 1e-4).tolist(),
        "weight_decay": 1e-4,  # np.geomspace(1e-2, 1e-4).tolist(),
        "scheduler": "linear",  # , 'cosine', 'exponential'],
        # Data
        "points": N,
        "noise": 0.5,
        "train_sim": 2**10,
        "val_sim": 2**8,
        "device": "cpu",
        # Test with assimilation window
        "x_dim": (1, 1, N, 1),
        "y_dim": (1, window, N//4, 1),
        "y_dim_emb": (1, 11, N, 1),
        'obs_mask': True,
        'roll': True,
        'ar': False,
        "observer_fp": f"experiments/{observerfname}",
    }

# from math import sqrt
# model = NPE(N, 5*N, build = NSF, passes = 2, hidden_features = [4*int(sqrt(N)),4*int(sqrt(N))], transforms = 2 + N//256)
# emb_out = torch.tensor(config["y_dim_emb"])
# emb = EmbedObs(
#     torch.tensor(config["y_dim"]),
#     emb_out,
#     conv_lay=config["embedding"],
#     observer_mask=None
# )
# model = myMOD(emb, model)
# print('NSF:',sum(param.numel() for param in model.parameters()))

# print(config)
    device = "cpu"
    model = build(**config).to(device)
    print('NPE:',sum(param.numel() for param in model.parameters()))

# with torch.no_grad():
#     model(
#         simulator.data[None, None, 0, 0, :, None].to(device),
#         simulator.obs[None, :window, 0, :, None].to(device),
#         simulator.time[0, 0, None].to(device),
#     )

# print(model)
state = torch.load(f"{modelfname}", map_location=torch.device(device))

model.load_state_dict(state)
model.eval()

size = sum(param.numel() for param in model.parameters())
print(f"Model has {size} trainable parameters")

# EVALUATE CORNER PLOT
x, y, t = (
    simulator.data[0, 0],
    simulator.obs[0, :window],
    simulator.time[0, 0],
)

x_ar = None
x = x[None, None, :, None]
y = y[None, :, :, None]
t = t.unsqueeze(-1)
# if config['ar']:
#     x_ar = simulator.data[0, window - 2]
#     x_ar = x_ar[None, None, :, None].to(device)

plt.plot(x.squeeze())
plt.show()

with torch.no_grad():
    x_s = (
        model.sample(y.to(device), t.to(device), 2**10, SIGMAY)
        .squeeze()
        .detach()
        .cpu()
    )

from lampe.plots import corner, mark_point
points = [0,1,2,3,4,5,6,7]

fig = corner(x_s[:, points], smooth=2, figsize=(6.8, 6.8), legend="q(x | y*)")
fig = corner(simulator.data[:,0,points], smooth = 2, legend="p(x)", figure = fig)

x_star = x.squeeze()[points]
mark_point(fig, x_star)

fig.savefig(f"experiments/{directory}/cornerNPESim.pdf")

fig = corner(x_s[:, points], smooth=2, figsize=(6.8, 6.8), legend="q(x | y*)")
x_star = x.squeeze()[points]
mark_point(fig, x_star)

fig.savefig(f"experiments/{directory}/cornerNPESimPost.pdf")

y_s = simulator.observe(x_s)
fig = corner(y_s, smooth=2, figsize=(6.8, 6.8), legend="q(y | y*)")
fig = corner(simulator.obs[:,window - 1], smooth = 2, legend="p(y)", figure = fig)

y_star = y.squeeze()#[-1] 
mark_point(fig, y_star)

fig.savefig(f"experiments/{directory}/cornerNPEObs.pdf")
fig.clear()

# EVALUATE TRAJECTORY

xgt, ygt, tgt = simulator.data[0], simulator.obs[0], simulator.time[0]
if config['ar']:
    xgt, ygt, tgt = simulator.data[0,1:], simulator.obs[0,1:], simulator.time[0,1:]

x_ar = None
y = ygt[..., None]
t = tgt#.unsqueeze(1)
if config['ar']:
    x_ar = simulator.data[0,:-1]
    x_ar = x_ar[:, None, :, None].to(device)

# ASSIM :
# y = torch.cat(
#     [y[i : i + window].unsqueeze(0) for i, _ in enumerate(y[: -window + 1])], dim=0
# )
# 1 STEP :
y = y.unsqueeze(1)
samp = model.sample(y.to(device), t.to(device), 16).squeeze().detach().cpu()
# print(samp.shape)
y_samp = simulator.observe(postprocess_x(samp)).mean((0))
samp = samp[0] #.mean(0)

simulator.data = postprocess_x(samp[None, ...])
simulator.obs = y_samp[None, ...]

xgt = postprocess_x(xgt)
ygt = postprocess_y(ygt)
simulator.display_sim(
    obs=True,
    filename=f"experiments/{directory}/hovSAMP",
    minMax=(xgt.min(), xgt.max(), ygt.min(), ygt.max()),
)
