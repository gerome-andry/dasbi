from dasbi.simulators.sim_lorenz96 import LZ96 as sim
from dasbi.simulators.observators.observator2D import ObservatorStation2D

from zuko.distributions import DiagNormal
from zuko.flows import Unconditional

import torch

from tqdm import tqdm

from LZ96 import build

# GENERATE DATA AND OBSERVATIONS
torch.manual_seed(42)

import pickle

n_sim = 2**10
N = 32
directory = "SmallLZ"
modelfname = "LZsmall1step.pth"
observerfname = "observer32narrowLZ.pickle"

simulator = sim(N=N, noise=0.1)
observer = ObservatorStation2D((N, 1), (4, 1), (2, 1), (4, 0), (2, 1))
with open(f"experiments/{observerfname}", "rb") as handle:
    observer = pickle.load(handle)
simulator.init_observer(observer)

# with open('experiments/observer32LZ.pickle', 'wb') as handle:
#     pickle.dump(observer, handle, protocol=pickle.HIGHEST_PROTOCOL)
# observer.visualize()

tmax = 10
traj_len = 1024
times = torch.linspace(0, tmax, traj_len)

simulator.generate_steps(torch.randn((n_sim, N)), times)

MUX = simulator.data.mean(dim=(0, 1))
SIGMAX = simulator.data.std(dim=(0, 1))

def preprocess_x(x):
    return (x - MUX) / SIGMAX


def postprocess_x(x):
    return x * SIGMAX + MUX


MUY = simulator.obs.mean(dim=(0, 1))
SIGMAY = simulator.obs.std(dim=(0, 1))


def preprocess_y(y):
    return (y - MUY) / SIGMAY


def postprocess_y(y):
    return y * SIGMAY + MUY


MUT = simulator.time.mean(dim=(0, 1))
SIGMAT = simulator.time.std(dim=(0, 1))

def preprocess_t(t):
    return (t - MUT) / SIGMAT


def postprocess_t(t):
    return t * SIGMAT + MUT


from lampe.plots import corner, mark_point
import matplotlib.pyplot as plt   
from matplotlib.animation import FuncAnimation

fig = plt.figure(figsize=(7,7))

def animate(i):
    fig = corner(simulator.obs[:, i], smooth=2, legend=f"p(y|t = {simulator.time[0, i]})")
    fig.show()

ani = FuncAnimation(fig, animate, interval=300)
plt.show()

exit()

start = 256
finish = 512
window = 1
# TRAIN A MODEL
simulator.data = simulator.data[:, start:finish]
simulator.obs = simulator.obs[:, start - window + 1 : finish]
simulator.time = simulator.time[:, start:finish]

simulator.display_sim(obs=True, filename=f"experiments/{directory}/hovGT")
simulator.data = preprocess_x(simulator.data)
simulator.obs = preprocess_y(simulator.obs)
simulator.time = preprocess_t(simulator.time)

base = Unconditional(
    DiagNormal,
    torch.zeros(N),
    torch.ones(N),
    buffer=True,
)

config = {
    "embedding": 3,
    "kernel_size": 2,
    "ms_modules": 1,
    "num_conv": 2,
    "N_ms": 2,
    # Training
    "epochs": 256,
    "batch_size": 32,
    "step_per_batch": 128,
    "optimizer": "AdamW",
    "learning_rate": 3e-3,  # np.geomspace(1e-3, 1e-4).tolist(),
    "weight_decay": 1e-4,  # np.geomspace(1e-2, 1e-4).tolist(),
    "scheduler": "linear",  # , 'cosine', 'exponential'],
    # Data
    "points": 32,
    "noise": 0.5,
    "train_sim": 2**10,
    "val_sim": 2**8,
    "device": "cuda",
    # Test with assimilation window
    "x_dim": (1, 1, 32, 1),
    "y_dim": (1, 1, 6, 1),
    "y_dim_emb": (1, 2, 32, 1),
    "obs_mask": False,
    "observer_fp": f"experiments/{observerfname}",
}

device = "cuda"
model = build(**config).to(device)

# with torch.no_grad():
#     model(
#         simulator.data[None, None, 0, 0, :, None].to(device),
#         simulator.obs[None, :window, 0, :, None].to(device),
#         simulator.time[0, 0, None].to(device),
#     )

# state = torch.load(f"{modelfname}", map_location=torch.device(device))

# 1 STEP :
with torch.no_grad():
    model(
        simulator.data[None, None, 0, 0, :, None].to(device),
        simulator.obs[None, None, 0, 0, :, None].to(device),
        simulator.time[0, 0, None].to(device),
    )

state = torch.load(f"{modelfname}", map_location=torch.device(device))

model.load_state_dict(state)
model.eval()

size = sum(param.numel() for param in model.parameters())
print(f"Model has {size} trainable parameters")

# EVALUATE CORNER PLOT
# x, y, t = (
#     simulator.data[0, window - 1],
#     simulator.obs[0, :window],
#     simulator.time[0, window - 1],
# )

# x = x[None, None, :, None]
# y = y[None, :, :, None]
# t = t.unsqueeze(-1)

# x_s = (
#     model.sample(y.to(device), t.to(device), 2**12, max_samp=2**8)
#     .squeeze()
#     .detach()
#     .cpu()
# )

# from lampe.plots import corner, mark_point

# fig = corner(x_s[:, ::5], smooth=2, figsize=(6.8, 6.8), legend="p(x | y*)")

# x_star = x.squeeze()[::5]
# mark_point(fig, x_star)

# fig.savefig(f"experiments/{directory}/cornerNPESim.pdf")

# y_s = simulator.observe(x_s)
# fig = corner(y_s, smooth=2, figsize=(6.8, 6.8), legend="p(y | y*)")

# y_star = y.squeeze()  # [-1]
# mark_point(fig, y_star)

# fig.savefig(f"experiments/{directory}/cornerNPEObs.pdf")
# fig.clear()

# EVALUATE TRAJECTORY

xgt, ygt, tgt = simulator.data[0], simulator.obs[0], simulator.time[0]

y = ygt[..., None]
t = tgt.unsqueeze(1)

x_s = []
y_s = []

# ASSIM :
# y = torch.cat(
#     [y[i : i + window].unsqueeze(0) for i, _ in enumerate(y[: -window + 1])], dim=0
# )
# 1 STEP :
y = y.unsqueeze(1)
samp = model.sample(y.to(device), t.to(device), 16, max_samp=1).squeeze().detach().cpu()

y_samp = simulator.observe(postprocess_x(samp)).mean((0))
samp = samp[0]  # .mean((0))

simulator.data = postprocess_x(samp[None, ...])
simulator.obs = y_samp[None, ...]

xgt = postprocess_x(xgt)
ygt = postprocess_y(ygt)
simulator.display_sim(
    obs=True,
    filename=f"experiments/{directory}/hovSAMP",
    minMax=(xgt.min(), xgt.max(), ygt.min(), ygt.max()),
)
