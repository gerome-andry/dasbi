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

simulator = sim(N=N, noise=0.5)
observer = ObservatorStation2D((N, 1), (4, 1), (2, 1), (4, 1), (2, 1))
with open("experiments/observer32moreLZ.pickle", "rb") as handle:
    observer = pickle.load(handle)
simulator.init_observer(observer)

# with open('experiments/observer32moreLZ.pickle', 'wb') as handle:
#     pickle.dump(observer, handle, protocol=pickle.HIGHEST_PROTOCOL)

tmax = 10
traj_len = 1024
times = torch.linspace(0, tmax, traj_len)

simulator.generate_steps(torch.randn((n_sim, N)), times)
print(simulator.obs.shape)

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

# print(MUT, SIGMAT)
def preprocess_t(t):
    return (t - MUT) / SIGMAT


def postprocess_t(t):
    return t * SIGMAT + MUT


# TRAIN A MODEL
simulator.data = simulator.data[:, 9 : 512 + 9]
simulator.obs = simulator.obs[:, : 512 + 10]
simulator.time = simulator.time[:, 9 : 512 + 9]
simulator.display_sim(obs=True, filename="hovGT")
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
    # Test with assimilation window
    "x_dim": (1, 1, 32, 1),
    "y_dim": (1, 10, 11, 1),
    "y_dim_emb": (1, 11, 32, 1),
    "observer_fp": "experiments/observer32LZ.pickle",
}

model = build(**config)

with torch.no_grad():
    model(
        simulator.data[None, None, 0, 0, :, None],
        simulator.obs[None, :10, 0, :, None],
        simulator.time[0, 0, None],
    )

state = torch.load("LZsmallAssimBigobs.pth", map_location=torch.device("cpu"))

# 1 STEP :
# with torch.no_grad():
#     model(simulator.data[None,None, 0, 0, :, None], simulator.obs[None, None, 0, 0, :, None], simulator.time[0, 0, None])

# state = torch.load('LZsmall1stepearly.pth', map_location = torch.device('cpu'))

model.load_state_dict(state)
model.eval()

# EVALUATE CORNER PLOT
# x,y,t = simulator.data[0, -1], simulator.obs[0, -10:], simulator.time[0, -1]

# x = x[None, None, :, None]
# y = y[None, :, :, None]
# t = t.unsqueeze(-1)

# x_s = model.sample(y, t, 2**13, max_samp = 2**7).squeeze().detach()


# # import lampe
# from lampe.plots import corner, mark_point

# fig = corner(x_s[:,::5], smooth=1, figsize=(6.8, 6.8), legend='p(x | y*)')

# x_star = x.squeeze()[::5]
# mark_point(fig, x_star)

# fig.savefig('cornerNPEtestSimAssimearly.pdf')

# y_s = simulator.observe(x_s)
# fig = corner(y_s, smooth=1, figsize=(6.8, 6.8), legend='p(y | y*)')

# y_star = y.squeeze()[-1]
# mark_point(fig, y_star)

# fig.savefig('cornerNPEtestObsfAssimearly.pdf')
# fig.clear()


# EVALUATE TRAJECTORY

xgt, ygt, tgt = simulator.data[0], simulator.obs[0], simulator.time[0]

# x = x[:, None, :, None]
y = ygt[..., None]
t = tgt.unsqueeze(1)

x_s = []
y_s = []

# ASSIM :
y = torch.cat([y[i : i + 10].unsqueeze(0) for i, _ in enumerate(y[:-10])], dim=0)
print(y.shape)
samp = model.sample(y, t, 16, max_samp=1).squeeze().detach()
print(samp.shape)
y_samp = simulator.observe(postprocess_x(samp)).mean((0))
samp = samp.mean((0))
# x_s.append(samp.unsqueeze(0))
# y_s.append(y_samp.unsqueeze(0))
# for yt,tt in tqdm(zip(y, t)):
#     samp = model.sample(yt.unsqueeze(0), tt.unsqueeze(0), 64).squeeze().detach()
#     y_samp = simulator.observe(postprocess_x(samp)).mean((0))
#     samp = samp.mean((0))
#     x_s.append(samp.unsqueeze(0))
#     y_s.append(y_samp.unsqueeze(0))

# x_s = torch.cat(x_s, dim = 0)
simulator.data = postprocess_x(samp[None, ...])
# y_s = torch.cat(y_s, dim = 0)
simulator.obs = y_samp[None, ...]

xgt = postprocess_x(xgt)
ygt = postprocess_y(ygt)
simulator.display_sim(
    obs=True,
    filename="hovSAMPAssimBig",
    minMax=(xgt.min(), xgt.max(), ygt.min(), ygt.max()),
)
