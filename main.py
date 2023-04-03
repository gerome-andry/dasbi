from dasbi.simulators.sim_lorenz96 import LZ96 as sim
from dasbi.simulators.observators.observator2D import ObservatorStation2D

from zuko.distributions import DiagNormal
from zuko.flows import Unconditional

import torch

from tqdm import tqdm

from LZ96_CONV import build

# GENERATE DATA AND OBSERVATIONS
torch.manual_seed(42)

import pickle

from lampe.inference import NPE     
from zuko.flows import NSF 
from dasbi.networks.embedding import EmbedObs

class myMOD(torch.nn.Module):
    def __init__(self, emb, NSF):
        super().__init__()
        self.flow = NSF
        self.emb = emb 

# for i in range(3,10):
#     N = 2**i
# exit()

n_sim = 2**10
N = 8
directory = "test"
modelfname = f"experiments/{directory}/test.pth"
observerfname = "observer8LZ.pickle"

simulator = sim(N=N, noise=0.5)
observer = ObservatorStation2D((N, 1), (3, 1), (1, 1), (2, 0), (.8, 1))
with open(f"experiments/{observerfname}", "rb") as handle:
    observer = pickle.load(handle)
simulator.init_observer(observer)


# with open('experiments/observer32LZ.pickle', 'wb') as handle:
#     pickle.dump(observer, handle, protocol=pickle.HIGHEST_PROTOCOL)
# observer.visualize()

tmax = 10
traj_len = tmax*10
times = torch.linspace(0, tmax, traj_len)

simulator.generate_steps(torch.randn((n_sim, N)), times)
print(simulator.data.shape)

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

simulator.data = preprocess_x(simulator.data)
simulator.obs = preprocess_y(simulator.obs)
simulator.time = preprocess_t(simulator.time)
# simulator.display_sim(obs=True, delay = 10)

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
    "embedding": 4,
    "kernel_size": 2,
    "ms_modules": 1 + N//256,
    "num_conv": 2,
    "N_ms": 2 + N//128,
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
    "y_dim": (1, 10, N//4, 1),
    "y_dim_emb": (1, 5, N, 1),
    'obs_mask': False,
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

device = "cpu"
model = build(**config).to(device)
print('NPE:',sum(param.numel() for param in model.parameters()))

with torch.no_grad():
    model(
        simulator.data[None, None, 0, 0, :, None].to(device),
        simulator.obs[None, :window, 0, :, None].to(device),
        simulator.time[0, 0, None].to(device),
        x_ar = simulator.data[None, None, 0, 0, :, None].to(device) if config['ar'] else None
    )

state = torch.load(f"{modelfname}", map_location=torch.device(device))

model.load_state_dict(state)
model.eval()

size = sum(param.numel() for param in model.parameters())
print(f"Model has {size} trainable parameters")

# EVALUATE CORNER PLOT
x, y, t = (
    simulator.data[0, window - 1],
    simulator.obs[0, :window],
    simulator.time[0, window - 1],
)

x_ar = None
x = x[None, None, :, None]
y = y[None, :, :, None]
t = t.unsqueeze(-1)
if config['ar']:
    x_ar = simulator.data[0, window - 2]
    x_ar = x_ar[None, None, :, None].to(device)

x_s = (
    model.sample(y.to(device), t.to(device), 2**12, max_samp=2**8, x_ar = x_ar)
    .squeeze()
    .detach()
    .cpu()
)

from lampe.plots import corner, mark_point
points = [0,1,2,
          14,15,16,
          29,30,31]

fig = corner(x_s[:, points], smooth=2, figsize=(6.8, 6.8), legend="q(x | y*)")
fig = corner(simulator.data[:,window - 1,points], smooth = 2, legend="p(x)", figure = fig)

x_star = x.squeeze()[points]
mark_point(fig, x_star)

fig.savefig(f"experiments/{directory}/cornerNPESim.pdf")

y_s = simulator.observe(x_s)
fig = corner(y_s, smooth=2, figsize=(6.8, 6.8), legend="q(y | y*)")
fig = corner(simulator.obs[:,window - 1], smooth = 2, legend="p(y)", figure = fig)

y_star = y.squeeze()[-1] 
mark_point(fig, y_star)

fig.savefig(f"experiments/{directory}/cornerNPEObs.pdf")
fig.clear()

# EVALUATE TRAJECTORY

xgt, ygt, tgt = simulator.data[0], simulator.obs[0], simulator.time[0]
if config['ar']:
    xgt, ygt, tgt = simulator.data[0,1:], simulator.obs[0,1:], simulator.time[0,1:]

x_ar = None
y = ygt[..., None]
t = tgt.unsqueeze(1)
if config['ar']:
    x_ar = simulator.data[0,:-1]
    x_ar = x_ar[:, None, :, None].to(device)

# ASSIM :
y = torch.cat(
    [y[i : i + window].unsqueeze(0) for i, _ in enumerate(y[: -window + 1])], dim=0
)
# 1 STEP :
# y = y.unsqueeze(1)
samp = model.sample(y.to(device), t.to(device), 16, max_samp=1, x_ar = x_ar).squeeze().detach().cpu()

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
