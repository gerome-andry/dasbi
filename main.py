from dasbi.simulators.sim_lorenz96 import LZ96 as sim 
from dasbi.simulators.observators.observator2D import ObservatorStation2D
from dasbi.networks.nfmodules import ConvNPE as NPE
from dasbi.networks.embedding import EmbedObs

from zuko.distributions import DiagNormal
from zuko.flows import Unconditional

import torch 

from tqdm import tqdm

# GENERATE DATA AND OBSERVATIONS
torch.manual_seed(42)

n_sim = 2**10
batch_size = 32
step_per_batch = 128

N = 32 

simulator = sim(N = N, noise=.5)
observer = ObservatorStation2D((32,1), (4,1), (2,1), (4,1), (2,1))
simulator.init_observer(observer)

tmax = 5
traj_len = 256
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

# print(MUT, SIGMAT)
def preprocess_t(t):
    return (t - MUT) / SIGMAT

def postprocess_t(t):
    return t * SIGMAT + MUT

# print(preprocess_t(torch.tensor(5.)))
# exit()

simulator.data = preprocess_x(simulator.data)
simulator.obs = preprocess_y(simulator.obs)
simulator.time = preprocess_t(simulator.time)

# TRAIN A MODEL 
base = Unconditional(
            DiagNormal,
            torch.zeros(N),
            torch.ones(N),
            buffer=True,
        )

x_dim = torch.tensor((1, 1, N, 1))
y_dim = torch.tensor((1, 2, N, 1))
model = NPE(x_dim, y_dim, base, 1, 3, torch.tensor((4, 1)), type = '1D')

y_dim = torch.tensor((1, 1, 6, 1))
emb_net = EmbedObs(y_dim, x_dim)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=0.5,
    min_lr=1e-5,
    patience=32,
    threshold=1e-2,
    threshold_mode='abs',
)

loss_plt = []
with tqdm(range(256), unit='epoch', ncols=88) as tq: #256 epoch
    for epoch in tq:
        subset_batch = torch.randint(len(simulator.data), (batch_size,)) #256 batch/epoch
        losses = []

        for xb,yb,tb in zip(simulator.data[subset_batch], simulator.obs[subset_batch], simulator.time[subset_batch]):
            subset_data = torch.randint(traj_len, (step_per_batch,))
            x,y,t = xb[subset_data], yb[subset_data], tb[subset_data]
            x = x[:,None,...,None]
            y = y[:,None,...,None]
            y = emb_net(y, t)
            l = model.loss(x, y)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            losses.append(l.item())


        l = sum(losses) / len(losses)

        loss_plt += [l]
        torch.save(model, f'mod_epoch_{epoch}.pt')
        torch.save(emb_net, f'emb_epoch_{epoch}.pt')

        tq.set_postfix(loss=l, lr=optimizer.param_groups[0]['lr'])
        scheduler.step(l)

import matplotlib.pyplot as plt 
plt.plot(loss_plt)
plt.show()