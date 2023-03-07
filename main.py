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
batch_size = 64
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

# TRAIN A MODEL 

# simulator.data = preprocess_x(simulator.data)
# simulator.obs = preprocess_y(simulator.obs)
# simulator.time = preprocess_t(simulator.time)

# base = Unconditional(
#             DiagNormal,
#             torch.zeros(N),
#             torch.ones(N),
#             buffer=True,
#         )

# x_dim = torch.tensor((1, 1, N, 1))
# y_dim = torch.tensor((1, 2, N, 1))
# model = NPE(x_dim, y_dim, base, 1, 3, torch.tensor((4, 1)), type = '1D')

# y_dim = torch.tensor((1, 1, 6, 1))
# emb_net = EmbedObs(y_dim, x_dim)

# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer,
#     factor=0.5,
#     min_lr=1e-5,
#     patience=32,
#     threshold=1e-2,
#     threshold_mode='abs',
# )

# loss_plt = []
# with tqdm(range(256), unit='epoch') as tq: #256 epoch
#     for epoch in tq:
#         subset_batch = torch.randint(len(simulator.data), (batch_size,))
#         losses = []

#         for xb,yb,tb in zip(simulator.data[subset_batch], simulator.obs[subset_batch], simulator.time[subset_batch]):
#             subset_data = torch.randint(traj_len, (step_per_batch,))
#             x,y,t = xb[subset_data], yb[subset_data], tb[subset_data]
#             x = x[:,None,...,None]
#             y = y[:,None,...,None]
#             y = emb_net(y, t)
#             l = model.loss(x, y)

#             optimizer.zero_grad()
#             l.backward()
#             optimizer.step()

#             losses.append(l.item())


#         l = sum(losses) / len(losses)

#         loss_plt += [l]
#         torch.save(model, f'mod_epoch_{epoch}.pt')
#         torch.save(emb_net, f'emb_epoch_{epoch}.pt')

#         tq.set_postfix(loss=l, lr=optimizer.param_groups[0]['lr'])
#         scheduler.step(l)

# import matplotlib.pyplot as plt 
# plt.plot(loss_plt)
# plt.show()

# EVALUATE THE MODEL 

simulator.generate_steps(torch.randn((1, N)), times)
simulator.data = preprocess_x(simulator.data)
simulator.obs = preprocess_y(simulator.obs)
simulator.time = preprocess_t(simulator.time)
simulator.display_sim(obs = True, filename='hovGT')

base = Unconditional(
            DiagNormal,
            torch.zeros(N),
            torch.ones(N),
            buffer=True,
        )

epoch = 255

x_dim = torch.tensor((1, 1, N, 1))
y_dim = torch.tensor((1, 2, N, 1))
model = NPE(x_dim, y_dim, base, 1, 3, torch.tensor((4, 1)), type = '1D')
model = torch.load(f'mod_epoch_{epoch}.pt')
model.eval()

y_dim = torch.tensor((1, 1, 6, 1))
emb_net = EmbedObs(y_dim, x_dim)
emb_net = torch.load(f'emb_epoch_{epoch}.pt')
emb_net.eval()

# EVALUATE CORNER PLOT
# x,y,t = simulator.data[0, 0], simulator.obs[0, 0], simulator.time[0, 0]

# x = x[None, None, :, None]
# y = y[None, None, :, None]
# t = t.unsqueeze(-1)

# y_t = emb_net(y, t)
# x_s = model.sample(y_t, 2**12, max_samp = 2**10).squeeze().detach()
# x_s = postprocess_x(x_s)
# print(x_s.shape)

# import lampe
# from lampe.plots import corner, mark_point

# fig = lampe.plots.corner(x_s[:,::5], smooth=1, figsize=(6.8, 6.8), legend="p(x | y*)")

# x_star = x.squeeze()[::5]
# lampe.plots.mark_point(fig, x_star)

# fig.savefig('cornerNPEtestSim0.pdf')

# y_s = simulator.observe(x_s)
# fig = lampe.plots.corner(y_s, smooth=1, figsize=(6.8, 6.8), legend="p(y | y*)")

# y_star = postprocess_y(y.squeeze())
# lampe.plots.mark_point(fig, y_star)

# fig.savefig('cornerNPEtestObs0.pdf')


# EVALUATE TRAJECTORY

x,y,t = simulator.data[0], simulator.obs[0], simulator.time[0]

x = x[:, None, :, None]
y = y[:, None, :, None]
t = t.unsqueeze(-1)

y_t = emb_net(y, t)
x_s = []
for yt in y_t:
    samp = model.sample(yt.unsqueeze(0), 1).squeeze().detach()
    x_s.append(samp.unsqueeze(0))

x_s = torch.cat(x_s, dim = 0)
simulator.data = x_s[None,...]
simulator.obs = simulator.observe()
simulator.display_sim(obs=True, filename='hovSAMP')