from dasbi.simulators.sim_lorenz96 import LZ96 as sim 
from dasbi.simulators.observators.observator2D import ObservatorStation2D
from dasbi.inference.models import ConvNPE as NPE
from dasbi.networks.embedding import EmbedObs

from zuko.distributions import DiagNormal
from zuko.flows import Unconditional

import torch 

from tqdm import tqdm

import wandb

# wandb.login()
# fbdbacb541a16364f2016c691c059166f30d86bb

# GENERATE DATA AND OBSERVATIONS
torch.manual_seed(42)

# sweep_configuration = {
#     'method': 'random',
#     'name': 'mysweep',
#     'metric': {
#         'goal': 'minimize', 
#         'name': 'train_loss'
#         },
#     'parameters': {
#         'batch_size': {'values': [16, 32, 64]},
#         'step_per_batch': {'max': 128, 'min': 64}
#      }
# }
# sweep_id = wandb.sweep(sweep=sweep_configuration, project='dasbi')
import pickle

n_sim = 2**10

N = 32 

simulator = sim(N = N, noise=.5)
observer = ObservatorStation2D((N,1), (4,1), (2,1), (4,1), (2,1))
with open('experiments/observer32LZ.pickle', 'rb') as handle:
    observer = pickle.load(handle)
simulator.init_observer(observer)

# with open('experiments/observer128LZ.pickle', 'wb') as handle:
#     pickle.dump(observer, handle, protocol=pickle.HIGHEST_PROTOCOL)

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

# print(MUT, SIGMAT)
def preprocess_t(t):
    return (t - MUT) / SIGMAT

def postprocess_t(t):
    return t * SIGMAT + MUT

# TRAIN A MODEL 

simulator.data = preprocess_x(simulator.data)[:,:200]
simulator.obs = preprocess_y(simulator.obs)[:,:200]
simulator.time = preprocess_t(simulator.time)[:,:200]
simulator.display_sim(obs = True, filename='hovGT')

base = Unconditional(
            DiagNormal,
            torch.zeros(N),
            torch.ones(N),
            buffer=True,
        )

x_dim = torch.tensor((1, 1, N, 1))
y_dim = torch.tensor((1, 1, 6, 1))
emb_net = EmbedObs(y_dim, x_dim, conv_lay=4)

y_dim_emb = torch.tensor((1, 2, N, 1))

mod_args = {'x_dim' : x_dim,
            'y_dim' : y_dim_emb,
            'n_modules' : 1,
            'n_c' : 2,
            'k_sz' : torch.tensor((2, 1)),
            'type' : '1D'}

model = NPE(1, base, emb_net, mod_args)

with torch.no_grad():
    model(simulator.data[None,None, 0, 0, :, None], simulator.obs[None,None, 0, 0, :, None], simulator.time[0, 0, None])

state = torch.load('checkpoint_0188.pth', map_location = torch.device('cpu'))

model.load_state_dict(state)
model.eval()

# wandb.init(
#     project = 'dasbi',
#     config = {**{
#         'architecture':'NPE2',
#         'epoch': 4,
#         'layers':1
#         }, **mod_args},
#     name = 'test'
# )

# def main_train():
#     wandb.init(group = 'sweep_test')
#     batch_size = wandb.config.batch_size
#     step_per_batch = wandb.config.step_per_batch
#     n_epochs = 5

#     model = NPE(1, base, emb_net, mod_args)

#     wandb.watch(model, log_freq = 1)
    
#     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer,
#         factor=0.5,
#         min_lr=1e-5,
#         patience=32,
#         threshold=1e-2,
#         threshold_mode='abs',
#     )

#     loss_plt = []
#     with tqdm(range(n_epochs), unit='epoch') as tq: #256 epoch
#         for epoch in tq:
#             subset_batch = torch.randint(len(simulator.data), (batch_size,))
#             losses = []

#             for xb,yb,tb in zip(simulator.data[subset_batch], simulator.obs[subset_batch], simulator.time[subset_batch]):
#                 subset_data = torch.randint(traj_len, (step_per_batch,))
#                 x,y,t = xb[subset_data], yb[subset_data], tb[subset_data]
#                 x = x[:,None,...,None]
#                 y = y[:,None,...,None]
#                 l = model.loss(x, y, t)

#                 optimizer.zero_grad()
#                 l.backward()
#                 optimizer.step()

#                 losses.append(l.item())


#             l = sum(losses) / len(losses)

#             wandb.log({'train_loss' : l})
#             loss_plt += [l]
#             # torch.save(model, f'mod_epoch_{epoch}.pt')

#             tq.set_postfix(loss=l, lr=optimizer.param_groups[0]['lr'])
#             scheduler.step(l)

# wandb.agent(sweep_id, function = main_train, count = 3)
# import matplotlib.pyplot as plt 
# plt.plot(loss_plt)
# plt.show()

# EVALUATE THE MODEL 

# simulator.generate_steps(torch.randn((1, N)), times)
# simulator.data = preprocess_x(simulator.data)[:,:20]
# simulator.obs = preprocess_y(simulator.obs)[:,:20]
# simulator.time = preprocess_t(simulator.time)[:,:20]
# simulator.display_sim(obs = True, filename='hovGT')

# base = Unconditional(
#             DiagNormal,
#             torch.zeros(N),
#             torch.ones(N),
#             buffer=True,
#         )

# epoch = 255

# x_dim = torch.tensor((1, 1, N, 1))
# y_dim = torch.tensor((1, 2, N, 1))
# model = NPE(x_dim, y_dim, base, 1, 3, torch.tensor((4, 1)), type = '1D')
# model = torch.load(f'mod_epoch_{epoch}.pt')
# model.eval()

# y_dim = torch.tensor((1, 1, 6, 1))
# emb_net = EmbedObs(y_dim, x_dim)
# emb_net = torch.load(f'emb_epoch_{epoch}.pt')
# emb_net.eval()

# EVALUATE CORNER PLOT
# x,y,t = simulator.data[0, -1], simulator.obs[0, -1], simulator.time[0, -1]

# x = x[None, None, :, None]
# y = y[None, None, :, None]
# t = t.unsqueeze(-1)

# x_s = model.sample(y, t, 2**12, max_samp = 2**7).squeeze().detach()


# # import lampe
# from lampe.plots import corner, mark_point

# fig = corner(x_s[:,::5], smooth=1, figsize=(6.8, 6.8), legend="p(x | y*)")

# x_star = x.squeeze()[::5]
# mark_point(fig, x_star)

# fig.savefig('cornerNPEtestSimf.pdf')

# y_s = simulator.observe(x_s)
# fig = corner(y_s, smooth=1, figsize=(6.8, 6.8), legend="p(y | y*)")

# y_star = y.squeeze()
# mark_point(fig, y_star)

# fig.savefig('cornerNPEtestObsf.pdf')
# fig.clear()


# EVALUATE TRAJECTORY

x,y,t = simulator.data[0], simulator.obs[0], simulator.time[0]

x = x[:, None, :, None]
y = y[:, None, :, None]
t = t.unsqueeze(1)

x_s = []

for yt,tt in tqdm(zip(y[:200], t[:200])):
    samp = model.sample(yt.unsqueeze(0), tt.unsqueeze(0), 128).squeeze().detach()
    samp = samp.mean((0))
    x_s.append(samp.unsqueeze(0))

x_s = torch.cat(x_s, dim = 0)
simulator.data = x_s[None,...]

simulator.obs = simulator.observe()
simulator.display_sim(obs=True, filename='hovSAMP')