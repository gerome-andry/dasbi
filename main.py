import dasbi.simulators
from dasbi.simulators.sim_lorenz96 import *
import zuko

import torch
import lampe
import matplotlib.pyplot as plt 

import torch.nn as nn
import torch.optim as optim
import zuko

from itertools import islice
from tqdm import tqdm


from lampe.data import JointLoader
from lampe.inference import NPE, NPELoss
from lampe.plots import nice_rc, corner, mark_point
from lampe.utils import GDStep

torch.manual_seed(42)

#========== test syst ==========

syst = LZ96(N = 10, noise=0.1)

# LOWER = -syst.F*torch.ones(syst.N)
# UPPER = syst.F*torch.ones(syst.N)

# prior = zuko.distributions.BoxUniform(LOWER, UPPER)

# X0 = prior.sample((2**20,))

# n_steps = 100
# t_eval = torch.linspace(0, 5, n_steps)

# o = ObservatorStation2D((10, 1), (2, 1), (1, 0), (2, 0), (.75, 1))

# syst.init_observer(o)
# syst.generate_steps(X0, t_eval, observe=True)
syst.load_raw('LZ96/10pt.npy')
print(syst.data.shape)
def show_hist(data):
    s = data.shape
    for i in range(s[1]):
        plt.clf()
        plt.hist(data[:2**18, i, 5].unsqueeze(0), density = True, histtype = 'step', bins = 100)
        plt.title(i*5/100)
        plt.show(block = False)
        plt.pause(0.01)

show_hist(syst.data)
# syst.observe()
# syst.display_sim(filename = 'LZ96/test_data', obs=True, delay=1)
# syst.save_h5_data(filename = 'LZ96/LZ96_10pt_Noise01')

#========== test train ==========

# dataset_train = lampe.data.H5Dataset(
#     "LZ96/LZ96_10pt_Noise01_train.h5", batch_size=256, shuffle=True
# )
# dataset_test = lampe.data.H5Dataset("LZ96/LZ96_10pt_Noise01_test.h5", batch_size=1, shuffle=True)

# x_dim = syst.N
# y_dim = 4

# estimator = lampe.inference.NPE(x_dim, y_dim, transforms=3, hidden_features=[64] * 10, build = zuko.flows.NSF)

# loss = lampe.inference.NPELoss(estimator)

# optimizer = optim.AdamW(estimator.parameters(), lr=1e-3)
# step = GDStep(optimizer, clip=1.0)  # gradient descent step with gradient clipping

# estimator.train()
# lo = []
# ne = 256
# with tqdm(range(ne), unit="epoch") as tq:
#     for epoch in tq:
#         losses = torch.stack(
#             [
#                 step(loss(x, y))
#                 for x, y in islice(dataset_train, 256)  # 256 batches per epoch
#             ]
#         )

#         lo.append(-losses.mean().item())
#         tq.set_postfix(loss=-lo[-1])

# plt.plot(range(1,ne + 1), lo)
# plt.xlabel('epochs')
# plt.ylabel('E(log)')
# plt.grid()
# plt.savefig('lossNoise01.pdf')
# plt.close()

# for x, y in islice(dataset_test, 1):
#     # TRY TO SAMPLE MORE AND AVERAGE TO LOOK AT
#     x_star, y_star = x, y

# print(x_star, y_star)
# estimator.eval()

# with torch.no_grad():
#     samples = estimator.sample(y_star, (2**16,))
# plt.rcParams.update(nice_rc(latex=False))  # nicer plot settings

# LABELS = [f'X{i}' for i in range(1, x_dim + 1)]

# LOWER = -(syst.F+2)*torch.ones(x_dim)
# UPPER = -LOWER

# samples = samples.reshape(-1, x_dim)
# print(samples.shape)

# fig = lampe.plots.corner(
#     samples,
#     smooth=2,
#     labels=LABELS,
#     legend="p(x | y*)",
#     figsize=(6.8, 6.8),
# )

# x_star = x_star.reshape(-1, 1)

# lampe.plots.mark_point(fig, x_star)

# fig.savefig('cornerNoise01.pdf')

# torch.save(estimator, 'NSF_10pts_Noise01.pt')