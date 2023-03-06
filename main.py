from dasbi.simulators.sim_lorenz96 import LZ96 as sim 
from dasbi.simulators.observators.observator2D import ObservatorStation2D
import torch 

# GENERATE DATA AND OBSERVATIONS
torch.manual_seed(42)

n_sim = 2**10
batch_size = 16
N = 32 

simulator = sim(N = N, noise=.5)
observer = ObservatorStation2D((32,1), (4,1), (2,1), (4,1), (2,1))
simulator.init_observer(observer)

tmax = 5
traj_len = 256
times = torch.linspace(0, tmax, traj_len)

simulator.generate_steps(torch.randn((n_sim, N)), times)
print(simulator.data.shape, simulator.obs.shape, simulator.time.shape)
# simulator.save_h5_data('./data/LZ96/LZ96_32pts_n05')