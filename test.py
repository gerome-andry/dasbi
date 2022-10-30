from kolmogorov_methods import *
from dynamical_system import KolmogorovFlow

dyn_sys = KolmogorovFlow(64, 100, 25, 1e-2, 4)
a = generate_data_kolmogorov(random.PRNGKey(42), dyn_sys, 1, 10, 0)

#python run_generate_training_data.py --config config_files/data_generation/kolmogorov_generate_data.config