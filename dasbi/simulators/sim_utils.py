import torch
import numpy as np
import lampe
from tqdm import tqdm


class Simulator:
    def __init__(self):
        self.data = None
        self.time = None
        self.obs = None
        self.observable = False
        self.observer = None

    def init_observer(self, observer):
        self.observer = observer
        self.observable = True

    def save_raw(self, fname):
        with open(fname, "ab") as torch_file:
            torch.save(self.data, torch_file)
            # time ?

    def load_raw(self, fname):
        self.data = torch.load(fname)
        # time ?

    def save_h5_data(self, filename, split=True, set_splits=[0.8, 0.1, 0.1]):
        """
        Data always of the form (B x T x ..(space)..)
        """
        assert np.sum(set_splits) == 1
        sets = [[], [], []]

        nb_sim = self.data.shape[0]
        sizes = [
            int(np.round(set_splits[0] * nb_sim)),
            int(np.round(set_splits[1] * nb_sim)),
        ]
        sizes.append(nb_sim - np.sum(sizes))

        # tqdm and explain
        for i in tqdm(range(self.data.shape[1])):
            sz = sizes[0]
            sets[0].append((self.data[:sz, i, ...], self.obs[:sz, i, ...]))

            sz += sizes[1]
            sets[1].append(
                (self.data[sizes[0] : sz, i, ...], self.obs[sizes[0] : sz, i, ...])
            )
            sets[2].append((self.data[sz:, i, ...], self.obs[sz:, i, ...]))

        if split:
            lampe.data.H5Dataset.store(sets[0], filename + "_train.h5", size=sizes[0])
            lampe.data.H5Dataset.store(sets[1], filename + "_val.h5", size=sizes[1])
            lampe.data.H5Dataset.store(sets[2], filename + "_test.h5", size=sizes[2])
        else:
            lampe.data.H5Dataset.store(
                np.concatenate(sets), filename + ".h5", size=nb_sim
            )

    def generate_steps(self, x0, t_vect, observe=True):
        if observe:
            assert self.observable == True

    def odefun(t, init_state):
        pass

    def display_sim(self):
        pass

    def observe(self, data, tspan=1):
        pass

    def __str__(self):
        str = ""
        for k, v in self.__dict__.items():
            ls = ["\n", k, "=>", v.__str__()]
            str += " ".join(ls)

        return str + "\n----------\n"
