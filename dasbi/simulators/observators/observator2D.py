import torch
import numpy as np
from tqdm import tqdm
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class ObservatorStation2D:
    def __init__(self, space_dim, kernel_size, stride, aoe, spatial_field):
        self.dims = space_dim
        self.k_sz = kernel_size
        self.s = stride
        self.aoe = aoe
        self.station = torch.zeros(space_dim)
        self.spf = spatial_field
        self.place_stations()

    def place_stations(self):
        xl = torch.arange(0, self.dims[0], self.k_sz[0] + self.s[0])

        yu = torch.arange(0, self.dims[1], self.k_sz[1] + self.s[1])

        x_pos = torch.randint(low=0, high=self.k_sz[0], size=(len(yu), len(xl))) + xl
        x_pos = torch.clamp(x_pos, min=0, max=self.dims[0] - 1).T

        y_pos = torch.randint(low=0, high=self.k_sz[1], size=(len(xl), len(yu))) + yu
        y_pos = torch.clamp(y_pos, min=0, max=self.dims[1] - 1)

        self.station = x_pos, y_pos

    def get_mask(self):
        stat_map = torch.zeros(self.dims)
        stat_map[self.station[0], self.station[1]] = 1

        return stat_map
    
    def visualize(self):
        import matplotlib.pyplot as plt

        # STATION
        stat_map = torch.zeros(self.dims)
        stat_map[self.station[0], self.station[1]] = 1
        plt.imshow(stat_map, aspect = 'auto')
        plt.show()
        plt.close()

        # KERNEL
        eval_grid = torch.meshgrid(
            torch.arange(-self.aoe[0], self.aoe[0] + 1),
            torch.arange(-self.aoe[1], self.aoe[1] + 1),
            indexing="xy",
        )
        eval_grid = torch.cat([X.unsqueeze(-1) for X in eval_grid], dim=-1)

        gaussian_kernel = self.gaussian_2D(
            torch.zeros(2), torch.tensor(self.spf), eval_grid
        )
        gaussian_kernel = (gaussian_kernel / torch.max(gaussian_kernel)).T
        plt.imshow(gaussian_kernel, aspect = 'auto')
        plt.colorbar()
        plt.tight_layout()
        plt.show()
        plt.close()
        plt.clf()

        importance = self.observe(None, True)
        plt.imshow(importance[:, :], aspect = 'auto')
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    def observe(self, data, get_imp = False, imp_idx = None):
        eval_grid = torch.meshgrid(
            torch.arange(-self.aoe[0], self.aoe[0] + 1),
            torch.arange(-self.aoe[1], self.aoe[1] + 1),
            indexing="xy",
        )
        eval_grid = torch.cat([X.unsqueeze(-1) for X in eval_grid], dim=-1)

        gaussian_kernel = self.gaussian_2D(
            torch.zeros(2), torch.tensor(self.spf), eval_grid
        )
        gaussian_kernel = (gaussian_kernel / torch.max(gaussian_kernel)).T

        x_l = torch.clamp(self.station[0] - self.aoe[0], min=0)
        x_r = torch.clamp(self.station[0] + self.aoe[0] + 1, max=self.dims[0])

        y_d = torch.clamp(self.station[1] - self.aoe[1], min=0)
        y_u = torch.clamp(self.station[1] + self.aoe[1] + 1, max=self.dims[1])


        if get_imp:
            importance = torch.zeros(self.dims)
        else:
            obs = torch.zeros(
            torch.Size(
                torch.cat(
                    (torch.tensor(data.shape[:-2]), torch.tensor(self.station[0].shape))
                    )
                )
            )
            print(obs.shape)

        
        for x, row in enumerate(x_l):
            for y, _ in enumerate(row):
                #
                if get_imp:
                    if (imp_idx is None) or (imp_idx is not None and x*len(row) + y == imp_idx): 
                        importance[
                            ..., x_l[x, y] : x_r[x, y], y_d[x, y] : y_u[x, y]
                        ] += gaussian_kernel[
                            self.aoe[0]
                            - (self.station[0][x, y] - x_l[x, y]) : self.aoe[0]
                            + (x_r[x, y] - self.station[0][x, y]),
                            self.aoe[1]
                            - (self.station[1][x, y] - y_d[x, y]) : self.aoe[1]
                            + (y_u[x, y] - self.station[1][x, y]),
                        ]
                #
                else:
                    obs[..., x, y] = torch.tensordot(
                        data[..., x_l[x, y] : x_r[x, y], y_d[x, y] : y_u[x, y]],
                        gaussian_kernel[
                            self.aoe[0]
                            - (self.station[0][x, y] - x_l[x, y]) : self.aoe[0]
                            + (x_r[x, y] - self.station[0][x, y]),
                            self.aoe[1]
                            - (self.station[1][x, y] - y_d[x, y]) : self.aoe[1]
                            + (y_u[x, y] - self.station[1][x, y]),
                        ],
                    )

        if get_imp:
            return importance
        else:
            return obs


    def gaussian_2D(self, mean, std, pts):
        return (1 / (2 * torch.pi * torch.prod(std))) * torch.exp(
            -(
                ((pts[..., 0] - mean[0]) ** 2) / (2 * std[0] ** 2)
                + ((pts[..., 1] - mean[1]) ** 2) / (2 * std[1] ** 2)
            )
        )
    
    def get_Obs_mat(self):
        idx = self.get_mask().flatten()
        N = len(idx)
        idx = idx.nonzero()
        A = torch.zeros((len(idx), N))

        for i in range(len(idx)):
            A[i,:] = self.observe(None, get_imp=True, imp_idx=i).flatten()
         
        return A

if __name__ == "__main__":
    torch.manual_seed(42)

    o = ObservatorStation2D((8, 1), (3, 3), (1, 1), (3, 3), (.75, .75))
    m = o.get_mask()
    print(m, m.shape)
    A = o.get_Obs_mat()
    # print(A)
    import matplotlib.pyplot as plt
    exact = A@A.transpose(-2,-1)
    plt.imshow(exact)
    plt.show()

    d = torch.diag((A**2).sum(1))
    plt.imshow(d)
    plt.show()

    plt.imshow(((exact - d)**2).sqrt())
    plt.show()
    print((((exact - d))**2).sqrt().sum())
    print(((exact)**2).sqrt().sum())
    o.visualize()
    # print(o.get_mask())
    # idx = o.get_mask().flatten().nonzero()
    # A = torch.zeros((4,64))
    # A[range(4), idx] = 1
    # print(A)
    # o.observe(torch.rand((1, 512, 512)))

    # o = ObservatorStation2D((5, 1), (2, 2), (0, 0), (5, 5), (2, 2))
    # o.observe(torch.rand((3, 5, 1)))
