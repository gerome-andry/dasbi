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

        # import matplotlib.pyplot as plt

        # stat_map = torch.zeros(self.dims)
        # stat_map[x_pos, y_pos] = 1
        # plt.imshow(stat_map)
        # plt.show()

        # FIX VISUALIZE METHOD TO WRAP THAT !!!

        self.station = x_pos, y_pos


    def observe(self, data):
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

        # #
        # import matplotlib.pyplot as plt

        

        # plt.imshow(gaussian_kernel)
        # plt.tight_layout()
        # plt.show()
        # plt.close()
        # #
        x_l = torch.clamp(self.station[0] - self.aoe[0], min=0)
        x_r = torch.clamp(self.station[0] + self.aoe[0] + 1, max=self.dims[0])

        y_d = torch.clamp(self.station[1] - self.aoe[1], min=0)
        y_u = torch.clamp(self.station[1] + self.aoe[1] + 1, max=self.dims[1])

        obs = torch.zeros(
            torch.Size(
                torch.cat(
                    (torch.tensor(data.shape[:-2]), torch.tensor(self.station[0].shape))
                )
            )
        )
        # #
        # importance = torch.zeros_like(data)
        # #
        for x, row in enumerate(x_l):
            for y, _ in enumerate(row):
                # #
                # importance[
                #     ..., x_l[x, y] : x_r[x, y], y_d[x, y] : y_u[x, y]
                # ] += gaussian_kernel[
                #     self.aoe[0]
                #     - (self.station[0][x, y] - x_l[x, y]) : self.aoe[0]
                #     + (x_r[x, y] - self.station[0][x, y]),
                #     self.aoe[1]
                #     - (self.station[1][x, y] - y_d[x, y]) : self.aoe[1]
                #     + (y_u[x, y] - self.station[1][x, y]),
                # ]
                # #
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

        # fig = plt.figure()
        # ax = plt.axes(projection="3d")
        # if len(importance.shape) == 3:
        #     plt.imshow(importance[0, :, :].T)
        # else:
        #     plt.imshow(importance[0, :, :, 0].T)
        # plt.colorbar()
        # plt.tight_layout()
        # plt.show()

        return obs

    def gaussian_2D(self, mean, std, pts):
        return (1 / (2 * torch.pi * torch.prod(std))) * torch.exp(
            -(
                ((pts[..., 0] - mean[0]) ** 2) / (2 * std[0] ** 2)
                + ((pts[..., 1] - mean[1]) ** 2) / (2 * std[1] ** 2)
            )
        )


if __name__ == "__main__":
    torch.manual_seed(42)

    o = ObservatorStation2D((512, 512), (16, 16), (8, 8), (30, 30), (10, 10))
    o.observe(torch.rand((1, 512, 512)))

    # o = ObservatorStation2D((5, 1), (2, 2), (0, 0), (5, 5), (2, 2))
    # o.observe(torch.rand((3, 5, 1)))
