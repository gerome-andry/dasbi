import torch
import torch.nn as nn


class EmbedObs(nn.Module):
    def __init__(self, in_shape, out_shape, conv_lay=2):
        super().__init__()
        self.x_shape = tuple(out_shape)
        self.y_shape = tuple(in_shape)
        h, w = out_shape[-2:]
        self.register_buffer(
            "freq",
            torch.cat([torch.arange(i, i + h // 2) for i in range(1, w + 1)])
            * torch.pi,
        )

        # assumption obs have smaller size than x
        self.extract = nn.ModuleList(
            [
                nn.Conv2d(self.y_shape[1] if i == 0 else 4 * i, 4 * (i + 1), 1)
                for i in range(conv_lay)
            ]
        )
        self.upsample = nn.ConvTranspose2d(
            4 * conv_lay,
            32,
            (
                self.x_shape[-2] - self.y_shape[-2] + 1,
                self.x_shape[-1] - self.y_shape[-1] + 1,
            ),
        )
        self.head = nn.Conv2d(32, self.x_shape[1] - 1, 1)

    def time_embed(self, t):
        # extend to multiple times
        # time between 0 and 1 ?
        t = self.freq * t[..., None]
        t = t.transpose(0, 1)
        t = torch.cat(
            [torch.stack([tc, ts], dim=1) for tc, ts in zip(t.cos(), t.sin())], dim=-1
        )
        # import matplotlib.pyplot as plt
        # plt.imshow(t)
        # plt.show()
        # plt.clf()
        t = t.reshape((-1, 1) + self.x_shape[2:])

        return t

    def forward(self, y, t):
        t_emb = self.time_embed(t)
        y_emb = y

        for e in self.extract:
            y_emb = e(y_emb)

        y_emb = self.head(self.upsample(y_emb))

        return torch.cat((y_emb, t_emb), dim=1)


if __name__ == "__main__":
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    eo = EmbedObs((1, 1, 3, 3), (1, 1, 16, 16))
    eo.time_embed(torch.linspace(0, 1, 16**2))
