import nfmodules as nf
import torch
from zuko.distributions import DiagNormal
from zuko.flows import Unconditional

torch.manual_seed(42)


class CrossData:
    def __init__(self, space_dim, permute):
        self.sd = space_dim
        self.perm = permute

    def generateX(self, n):
        x = torch.randn((n, 1, self.sd, self.sd))
        for i in range(self.sd):
            x[..., i, i] = 3
            x[..., i, -(i + 1)] = 3

        return x

    def generateY(self, X):
        X = X.flatten(1)
        return X[:, self.perm].unflatten(1, (1, self.sd, self.sd))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    sd = 8
    perm = torch.randperm(sd**2)
    cd = CrossData(sd, perm)
    ns = 2**8
    bs = 16
    x = cd.generateX(ns)
    y = cd.generateY(x)

    data = [(x[i : (i + 1) * bs], y[i : (i + 1) * bs]) for i in range(ns // bs)]

    base = Unconditional(
        DiagNormal,
        torch.zeros(sd**2),
        torch.ones(sd**2),
        buffer=True,
    )

    model = nf.ConvNPE(
        torch.tensor((1, 1, sd, sd)),
        torch.tensor((1, 1, sd, sd)),
        base,
        2,
        1,
        torch.tensor((3, 3)),
    ).cuda()

    print(count_parameters(model))
    import torch.optim as optim
    from lampe.utils import GDStep

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 64)
    step = GDStep(optimizer, clip=1.0)  # gradient descent step with gradient clipping

    from tqdm import tqdm
    import matplotlib.pyplot as plt

    with tqdm(range(64), unit="epoch") as tq:
        for epoch in tq:
            model.train()
            losses = torch.stack(
                [step(model.loss(im.cuda(), obs.cuda())) for im, obs in data]
            )

            tq.set_postfix(loss=losses.mean().item())

            scheduler.step()

            model.eval()
            torch.save(model, f"toy_{epoch}.pt")
            samp = model.sample(y[-1, ...].unsqueeze(0), 1)
            plt.clf()
            plt.imshow(samp.squeeze().detach())
            plt.show(block=False)
            plt.pause(1)
