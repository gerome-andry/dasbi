import torch
import torch.nn as nn
from .transforms import *
from zuko.distributions import DiagNormal 

class ConvNPE(Transform):
    def __init__(self, x_dim, y_dim, base, n_modules, n_c, k_sz, type = '2D', emb_net = None):
        # Deal with Embedding network !!!
        super().__init__()
        self.x_dim = x_dim.clone()
        self.n_mod = n_modules
        self.transforms = nn.ModuleList([])
        self.ssplit = SpatialSplit()
        self.base_dist = base
        self.type = type

        xd = x_dim.clone()
        yd = y_dim.clone()
        for _ in range(n_modules):
            self.transforms.append(ConvStep(xd, yd, n_c, k_sz))
            xd[1] *= 3
            if type == '2D':
                xd[-2:] //= 2
                yd[-2:] //= 2
            else:
                xd[-2:] //= 4
                yd[-2:] //= 4

            xd[-2:] = torch.clamp(x_dim[-2:], 1)
            yd[-2:] = torch.clamp(y_dim[-2:], 1)

        self.conv_y = nn.ModuleList([nn.Conv2d(4*y_dim[1], y_dim[1], 1) for _ in range(self.n_mod - 1)])

    def forward(self, x, y):
        # EMBED !!!
        z = []
        ladj = x.new_zeros(x.shape[0])
        init_shape = x.shape
        for i, t in enumerate(self.transforms):
            z_i, ladj_i = t(x, y)
            c = z_i.shape[1]
            z_i, x = z_i.split((c*1//4, c*3//4), dim = 1)
            z.append(z_i)
            ladj += ladj_i
            if i < self.n_mod - 1:
                y, _ = self.ssplit(y)
                y = self.conv_y[i](y)
        
        z.append(x)
        z = [x.flatten(1) for x in z]
        z = torch.cat(z, dim = 1).reshape(init_shape)

        return z, ladj

    def inverse(self, z, y):
        emb_context = [y]
        for c_y in self.conv_y:
            y, _ = self.ssplit(y)
            emb_context.append(c_y(y))
        emb_context.reverse()

        b,c,h,w = z.shape
        shapes = []
        for i in range(self.n_mod):
            if self.type == '2D':
                new_h = torch.clamp(torch.tensor([h//2]), 1)
                new_w = torch.clamp(torch.tensor([w//2]), 1)
            else:
                new_h = torch.clamp(torch.tensor([h//4]), 1)
                new_w = torch.clamp(torch.tensor([w//4]), 1)

            shapes.append((b,c,new_h, new_w))
            c *= 3
            h = new_h
            w = new_w

        shapes.append((b,c,h,w))
        x = z.reshape(b,-1)
        x_f_ls = x.split([torch.prod(torch.tensor(S[1:])) for S in shapes], dim = 1)
        x_i = []
        for i, x_f in enumerate(x_f_ls):
            x_i.append(x_f.unflatten(1,shapes[i][1:]))
        
        x_i[-2] = torch.cat((x_i[-2], x_i[-1]), dim = 1)
        x_i = x_i[:-1]
        x_i.reverse()

        x_trans = x_i[0]
        for i, t in enumerate(reversed(self.transforms)):
            x_trans, _ = t.inverse(x_trans, emb_context[i])
            if i < len(self.transforms) - 1:
                x_trans = torch.cat((x_i[i+1], x_trans), dim = 1)

        x = x_trans
        ladj = None

        return x, ladj

    def sample(self, y, n):
        #try flow(y) to create a sampler
        assert y.shape[0] == 1, "Can only condition on a single observation for sampling"
        y = y.expand(n, -1, -1, -1)
        z = self.base_dist().sample((n,))
        s_dim = self.x_dim
        s_dim[0] = n
        z = z.reshape(tuple(s_dim))

        return self.inverse(z, y)[0]

    def loss(self, x, y):
        z, ladj = self.forward(x,y)
        z = z.reshape((z.shape[0], -1)) # B x elem
        return -(ladj + self.base_dist().log_prob(z)).mean()


class ConvCoup(nn.Module):
    def __init__(self, input_chan, output_chan, lay=3, chan=32, ks=3):
        super().__init__()

        assert output_chan % 2 == 0, "Need pair output channel"
        self.oc = output_chan
        self.head = nn.Conv2d(input_chan, chan, ks, padding=(ks - 1) // 2)
        self.conv = nn.ModuleList(
            [nn.Conv2d(chan + input_chan, chan, 1) for _ in range(lay)]
        )
        self.tail = nn.Conv2d(chan + input_chan, output_chan, ks, padding=(ks - 1) // 2)
        self.act = nn.ReLU()

    def forward(self, x):
        x_skip = x
        x = self.head(x)
        for c in self.conv:
            x = self.act(x)
            x = c(torch.cat((x, x_skip), dim=1))

        x = self.act(x)
        x = self.tail(torch.cat((x, x_skip), dim=1))

        return x[:, : self.oc // 2, ...], x[:, self.oc // 2 :, ...]


class ConvEmb(nn.Module):
    def __init__(self, input_dim, output_lg):
        super().__init__()
        ks = torch.clamp(input_dim[-2:] // 3, 1)
        self.conv1 = nn.Conv2d(input_dim[1], input_dim[1] * 4, ks)
        self.mpool_in = nn.MaxPool2d(tuple(ks), stride=1)
        self.conv2 = nn.Conv2d(input_dim[1] * 5, 1, (1, 1))
        self.act = nn.ReLU()

        self.lin = nn.Linear(torch.prod(input_dim[-2:] - ks + 1), output_lg)

    def forward(self, x, y):
        emb_y = self.conv1(y)
        emb_y = self.act(emb_y)
        emb_y = torch.cat((self.mpool_in(y), emb_y), dim=1)
        emb_y = self.conv2(emb_y)
        emb_y = self.act(emb_y).flatten()
        out = self.lin(emb_y) + x

        return self.act(out)


class ConvStep(Transform):
    def __init__(self, input_dim, context_dim, n_conv, kernel_sz):
        super().__init__()
        self.mod = nn.ModuleList([SpatialSplit(), ActNorm()])
        self.nc = n_conv
        self.conv_mod = nn.ModuleList()
        mode = ["UL", "LR", "UR", "LL"]

        for _ in range(n_conv):
            k_net = ConvEmb(context_dim, torch.prod(kernel_sz) - 1)
            self.conv_mod.append(
                nn.ModuleList(
                    [
                        InvConv(kernel_sz, k_net, mode=mode[0]),
                        InvConv(kernel_sz, k_net, mode=mode[1]),
                        InvConv(kernel_sz, k_net, mode=mode[2]),
                        InvConv(kernel_sz, k_net, mode=mode[3]),
                        ActNorm(),
                    ]
                )
            )
            mode.reverse()

        chan_c = input_dim[1]
        self.coup = QuadCoupling(
            [ConvCoup(chan_c * i + 4 * context_dim[1], 2 * chan_c) for i in range(1, 4)]
        )

    def forward(self, x, context):
        assert (
            x.shape[-2:] == context.shape[-2:]
        ), "Need same spatial dimensions for x and y"

        b = x.shape[0]
        ladj = x.new_zeros(b)
        scaled_context, _ = self.mod[0](context)
        z = x
        for m in self.mod:
            z, ladj_i = m(z)
            # print(z)
            ladj += ladj_i

        c = z.shape[1]
        for c_ls in self.conv_mod:
            for i, mc in enumerate(c_ls[:-1]):
                for j in range(c // 4):
                    z_c, ladj_i = mc(z[:, i * c // 4 + j, ...].unsqueeze(1), context)
                    z[:, i * c // 4 + j, ...] = z_c.squeeze(1)
                    ladj += ladj_i

            z, ladj_i = c_ls[-1](z)
            ladj += ladj_i

        z, ladj_i = self.coup(z, scaled_context)
        ladj += ladj_i

        return z, ladj

    def inverse(self, z, context):
        x = z
        scaled_context, _ = self.mod[0](context)
        x, _ = self.coup.inverse(x, scaled_context)

        c = x.shape[1]
        for c_ls in reversed(self.conv_mod):
            x, _ = c_ls[-1].inverse(x)
            for i, mc in enumerate(
                c_ls[:-1]
            ):  # not reverse to keep the same padding order
                for j in range(c // 4):
                    x_c, _ = mc.inverse(x[:, i * c // 4 + j, ...].unsqueeze(1), context)
                    x[:, i * c // 4 + j, ...] = x_c.squeeze(1)

        for m in reversed(self.mod):
            x, _ = m.inverse(x)

        ladj = None

        return x, ladj


if __name__ == "__main__":
    import os 
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    x_dim = (20, 1, 1, 16)
    y_dim = (20, 1, 1, 16)
    elm = x_dim[-2]*x_dim[-1]
    cs = ConvNPE(torch.tensor(x_dim), torch.tensor(y_dim), DiagNormal(torch.zeros(elm), torch.ones(elm)), 2, 1, torch.tensor((3,1)), type = '1D')
    x = torch.randn(x_dim)
    y = torch.randn(y_dim)
    # print(x[0,0])
    z, l = cs(x, y)
    # print(z)
    print(l)
    x_b, _ = cs.inverse(z, y)
    # print(x_b[0,0])
    print(torch.allclose(x, x_b, atol=1e-3, rtol=0))
    print(cs.loss(x,y).mean())
    # print(cs)
