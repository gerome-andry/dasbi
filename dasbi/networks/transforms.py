import torch
import torch.nn as nn


class Transform(nn.Module):
    def forward(self, x, context=None):
        raise NotImplementedError()

    def inverse(self, z, context=None):
        raise NotImplementedError()


class ActNorm(Transform):
    # Take a look at conditioning ?
    def __init__(self):
        super().__init__()
        self.mu = None
        self.log_sig = None

    def forward(self, x, context=None):
        if self.mu is None:
            self.mu = nn.Parameter(x.mean(dim=0))
            self.log_sig = torch.nan_to_num(x.std(dim=0))
            self.log_sig[self.log_sig == 0] = 1
            self.log_sig = nn.Parameter(self.log_sig.log())

        z = (x - self.mu) / self.log_sig.exp()
        batch_size = x.shape[0]
        ladj = (-self.log_sig).sum() * z.new_ones(batch_size)

        return z, ladj

    def inverse(self, z, context=None):
        assert self.mu is not None, "Parameters not initialized (need forward pass)"

        x = z * (self.log_sig.exp()) + self.mu
        ladj = None

        return x, ladj


class InvConv(Transform):
    # fix multiple channels ???

    def __init__(self, kernel_sz, kern_net, mode="UL"):
        params_k = torch.prod(kernel_sz)
        assert params_k > 1, "Too small kernel, must contain more than 1 element"
        super().__init__()

        self.mode = mode
        self.net = kern_net
        self.ks = kernel_sz
        self.kernel = nn.Parameter(torch.randn(params_k - 1))

        self.mask = torch.ones(tuple(self.ks), dtype=torch.bool)
        hpad, wpad = self.ks - 1
        if self.mode == "UL":
            self.pad = lambda x: nn.functional.pad(x, (wpad, 0, hpad, 0))
            self.unpad = lambda x: x[..., hpad:, wpad:]
            self.mask[-1, -1] = 0
        elif self.mode == "UR":
            self.pad = lambda x: nn.functional.pad(x, (0, wpad, hpad, 0))
            self.unpad = (
                lambda x: x[..., hpad:, :-wpad] if wpad > 0 else x[..., hpad:, :]
            )
            self.mask[-1, 0] = 0
        elif self.mode == "LL":
            self.pad = lambda x: nn.functional.pad(x, (wpad, 0, 0, hpad))
            self.unpad = (
                lambda x: x[..., :-hpad, wpad:] if hpad > 0 else x[..., :, wpad:]
            )
            self.mask[0, -1] = 0
        else:  # LR
            self.pad = lambda x: nn.functional.pad(x, (0, wpad, 0, hpad))
            self.unpad = (
                lambda x: x[..., :-hpad, :-wpad]
                if hpad > 0 and wpad > 0
                else x[..., :-hpad, :]
                if hpad > 0 and wpad == 0
                else x[..., :, :-wpad]
                if hpad == 0 and wpad > 0
                else x
            )
            self.mask[0, 0] = 0

    def forward(self, x, context):
        batch_size = x.shape[0]
        z = torch.zeros_like(x)

        for b in range(batch_size):
            weights = self.conv_kern(self.net(self.kernel, context[b].unsqueeze(0)))
            weights = weights.reshape((1, 1) + tuple(self.ks))
            x_p = self.pad(x[b].unsqueeze(0))
            z[b] = nn.functional.conv2d(x_p, weights)

        ladj = z.new_zeros(batch_size)

        return z, ladj

    def inverse(self, z, context):
        batch_size, c, h, w = z.shape
        x = torch.zeros_like(z)

        for b in range(batch_size):
            weights = self.conv_kern(self.net(self.kernel, context[b].unsqueeze(0)))
            z_b = z[b].unsqueeze(0)
            c_mat = self.fc_from_conv(weights, z_b)
            z_b = z_b.permute((2, 3, 1, 0))
            z_b = z_b.reshape((c * h * w, 1))
            x[b] = (
                torch.linalg.solve(c_mat, z_b).reshape(h, w, c, 1).permute((3, 2, 0, 1))
            )

        ladj = None

        return x, ladj

    def conv_kern(self, w):
        ck = torch.ones(tuple(self.ks))
        ck[self.mask] = w

        return ck

    def fc_from_conv(self, kern, x):
        xdim = x.shape[-2:]
        K = torch.ones(xdim)
        K = self.pad(K)
        c_mat = torch.zeros((x[0, ...].numel(),) * 2)

        row = 0
        for i in range(K.shape[0] - kern.shape[0] + 1):
            for j in range(K.shape[1] - kern.shape[1] + 1):
                K_check = torch.zeros_like(K)

                K_check[i : i + kern.shape[0], j : j + kern.shape[1]] = (
                    kern * K[i : i + kern.shape[0], j : j + kern.shape[1]]
                )

                K_check = self.unpad(K_check)
                c_mat[row, :] = K_check.flatten()
                row += 1

        return c_mat


class SpatialSplit(Transform):
    def __init__(self):
        super().__init__()
        self.type = "2D"

    def forward(self, x, context=None):
        b, c, h, w = x.shape
        if h > 1 and w > 1:  # 2D data
            o_shape = (b, c, h // 2, w // 2)

            z1 = x[..., ::2, ::2].reshape(o_shape)
            z2 = x[..., ::2, 1::2].reshape(o_shape)
            z3 = x[..., 1::2, ::2].reshape(o_shape)
            z4 = x[..., 1::2, 1::2].reshape(o_shape)
        else:
            if h == 1:
                o_shape = (b, c, h, w // 4)
                z1 = x[..., ::4].reshape(o_shape)
                z2 = x[..., 1::4].reshape(o_shape)
                z3 = x[..., 2::4].reshape(o_shape)
                z4 = x[..., 3::4].reshape(o_shape)
                self.type = "1DH"
            else:
                o_shape = (b, c, h // 4, w)
                z1 = x[..., ::4, :].reshape(o_shape)
                z2 = x[..., 1::4, :].reshape(o_shape)
                z3 = x[..., 2::4, :].reshape(o_shape)
                z4 = x[..., 3::4, :].reshape(o_shape)
                self.type = "1DW"

        z = torch.cat((z1, z2, z3, z4), dim=1)
        ladj = x.new_zeros(b)

        return z, ladj

    def inverse(self, z, context=None):
        b, c, h, w = z.shape

        new_c = c // 4
        if self.type == "2D":  # 2D data
            x = torch.zeros((b, c // 4, 2 * h, 2 * w))
            x[..., ::2, ::2] = z[:, :new_c, ...]
            x[..., ::2, 1::2] = z[:, new_c : 2 * new_c, ...]
            x[..., 1::2, ::2] = z[:, 2 * new_c : 3 * new_c, ...]
            x[..., 1::2, 1::2] = z[:, 3 * new_c : c, ...]
        else:
            if self.type == "1DH":
                x = torch.zeros((b, c // 4, h, 4 * w))
                x[..., ::4] = z[:, :new_c, ...]
                x[..., 1::4] = z[:, new_c : 2 * new_c, ...]
                x[..., 2::4] = z[:, 2 * new_c : 3 * new_c, ...]
                x[..., 3::4] = z[:, 3 * new_c : c, ...]
            else:
                x = torch.zeros((b, c // 4, 4 * h, w))
                x[..., ::4, :] = z[:, :new_c, ...]
                x[..., 1::4, :] = z[:, new_c : 2 * new_c, ...]
                x[..., 2::4, :] = z[:, 2 * new_c : 3 * new_c, ...]
                x[..., 3::4, :] = z[:, 3 * new_c : c, ...]

        ladj = None

        return x, ladj


class AffineCoupling(Transform):
    def __init__(self, net):
        super().__init__()
        self.st_net = net

    def forward(self, x, context):
        # context contains x prev and y
        log_s, t = self.st_net(context)
        z = (x + t) * log_s.exp()
        ladj = log_s.sum((1, 2, 3))
        return z, ladj

    def inverse(self, z, context):
        log_s, t = self.st_net(context)
        x = z / (log_s.exp()) - t
        ladj = None

        return x, ladj


class QuadCoupling(Transform):
    def __init__(self, nets):
        super().__init__()
        self.coupling_nets = [AffineCoupling(nn) for nn in nets]

    def forward(self, x, context):
        b, c, h, w = x.shape
        assert c % 4 == 0, "Must contain 4n channels"
        chan_coup = c // 4

        z = x[:, :chan_coup, ...]
        ladj = x.new_zeros(x.shape[0])
        it = 1
        for ac in self.coupling_nets:
            new_x = x[:, it * chan_coup : (it + 1) * chan_coup, ...]
            emb_x = torch.cat((x[:, : it * chan_coup, ...], context), 1)
            z_i, ladj_i = ac(new_x, emb_x)
            z = torch.cat((z, z_i), 1)
            ladj += ladj_i
            it += 1

        return z, ladj

    def inverse(self, z, context):
        b, c, h, w = z.shape
        assert c % 4 == 0, "Must contain 4n channels"
        chan_coup = c // 4

        x = z[:, :chan_coup, ...]
        it = 1
        for ac in self.coupling_nets:
            new_z = z[:, it * chan_coup : (it + 1) * chan_coup, ...]
            emb_z = torch.cat((x, context), 1)
            x_i, _ = ac.inverse(new_z, emb_z)
            x = torch.cat((x, x_i), 1)
            it += 1

        ladj = None

        return x, ladj


class IdentityLayerx2(nn.Module):
    def __init__(self, o_lg):
        super().__init__()
        self.o_lg = o_lg

    def forward(self, x, y):
        return y.sum(1).flatten()[: self.o_lg]  # , y.sum(1).flatten()[:self.o_lg] ** 2


if __name__ == "__main__":
    # ac = QuadCoupling([IdentityLayerx2(),]*3, 0)
    # x = torch.randint(10,(20,4,2,2))
    # print(x)
    # y = 2*torch.ones_like(x)
    # z, l = ac(x,y)
    # print(l)
    # x, _ = ac.inverse(z,y)
    # print(x)

    # an = ActNorm()
    # x = 2 + torch.randn((10,5,5))
    # print(x)
    # z,l = an.forward(x)
    # print(z.mean(dim = 0), z.std(dim = 0))
    # print(an.inverse(z))

    ic = InvConv((3, 3), IdentityLayerx2(8), mode="LR")
    x = torch.randint(10, (20, 1, 3, 3))
    print(x)
    y = torch.ones((1, 1, 3, 3))
    z, _ = ic.forward(x.float(), y)
    print(z)
    x, _ = ic.inverse(z, y)
    print(x)

    # ss = SpatialSplit()
    # x = torch.randint(10,(1,3,2,2))
    # print(x)
    # z,l = ss(x)
    # print(l.shape)
    # print(z)
    # x, _ = ss.inverse(z)
    # print(x)
    pass
