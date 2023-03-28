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
        print("AN", z.isnan().sum())

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
        self.kernel = nn.Parameter(torch.rand(params_k - kernel_sz[0]))

        self.mask = torch.ones(tuple(self.ks), dtype=torch.bool)
        hpad, wpad = self.ks[1:] - 1
        if self.mode == "UL":
            self.pad = (wpad, 0, hpad, 0)
            self.mask[:,-1, -1] = 0
        elif self.mode == "UR":
            self.pad = (0, wpad, hpad, 0)
            self.mask[:,-1, 0] = 0
        elif self.mode == "LL":
            self.pad = (wpad, 0, 0, hpad)
            self.mask[:,0, -1] = 0
        else:  # LR
            self.pad = (0, wpad, 0, hpad)
            self.mask[:,0, 0] = 0

    def triang_pad(self, x):
        return nn.functional.pad(x, self.pad)

    def triang_unpad(self, x):
        if self.mode == "UL":
            tu = lambda x: x[..., self.pad[2] :, self.pad[0] :]
        elif self.mode == "UR":
            tu = (
                lambda x: x[..., self.pad[2] :, : -self.pad[1]]
                if self.pad[1] > 0
                else x[..., self.pad[2] :, :]
            )
        elif self.mode == "LL":
            tu = (
                lambda x: x[..., : -self.pad[3], self.pad[0] :]
                if self.pad[3] > 0
                else x[..., :, self.pad[0] :]
            )
        else:
            tu = (
                lambda x: x[..., : -self.pad[3], : -self.pad[1]]
                if self.pad[3] > 0 and self.pad[1] > 0
                else x[..., : -self.pad[3], :]
                if self.pad[3] > 0 and self.pad[1] == 0
                else x[..., :, : -self.pad[1]]
                if self.pad[3] == 0 and self.pad[1] > 0
                else x
            )

        return tu(x)

    def forward(self, x, context):
        b,c,h,w = x.shape
        z = torch.zeros_like(x)

        weights = self.conv_kern(self.net(self.kernel, context))
        weights = torch.nan_to_num(weights)
        x_p = self.triang_pad(x)
        _,_,hp,wp = x_p.shape
        z = nn.functional.conv2d(x_p.view(1,b*c,hp,wp), weights.view((b*c,1,) + weights.shape[-2:]), groups = b*c)
        z = z.reshape((b,c,h,w))

        ladj = z.new_zeros(b)
        print("IC", z.isnan().sum())

        return z, ladj

    def inverse(self, z, context):
        b, c, h, w = z.shape
        x = torch.zeros_like(z)

        weights = self.conv_kern(self.net(self.kernel, context))
        weights = torch.nan_to_num(weights)
        c_mat = self.fc_from_conv(weights.view(b*c,weights.shape[-2], weights.shape[-1]), z.view(b*c,h,w))
        x = z#.permute((0, 2, 3, 1))
        x = x.reshape((b* c , h * w))
        x = torch.linalg.solve(c_mat, x).reshape(b, c, h, w)#.permute((0, 3, 1, 2))

        ladj = None

        return x, ladj

    def conv_kern(self, w):
        ck = w.new_ones((w.shape[0],)+tuple(self.ks))
        ck[:,self.mask] = w

        return ck

    def fc_from_conv(self, kern, x):
        xdim = x.shape
        K = x.new_ones(xdim)
        K = self.triang_pad(K)
        c_mat = x.new_zeros((x.shape[0],) + (x[0, ...].numel(),) * 2)

        row = 0
        for i in range(K.shape[1] - kern.shape[1] + 1):
            for j in range(K.shape[2] - kern.shape[2] + 1):
                K_check = torch.zeros_like(K)

                K_check[...,i : i + kern.shape[1], j : j + kern.shape[2]] = (
                    kern * K[...,i : i + kern.shape[1], j : j + kern.shape[2]]
                )

                K_check = self.triang_unpad(K_check)
                c_mat[...,row, :] = K_check.flatten(start_dim =1)
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
        print("SS", z.isnan().sum())

        return z, ladj

    def inverse(self, z, context=None):
        b, c, h, w = z.shape

        new_c = c // 4
        if self.type == "2D":  # 2D data
            x = z.new_zeros((b, c // 4, 2 * h, 2 * w))
            x[..., ::2, ::2] = z[:, :new_c, ...]
            x[..., ::2, 1::2] = z[:, new_c : 2 * new_c, ...]
            x[..., 1::2, ::2] = z[:, 2 * new_c : 3 * new_c, ...]
            x[..., 1::2, 1::2] = z[:, 3 * new_c : c, ...]
        else:
            if self.type == "1DH":
                x = z.new_zeros((b, c // 4, h, 4 * w))
                x[..., ::4] = z[:, :new_c, ...]
                x[..., 1::4] = z[:, new_c : 2 * new_c, ...]
                x[..., 2::4] = z[:, 2 * new_c : 3 * new_c, ...]
                x[..., 3::4] = z[:, 3 * new_c : c, ...]
            else:
                x = z.new_zeros((b, c // 4, 4 * h, w))
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
        log_s = torch.nan_to_num(log_s)
        t = torch.nan_to_num(t)
        z = (x + t) * log_s.exp()
        ladj = log_s.sum((1, 2, 3))
        print("AC", z.isnan().sum())

        return z, ladj

    def inverse(self, z, context):
        log_s, t = self.st_net(context)
        log_s = torch.nan_to_num(log_s)
        t = torch.nan_to_num(t)
        x = z / (log_s.exp()) - t
        ladj = None

        return x, ladj


class QuadCoupling(Transform):
    def __init__(self, nets):
        super().__init__()
        self.coupling_nets = nn.ModuleList([AffineCoupling(nn) for nn in nets])

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

        print("QC", z.isnan().sum())

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

    # ss = SpatialSplit()
    # x = torch.randint(10,(1,3,2,2))
    # print(x)
    # z,l = ss(x)
    # print(l.shape)
    # print(z)
    # x, _ = ss.inverse(z)
    # print(x)
    class ConvEmb(nn.Module):
        def __init__(self, input_dim, output_lg):
            super().__init__()
            ks = torch.clamp(input_dim[-2:] // 3, 1)
            strides = torch.clamp((input_dim[-2:] - ks)//7, 1)
            self.conv1 = nn.Conv2d(input_dim[1], input_dim[1] * 4, tuple(ks), stride = tuple(strides))
            self.apool_in = nn.AvgPool2d(tuple(ks), stride=tuple(strides))
            self.conv2 = nn.Conv2d(input_dim[1] * 5, 1, (1, 1))
            self.act = nn.ReLU()

            self.lin = nn.Linear(torch.prod((input_dim[-2:] - ks)//strides + 1), output_lg)

        def forward(self, x, y):
            emb_y = self.conv1(y)
            emb_y = self.act(emb_y)
            emb_y = torch.cat((self.apool_in(y), emb_y), dim=1)
            emb_y = self.conv2(emb_y)
            emb_y = self.act(emb_y).flatten(start_dim = 1)
            out = self.lin(emb_y) + x

            return self.act(out)
        
    torch.manual_seed(42)
    ic = InvConv(torch.tensor((2,3,3)), ConvEmb(torch.tensor((1,1,1,1)), 16))
    x = torch.randint(10, (2,2,3,3)).float()
    print(x)
    z,_ = ic(x, torch.ones((2,1,1,1)))
    print(z)
    x = ic.inverse(z, torch.ones((2,1,1,1)))
    print(x)
    pass
