import torch
import torch.nn as nn
import transforms as tf


class ConvEmb(nn.Module):
    def __init__(self, input_dim, output_lg):
        super().__init__()
        ks = torch.clamp(input_dim[-2:] // 3, 1)
        self.conv1 = nn.Conv2d(input_dim[1], input_dim[1] * 4, ks)
        self.mpool_in = nn.MaxPool2d(tuple(ks))
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


class ConvStep(tf.Transform):
    def __init__(self, context_dim, n_conv, kernel_sz):
        super().__init__()
        # ADD QUAD COUPLING !!!
        self.mod = nn.ModuleList([tf.SpatialSplit(), tf.ActNorm()])
        self.nc = n_conv
        self.conv_mod = nn.ModuleList()
        mode = ['UL', 'LR', 'UR', 'LL']

        for _ in range(n_conv):
            k_net = ConvEmb(context_dim, torch.prod(kernel_sz) - 1)
            self.conv_mod.append(
                nn.ModuleList(
                    [
                        tf.InvConv(kernel_sz, k_net, mode=mode[0]),
                        tf.InvConv(kernel_sz, k_net, mode=mode[1]),
                        tf.InvConv(kernel_sz, k_net, mode=mode[2]),
                        tf.InvConv(kernel_sz, k_net, mode=mode[3]),
                        tf.ActNorm(),
                    ]
                )
            )
            mode.reverse()

    def forward(self, x, context):
        b = x.shape[0]
        ladj = x.new_zeros(b)

        print(x.shape)
        z = x
        for m in self.mod:
            z, ladj_i = m(z)
            print(z.shape)
            ladj += ladj_i

        c = z.shape[1]
        for c_ls in self.conv_mod:
            for i, mc in enumerate(c_ls[:-1]):
                for j in range(c // 4):
                    z[:, i*c//4 + j, ...], ladj_i = mc(z[:, i*c//4 + j, ...].unsqueeze(1), context)
                    ladj += ladj_i
            
            z, ladj_i = c_ls[-1](z)
            ladj += ladj_i
        
        #add quadcoup here

        return z, ladj
    
    def inverse(self, z, context):
        x = z
        #add quadcoup here

        c = x.shape[1]
        for c_ls in reversed(self.conv_mod):
            x, _ = c_ls[-1].inverse(x)
            for i, mc in enumerate(c_ls[:-1]): #not reverse to keep the same padding order
                for j in range(c // 4):
                    x[:, i*c//4 + j, ...], _ = mc.inverse(x[:, i*c//4 + j, ...].unsqueeze(1), context)

        for m in reversed(self.mod):
            x, _ = m.inverse(x)

        ladj = None

        return x, ladj


if __name__ == '__main__':
    cs = ConvStep(torch.tensor((1,1,3,3)), 1, torch.tensor((3,3)))
    x = torch.randint(10, (2,1,4,4)).float()
    y = torch.randint(10, (1,1,3,3)).float()
    print(x)
    z, l = cs(x,y)
    print(z)
    # print(l)
    x, _ = cs.inverse(z,y)
    # print(x)
    # print(cs)