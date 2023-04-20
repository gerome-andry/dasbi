import torch
import torch.nn as nn
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

def time_embed(t, space_dim):
    h, w = space_dim
    freq = (torch.cat([torch.arange(i, i + h // 2) for i in range(1, w + 1)]) * torch.pi).to(t)
    t = freq * t[..., None]
    t = t.transpose(0, 1)
    t = torch.cat(
        [torch.stack([tc, ts], dim=1) for tc, ts in zip(t.cos(), t.sin())], dim=-1
    )
    
    t = t.reshape((-1, 1) + space_dim)

    return t


def pos_embed(space_dim):
    h = (
        torch.pi
        * torch.arange(space_dim[0])
        .unsqueeze(1)
        .repeat(1, space_dim[1])[None, None, ...]
        / (2 * space_dim[0])
    )
    w = (
        torch.pi
        * torch.arange(space_dim[1]).repeat(space_dim[0], 1)[None, None, ...]
        / (2 * space_dim[1])
    )

    return torch.cat((h.sin(), w.sin()), dim=1)


class CombineConv(nn.Module):
    def __init__(self, chan, nc, kern=3, type = '2D'):
        super().__init__()
        self.act = nn.ELU()

        k = kern
        p = (kern - 1)//2
        if type == '1D':
            k = (k, 1)
            p = (p, 0)

        self.combine = nn.ModuleList(
            [nn.Conv2d(chan, chan, k, padding=p) for _ in range(nc)]
        )

    def forward(self, x):
        # print(x.shape)
        for c in self.combine:
            x = x + self.act(c(x))

        return x


class SampleCheck(nn.Module):
    def __init__(self, x_shape, y_shape, nc = 3, hidden_fc = 64, init_c=16, reduce=3, factor=2, type = '2D'):
        super().__init__()
        # spatial encoding
        # time embedding
        add_chan = 2 + 1  # spatial encoding + time encoding
        self.extend_y = nn.ModuleList(
            [nn.Upsample(x_shape[-2:]), nn.Conv2d(y_shape[1] + add_chan, init_c, 1)]
        )
        self.act = nn.ELU()
        self.combine_x = nn.Conv2d(x_shape[1] + add_chan, init_c, 1)
        self.extract = nn.ModuleList([CombineConv((2*init_c) + init_c*i, nc, type = type) for i in range(reduce)])
        k = factor
        s = factor
        if type == '1D':
            k = (k,1)
            s = (s,1)
        self.reduce = nn.ModuleList([nn.Conv2d((2*init_c) + init_c*i, 
                                               (2*init_c) + init_c*(i + 1), 
                                               k, stride = s) for i in range(reduce)])

        in_size = torch.prod(torch.tensor(x_shape[-2:]))
        self.head = nn.ModuleList([nn.Conv2d((2*init_c) + (init_c*reduce), 1, 1), nn.Linear(in_size // (factor**reduce), hidden_fc)])
        self.head.extend([nn.Linear(hidden_fc, hidden_fc) for _ in range(nc - 1)])

        self.classify = nn.ModuleList([nn.Linear(hidden_fc, 2), nn.Softmax(dim = 1)])

    def forward(self, x, y, t):
        sp_x = pos_embed(x.shape[-2:]).expand(x.shape[0], -1, -1, -1).to(x)
        sp_y = pos_embed(y.shape[-2:]).expand(y.shape[0], -1, -1, -1).to(x)
        t_x = time_embed(t, x.shape[-2:])
        t_y = time_embed(t, y.shape[-2:])
        x_in = torch.cat((t_x, sp_x, x), dim = 1)
        y_in = torch.cat((t_y, sp_y, y), dim = 1)

        y_emb = y_in
        for ey in self.extend_y:
            y_emb = ey(y_emb)

        x = torch.cat((y_emb, self.combine_x(x_in)), dim = 1)

        for e,r in zip(self.extract, self.reduce):
            x = e(x)
            x = r(x)
        
        x = self.head[0](x).flatten(start_dim = 1)
        for lay in self.head[1:]:
            x = self.act(lay(x))

        for c in self.classify:
            x = c(x)

        return x
    
    def loss(self, x, y, t, labels):
        logits = self(x,y,t)
        loss = nn.CrossEntropyLoss()
        return loss(logits, labels)
    
    def AUC(self, x, y, t, labels, levels = 100):
        logits = self(x,y,t)
        # logits = nn.functional.softmax(torch.rand_like(labels), dim = 1)#labels.clone()
        # logits[0,0] = .8
        # logits[0,1] = .2
        thresholds = torch.linspace(0,1,levels)
        fpr = [0, 1]
        tpr = [0, 1]
        N = logits.shape[0]
        for th in thresholds:
            pos_pred = logits[:,0] > th
            pos_gt = labels[:,0] == 1 
            tp = torch.bitwise_and(pos_pred, pos_gt).sum()
            fp = pos_pred.sum() - tp
            tn = torch.bitwise_and(torch.bitwise_not(pos_pred),torch.bitwise_not(pos_gt)).sum()
            fn = N - (fp + tp + tn)
            tpr.append(tp/(tp + fn))
            fpr.append(fp/(tn + fp))
            # print(tpr[-1], fpr[-1])

        fpr, indices = torch.sort(torch.tensor(fpr))
        tpr = torch.tensor(tpr)[indices]
        # import matplotlib.pyplot as plt 
        # plt.plot(fpr, tpr)
        # plt.show()
        return fpr, tpr, torch.trapezoid(tpr, fpr)
    

# m = SampleCheck((1,1,32,32), (1,1,8,8), type = '2D')
# B = 2**20
# labels = torch.randint(2,(B,2)).float()
# labels[:,1] = 1 - labels[:,0]
# print(labels)

# print(m.AUC(None, None, None, labels))