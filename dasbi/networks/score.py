import torch 
import torch.nn as nn 
from ..diagnostic.classifier import TimeEmb    

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        var, mu = torch.var_mean(x, dim = self.dim, keepdim = True, unbiased = True)
        # print(var, mu)
        return (x - mu)/(var + self.eps).sqrt()
    
class AttentionSkip(nn.Module):
    def __init__(self, chan, type = '2D', factor = 2):
        super().__init__()
        k = factor if type == '2D' else (factor,1)
        self.scale_dpath = nn.Conv2d(chan, chan, k, stride = k)
        self.act = nn.ELU()
        self.up = nn.Upsample(scale_factor = k)
        self.combine = nn.Conv2d(chan, chan, 1)
        self.att_weights = nn.Sigmoid()
        
    def spatialSoftMax(self, x):
        return x.exp()/(x.exp().sum(dim = (-2,-1), keepdim = True))
    
    def forward(self, x, q):
        w_att = self.act(q + self.scale_dpath(x))
        w_att = self.combine(self.up(w_att))
        w_att = self.att_weights(w_att)

        return x*w_att

class downUpLayer(nn.Module):
    def __init__(self, input_c, output_c, kernel_sz = 3, n_c = 2):
        super().__init__()
        # output_c = int(torch.round(torch.tensor(input_c*scale_c)))
        self.act = nn.ELU()
        self.ln = LayerNorm((-2, -1))
        
        if type(kernel_sz) is tuple:
            p = ((kernel_sz[0] - 1)//2,0)
        else:
            p = (kernel_sz - 1)//2

        self.conv = nn.ModuleList([nn.Conv2d(input_c if i == 0 else output_c, output_c, 
                                             kernel_sz, padding = p,
                                             padding_mode = 'circular') for i in range(n_c)])

    def forward(self, x):
        x = self.ln(x)
        x = self.act(self.conv[0](x))
        for c in self.conv[1:]:
            x = self.act(x + c(x))

        return x

class MLP(nn.Module):
    def __init__(self, in_d, out_d, hidden = 256, n_lay = 5):
        self.net = nn.ModuleList([nn.Linear(in_d, hidden)])
        self.net.extend([nn.Linear(hidden, hidden) for _ in range(n_lay)])
        self.net.append(nn.Linear(hidden, out_d))
        self.act = nn.ELU()
        
    def forward(self, x, k):
        x = x + k 
        for l in self.net:
            x = l(self.act(x))

        return x 
    
class ScoreAttUNet(nn.Module):
    def __init__(self, input_c = 1, output_c = 1, depth = 3, 
                input_hidden = 64, spatial_scale = 2, n_c = 3, 
                ks = 3, type = '2D', temporal = 5):
        super().__init__()

    #CREATION
        self.time_embed = TimeEmb(temporal)
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.reduceChannel = nn.ModuleList()
        self.pool = nn.ModuleList()
        self.att_skip = nn.ModuleList()
    #DOWN
        if type == '1D':
            ks = (ks, 1)
        
        input_c += temporal #for time embedding
        # input_scale = input_hidden/input_c
        self.down.append(downUpLayer(input_c, input_hidden, n_c = n_c, kernel_sz = ks))
        nextLayC = input_hidden
        self.att_skip.extend([AttentionSkip(chan = nextLayC * spatial_scale ** i, type = type, factor = spatial_scale)
                            for i in range(depth - 1)])
        # + 1 in dow path for time embedding
        self.down.extend([downUpLayer(nextLayC * spatial_scale ** i + temporal, nextLayC * spatial_scale ** (i+1), n_c = n_c, kernel_sz = ks)
                            for i in range(depth - 1)])
        c = ((spatial_scale) if type == '2D' else (spatial_scale,1))
        self.pool.extend([nn.Conv2d(nextLayC * spatial_scale ** i, nextLayC * spatial_scale ** i, c, stride = c)
                            for i in range(depth - 1)])
        nextLayC = nextLayC * spatial_scale ** (depth-1)
    #UP & REDUCTION
        self.reduceChannel.extend([nn.Conv2d(nextLayC//(spatial_scale**i), nextLayC//(spatial_scale**(i+1)), 1)
                                    for i in range(depth)])
        self.up.extend([downUpLayer(nextLayC//(spatial_scale**i), nextLayC//(spatial_scale**(i+1)), n_c = n_c, kernel_sz = ks)
                                    for i in range(depth - 1)])
        outC = nextLayC//(spatial_scale**depth)
        self.tail = nn.Conv2d(outC, output_c, 1)

        self.upSample = nn.Upsample(scale_factor = ((spatial_scale) if type == '2D' else (spatial_scale,1)))


    def fullScaleUp(self, x, idx):
        y = self.reduceChannel[idx](x)
        return self.upSample(y)


    def forward(self, x, k): #x contains x_k and context ; k is the score time index [0, 1]
        x_skip = []
        for d,p in zip(self.down[:-1], self.pool):
            k_emb = self.time_embed(k, x.shape[-2:])
            x = torch.cat((x, k_emb), 1)
            x = d(x)
            x_skip.append(x)
            x = p(x)
        
        k_emb = self.time_embed(k, x.shape[-2:])
        x = torch.cat((x, k_emb), 1)
        x = self.down[-1](x)
        for i,u in enumerate(self.up):
            x = self.reduceChannel[i](x)
            x = torch.cat((self.att_skip[-(i+1)](x_skip[-(i+1)], x), self.upSample(x)), 1)
            x = u(x)

        x = self.reduceChannel[-1](x)
        x = self.tail(x)
        
        return x
    

# m = ScoreAttUNet(depth = 3, type = '2D', input_c = 5, output_c = 2, spatial_scale=2, n_c = 2, input_hidden=64)
# print(m)