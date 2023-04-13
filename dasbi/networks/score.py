import torch 
import torch.nn as nn     

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        var, mu = torch.var_mean(x, dim = self.dim, keepdim = True, unbiased = True)

        return (x - mu)/(var + self.eps).sqrt()
    
class AttentionSkip(nn.Module):
    def __init__(self, chan, type = '2D'):
        super().__init__()
        k = 2 if type == '2D' else (2,1)
        self.scale_dpath = nn.Conv2d(chan, chan, k, stride = k)
        self.act = nn.ELU()
        self.up = nn.Upsample(scale_factor = k)
        self.combine = nn.Conv2d(chan, chan, 1, 1)
        
    def spatialSoftMax(self, x):
        return x.exp()/(x.exp().sum(dim = (-2,-1), keepdim = True))
    
    def forward(self, x, q):
        w_att = self.act(q + self.scale_dpath(x))
        w_att = self.combine(self.up(w_att))
        w_att = self.spatialSoftMax(self.act(w_att))

        return x*w_att

class downUpLayer(nn.Module):
    def __init__(self, input_c, scale_c = 2, kernel_sz = 3, n_c = 2):
        super().__init__()
        output_c = int(input_c*scale_c)
        self.act = nn.ELU()
        self.ln = LayerNorm((-2, -1))
        
        if type(kernel_sz) is tuple:
            k = kernel_sz[0]
        else:
            k = kernel_sz
        self.conv = nn.ModuleList([nn.Conv2d(input_c if i == 0 else output_c, output_c, 
                                             kernel_sz, padding = (k - 1)//2) for i in range(n_c)])

    def forward(self, x):
        x = self.ln(x)
        x = self.act(self.conv[0](x))
        for c in self.conv[1:]:
            x = self.act(x + c(x))

        return x


class ScoreAttUNet(nn.Module):
    def __init__(self, input_c = 1, depth = 3, 
                input_scale = 64, spatial_scale = 2, n_c = 2, 
                ks = 3, type = '2D'):
        super().__init__()

    #CREATION
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.reduceChannel = nn.ModuleList()
        self.pool = nn.ModuleList()
        self.att_skip = nn.ModuleList()
    #DOWN
        if type == '1D':
            ks = (ks, 1)
        self.down.append(downUpLayer(input_c, input_scale, n_c = n_c, kernel_sz = ks))
        nextLayC = int(input_scale*input_c)
        self.att_skip.extend([AttentionSkip(chan = nextLayC * spatial_scale ** i, type = type)
                            for i in range(depth - 1)])
        self.down.extend([downUpLayer(nextLayC * spatial_scale ** i, spatial_scale, n_c = n_c, kernel_sz = ks)
                            for i in range(depth - 1)])
        self.pool.extend([nn.Conv2d(nextLayC * spatial_scale ** i, nextLayC * spatial_scale ** i, 2, stride = 2)
                            for i in range(depth - 1)])
        nextLayC = nextLayC * spatial_scale ** (depth-1)
    #UP & REDUCTION
        self.reduceChannel.extend([nn.Conv2d(nextLayC//(spatial_scale**i), nextLayC//(spatial_scale**(i+1)), 1)
                                    for i in range(depth)])
        self.up.extend([downUpLayer(nextLayC//(spatial_scale**i), 1/spatial_scale, n_c = n_c, kernel_sz = ks)
                                    for i in range(depth - 1)])
        outC = nextLayC//(spatial_scale**depth)
        self.tail = nn.Conv2d(outC, input_c, 1)

        self.upSample = nn.Upsample(scale_factor = ((2) if type == '2D' else (2,1)))


    def fullScaleUp(self, x, idx):
        y = self.reduceChannel[idx](x)
        return self.upSample(y)


    def forward(self, x):
        x_skip = []

        for d,p in zip(self.down[:-1], self.pool):
            x = d(x)
            x_skip.append(x)
            x = p(x)
        
        x = self.down[-1](x)
        for i,u in enumerate(self.up):
            x = self.fullScaleUp(x, i)
            if i == len(self.up) - 1:
                break
            # skip_shape = x.shape
            #for size compatibility with residual connections if the size is not multiple of 2
            xFull = torch.cat((x, self.att_skip[-(i+1)](x_skip[-(i+1)], x)), 1)
            x = u(xFull)

        x = self.up[-1](x)
        x = self.upSample(x)
        x = self.tail(x)
        
        return x
    

m = ScoreAttUNet(depth = 4, type = '2D', input_c = 5, spatial_scale=2, n_c = 2, input_scale=64/5)
print(m)