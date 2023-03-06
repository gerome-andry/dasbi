import torch 
import torch.nn as nn     

class EmbedObs(nn.Module):
    def __init__(self, in_shape, out_shape):
        super().__init__()
        self.x_shape = out_shape
        self.y_shape = in_shape

        h,w = out_shape[-2:]
        self.freq = torch.cat([torch.arange(i, i+h//2) for i in range(1, w+1)])*torch.pi

        # assumption obs have smaller size than x 
        

    def time_embed(self, t):
        # extend to multiple times
        # time between 0 and 1 ? 
        t = self.freq * t[...,None]
        t = t.transpose(0,1)
        t = torch.cat([torch.stack([tc, ts], dim = 1) for tc, ts in zip(t.cos(), t.sin())], dim = -1)
        t = t.reshape((-1,)+self.x_shape[1:])

        return t
    

if __name__ == '__main__':
    eo = EmbedObs((1,1,3,3), (1,1,4,4))
    print(eo.time_embed(torch.linspace(0,1,5)))


