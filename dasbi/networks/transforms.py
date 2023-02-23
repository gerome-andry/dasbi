import torch 
import torch.nn as nn     

class Transform(nn.Module):
    def forward(self, inputs, context):
        raise NotImplementedError()

    def inverse(self, inputs, context):
        raise NotImplementedError()


class ActNorm(Transform):
    # Take a look at conditioning
    def __init__(self, dim, eps = 1e-6):
        super().__init__()
        self.dim = dim
        self.mu = None
        self.sig = None
        self.eps = eps

    def forward(self, x):
        if self.mu is None:
            self.mu = nn.Parameter(x.mean(dim = 0))
            self.log_sig = nn.Parameter(x.std(dim = 0).exp())
        
        return (x - self.mu)/(self.sig + self.eps)
    
    def inverse(self, z):
        assert self.mu is not None, "Parameters not initialized (need forward pass)"

        return z*(self.sig + self.eps) + self.mu
        
