import torch 
import torch.nn as nn     

class Transform(nn.Module):
    def forward(self, x, context = None):
        raise NotImplementedError()

    def inverse(self, z, context = None):
        raise NotImplementedError()


class ActNorm(Transform):
    # Take a look at conditioning ? 
    def __init__(self):
        super().__init__()
        self.mu = None
        self.log_sig = None

    def forward(self, x, context = None):
        if self.mu is None:
            self.mu = nn.Parameter(x.mean(dim = 0))
            self.log_sig = nn.Parameter(x.std(dim = 0).log())
        
        z = (x - self.mu)/self.log_sig.exp()
        batch_size, _, _ = x.shape
        ladj = self.log_sig.sum()*z.new_ones(batch_size)

        return z, ladj
    
    def inverse(self, z, context = None):
        assert self.mu is not None, "Parameters not initialized (need forward pass)"

        x = z*(self.log_sig.exp()) + self.mu
        ladj = None

        return x, ladj


class InvConv(Transform):
    def __init__(self, kernel_sz, mode = 'UL'):
        params_k = torch.tensor(kernel_sz).prod()
        assert params_k > 1, "Too small kernel, must contain more than 1 element"
        super().__init__()

        self.mode = mode
        self.ks = kernel_sz
        self.kernel = nn.Parameter(torch.ones(params_k - 1))

        self.mask = torch.ones(self.ks, dtype = torch.bool)
        hpad, wpad = torch.tensor(self.ks) - 1
        if self.mode == 'UL':
            self.pad = lambda x : nn.functional.pad(x, (wpad, 0, hpad, 0))
            self.mask[-1,-1] = 0
        elif self.mode == 'UR':
            self.pad = lambda x : nn.functional.pad(x, (0, wpad, hpad, 0))
            self.mask[-1,0] = 0
        elif self.mode == 'LL':
            self.pad = lambda x : nn.functional.pad(x, (wpad, 0, 0, hpad))
            self.mask[0,-1] = 0
        else: # LR
            self.pad = lambda x : nn.functional.pad(x, (0, wpad, 0, hpad))
            self.mask[0,0] = 0

    def forward(self, x, context):
        weights = torch.ones(self.ks)
                
        weights[self.mask] = self.kernel
        weights = weights.reshape((1, 1)+ self.ks)
        x_p = self.pad(x)

        z = nn.functional.conv2d(x_p, weights)
        ladj = x.numel()

        return z, ladj

    def inverse(self, z, context):
        pass


if __name__ == '__main__':
    # an = ActNorm()
    # x = 2 + torch.randn((10,5,5))
    # print(x)
    # z,l = an.forward(x)
    # print(z.mean(dim = 0), z.std(dim = 0))
    # print(an.inverse(z))

    ic = InvConv((3,3), mode = 'LL')
    print(ic.forward(torch.randint(10, (1,1,3,3)).float(), 0))

