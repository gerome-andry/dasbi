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
        batch_size = x.shape[0]
        ladj = self.log_sig.sum()*z.new_ones(batch_size)

        return z, ladj
    
    def inverse(self, z, context = None):
        assert self.mu is not None, "Parameters not initialized (need forward pass)"

        x = z*(self.log_sig.exp()) + self.mu
        ladj = None

        return x, ladj


class InvConv(Transform):
    # fix multiple channels ???

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
            self.unpad = lambda x : x[...,hpad:,wpad:]
            self.mask[-1,-1] = 0
        elif self.mode == 'UR':
            self.pad = lambda x : nn.functional.pad(x, (0, wpad, hpad, 0))
            self.unpad = lambda x : x[...,hpad:,:-wpad] if wpad > 0 else x[...,hpad:,:]
            self.mask[-1,0] = 0
        elif self.mode == 'LL':
            self.pad = lambda x : nn.functional.pad(x, (wpad, 0, 0, hpad))
            self.unpad = lambda x : x[...,:-hpad,wpad:] if hpad > 0 else x[...,:,wpad:]
            self.mask[0,-1] = 0
        else: # LR
            self.pad = lambda x : nn.functional.pad(x, (0, wpad, 0, hpad))
            self.unpad = lambda x : x[...,:-hpad,:-wpad] if hpad > 0 and wpad > 0 else x[...,:-hpad,:] if hpad > 0 and wpad == 0 else x[...,:,:-wpad] if hpad == 0 and wpad > 0 else x
            self.mask[0,0] = 0

    def forward(self, x, context):
        weights = self.conv_kern()
        batch_size = x.shape[0]
        weights = weights.reshape((1, 1)+ self.ks)
        x_p = self.pad(x)

        z = nn.functional.conv2d(x_p, weights)
        ladj = z.new_zeros(batch_size)

        return z, ladj

    def inverse(self, z, context):
        weights = self.conv_kern()
        c_mat = self.fc_from_conv(weights, z)
        
        b,c,h,w = z.shape

        z = z.permute((2,3,1,0))
        z = z.reshape((c*h*w, b))
        x = torch.linalg.solve(c_mat, z).reshape(h,w,c,b)
        x = x.permute((3,2,0,1))
        ladj = None

        return x, ladj

    def conv_kern(self):
        ck = torch.ones(self.ks)
        ck[self.mask] = self.kernel 

        return ck

    def fc_from_conv(self, kern, x):
        xdim = x.shape[-2:]
        K = torch.ones(xdim)
        K = self.pad(K)
        c_mat = torch.zeros((x[0,...].numel(),)*2)

        row = 0
        for i in range(K.shape[0] - kern.shape[0] + 1):
            for j in range(K.shape[1] - kern.shape[1] + 1):
                K_check = torch.zeros_like(K)
    
                K_check[i:i+kern.shape[0], j:j+kern.shape[1]] = \
                    kern*K[i:i+kern.shape[0],j:j+kern.shape[1]]
                

                K_check = self.unpad(K_check)
                c_mat[row, :] = K_check.flatten()
                row += 1

        return c_mat

class SpatialSplit(Transform):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, context=None):
        b,c,h,w = x.shape
        z = x.reshape((b,4*c,h//2,w//2))
        z1 = z[:,0,...]
        z2 = z[:,1:,...].reshape((b,c,))

        return (z1, z2)


if __name__ == '__main__':
    # an = ActNorm()
    # x = 2 + torch.randn((10,5,5))
    # print(x)
    # z,l = an.forward(x)
    # print(z.mean(dim = 0), z.std(dim = 0))
    # print(an.inverse(z))

    ic = InvConv((3,3), mode = 'UL')
    x = torch.randint(10, (1,1,3,3))
    print(x)
    z,_ = ic.forward(x.float(), 0)
    print(z)
    x,_ = ic.inverse(z, 0)
    print(x)
