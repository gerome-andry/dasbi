import torch
import torch.nn as nn
from ..diagnostic.classifier import TimeEmb, pos_embed

class EmbedObs(nn.Module):
    def __init__(self, in_shape, out_shape, conv_lay=2, observer_mask = None, time_features = 4):
        super().__init__()
        self.x_shape = tuple(out_shape)
        self.y_shape = tuple(in_shape)
        self.t_emb = lambda t : TimeEmb(time_features)(t, self.x_shape[-2:])
        self.register_buffer('s_emb', pos_embed(self.x_shape[-2:]))
        self.obs = observer_mask
        self.up = nn.Upsample(tuple(self.x_shape[-2:]))
        add_chan = 2 if self.obs is None else 3 # spatial and obs_mask
        self.combine = nn.ModuleList([nn.Conv2d(self.y_shape[1] + add_chan, 16, 1)])
        self.combine.extend([nn.Conv2d(16*(i+2), 16*(i+2), 3, padding = 1) for i in range(conv_lay)])
        self.extract = nn.ModuleList([nn.Conv2d(16*(conv_lay+1), 32, 1),
                                      nn.Conv2d(32, self.x_shape[1] - time_features - 2, 1)])
        self.act = nn.ELU()
        

    def forward(self, y, t):
        b = y.shape[0]
        t_emb = self.t_emb(t)
        space_emb = self.s_emb.expand(b,-1,-1,-1)

        spy = pos_embed(y.shape[-2:]).expand(b, -1, -1, -1).to(y)
        y_emb = torch.cat((spy,y), dim = 1)
        y_emb = self.up(y_emb)
        
        if y_emb.isnan().sum() > 0:
            print("UP",y_emb.isnan().sum())
            y_emb = y_emb.nan_to_num()
        
        if self.obs is not None:
            mask = self.obs[None, None, ...].expand(b, -1, -1, -1)
            y_emb = torch.cat((y_emb,mask), dim = 1)
        
        y_emb = self.act(self.combine[0](y_emb))
        if y_emb.isnan().sum() > 0:
            print("COMB INIT",y_emb.isnan().sum())
            y_emb = y_emb.nan_to_num()
        y1 = y_emb.clone()
        for c in self.combine[1:]:
            y_emb = self.act(c(torch.cat((y1, y_emb), dim = 1)))
            if y_emb.isnan().sum() > 0:
                print("COMB",y_emb.isnan().sum())
                y_emb = y_emb.nan_to_num()

        for e in self.extract:
            y_emb = e(y_emb)
            if y_emb.isnan().sum() > 0:
                print("EXT",y_emb.isnan().sum())
                y_emb = y_emb.nan_to_num()

        return torch.cat((t_emb, space_emb, y_emb), dim = 1)
        # t_emb = self.time_embed(t)
        # if t_emb.isnan().sum():
        #     print('T')
        #     exit()
        # if y.isnan().sum():
        #     print('Y')
        #     exit()

        # if self.obs is None:
        #     y_emb = y
        # else:
        #     mask = self.obs[None,None,...].expand(y.shape[0], y.shape[1], -1, -1)
        #     y_emb = torch.zeros_like(mask)
        #     y_emb[mask == 1] = y.flatten()
        #     y_emb = torch.cat((y_emb, mask[:,:1,...]), dim = 1) 

        # y_in = y_emb.clone()
        # for i, e in enumerate(self.extract):
        #     # print(y_emb.isnan().sum(), y_emb.min(), y_emb.max(), y_emb.numel(), y_emb.shape)
        #     # y_emb = l(y_emb)
        #     if i != 0:
        #         y_emb = torch.cat((y_emb, y_in), dim = 1) #Res connection
        #     y_emb = e(y_emb)
        #     y_emb = self.act(y_emb)

        # if y_emb.isnan().sum():
        #     print('EXT')
        #     y_emb = torch.nan_to_num(y_emb)
        #     # exit()
        # if self.obs is None:
        #     y_emb = self.upsample(y_emb)
        #     y_emb = self.act(y_emb)
        # if y_emb.isnan().sum():
        #     print('UP')
        #     y_emb = torch.nan_to_num(y_emb)
        #     # exit()
        # for mu in self.mix_up:
        #     y_emb = mu(y_emb)
        #     y_emb = self.act(y_emb)
        # if y_emb.isnan().sum():
        #     print('MU')
        #     y_emb = torch.nan_to_num(y_emb)
        #     # exit()
        # y_emb = self.head(y_emb)
        # y_emb = self.act(y_emb)
        # if y_emb.isnan().sum():
        #     print('HEAD')
        #     y_emb = torch.nan_to_num(y_emb)
        #     # exit()
        # return torch.cat((y_emb, t_emb), dim=1)



