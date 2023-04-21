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
        self.combine.extend([nn.Conv2d(16*(i+1), 16*(i+2), 3, padding = 1) for i in range(conv_lay)])
        self.extract = nn.ModuleList([nn.Conv2d(16*(conv_lay+1), 32, 1),
                                      nn.Conv2d(32, self.x_shape[1] - time_features - 2, 1)])
        self.act = nn.ELU()
        # h, w = out_shape[-2:]
        # self.obs = observer_mask
        # self.act = nn.ELU()
        # self.register_buffer(
        #     "freq",
        #     torch.cat([torch.arange(i, i + h // 2) for i in range(1, w + 1)])
        #     * torch.pi,
        # )

        # self.extract = nn.ModuleList(
        #     [
        #         nn.Conv2d(self.y_shape[1] if i == 0 else 4 * i + self.y_shape[1], self.y_shape[1] + 4 * (i + 1), 1)#, padding = 1)
        #         for i in range(conv_lay)
        #     ]
        # )
        # # self.ln = nn.ModuleList(
        # #     [
        # #         nn.LayerNorm(self.y_shape[-2:])
        # #         for i in range(conv_lay)
        # #     ]
        # # )


        # # assumption obs have smaller size than x
        # if self.obs is None:
        #     # self.upsample = nn.ConvTranspose2d(
        #     #     4 * conv_lay,
        #     #     32,
        #     #     (
        #     #         self.x_shape[-2] - self.y_shape[-2] + 1,
        #     #         self.x_shape[-1] - self.y_shape[-1] + 1,
        #     #     ),
        #     # )
        #     self.upsample = nn.Upsample(tuple(self.x_shape[-2:]))

        # last_c = self.x_shape[1]*4
        # self.mix_up = nn.ModuleList([nn.Conv2d(4*conv_lay, 32, 1), nn.Conv2d(32, last_c, 3, padding = 1)])
        # self.head = nn.Conv2d(last_c, self.x_shape[1] - 1, 1)

    # def time_embed(self, t):
    #     # extend to multiple times
    #     # time between 0 and 1 ?
    #     t = self.freq * t[..., None]
    #     t = t.transpose(0, 1)
    #     t = torch.cat(
    #         [torch.stack([tc, ts], dim=1) for tc, ts in zip(t.cos(), t.sin())], dim=-1
    #     )
    #     # import matplotlib.pyplot as plt
    #     # plt.imshow(t)
    #     # plt.show()
    #     # plt.clf()
    #     t = t.reshape((-1, 1) + self.x_shape[2:])

    #     return t

    def forward(self, y, t):
        b = y.shape[0]
        t_emb = self.t_emb(t)
        space_emb = self.s_emb.expand(b,-1,-1,-1)

        spy = pos_embed(y.shape[-2:]).expand(b, -1, -1, -1).to(y)
        y_emb = torch.cat((spy,y), dim = 1)
        y_emb = self.up(y_emb)
        if self.obs is not None:
            mask = self.obs[None, None, ...].expand(b, -1, -1, -1)
            y_emb = torch.cat((y_emb,mask), dim = 1)
        
        for c in self.combine:
            y_emb = self.act(c(y_emb))

        for e in self.extract:
            y_emb = e(y_emb)

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



