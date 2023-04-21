from dasbi.networks.embedding import EmbedObs
import torch

if __name__ == "__main__":
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    eo = EmbedObs((1, 1, 512//4, 1), (1, 16, 512, 1), conv_lay=3, observer_mask=torch.ones((512,1)))
    print(eo)
    m = eo(torch.zeros((1,1,512//4,1)), torch.zeros((1,1)))
    print(m.shape)
    print('NPE:',sum(param.numel() for param in eo.parameters()))