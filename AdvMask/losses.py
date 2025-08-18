import torch.nn as nn
from torch.nn import CosineSimilarity


def get_loss(config):
    if config.dist_loss_type == 'cossim':
        return CosLoss()


class CosLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cos_sim = CosineSimilarity()

    def forward(self, emb1, emb2):
        cs = self.cos_sim(emb1, emb2)
        
        return (cs + 1) / 2