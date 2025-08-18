import torch
from torch import nn
from torchvision.models import vgg16
from collections import namedtuple


class Vgg16(torch.nn.Module):
    '''
    https://discuss.pytorch.org/t/accessing-intermediate-layers-of-a-pretrained-network-forward/12113
    '''

    def __init__(self):
        super(Vgg16, self).__init__()
        features = list(vgg16(pretrained=True).features)[:23]
        self.features = nn.ModuleList(features).eval()

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {3, 8, 15, 22}:
                results.append(x)

        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        return vgg_outputs(*results)
