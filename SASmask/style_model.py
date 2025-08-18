import torch
import torch.nn.functional as F
from torch import nn


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, myInstanceNorm2d):
        m.weight.data.normal_(0.0, 0.04 ** 2)
        m.bias.data.fill_(0)


mean = torch.Tensor((0.485, 0.456, 0.406)).unsqueeze(-1).unsqueeze(-1).cuda()
std = torch.Tensor((0.229, 0.224, 0.225)).unsqueeze(-1).unsqueeze(-1).cuda()
unnormalize = lambda x: x * std + mean
normalize = lambda x: (x - mean) / std


class myInstanceNorm2d(nn.InstanceNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False):
        super(myInstanceNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    def getGamma(self):
        return self.weight

    def getBeta(self):
        return self.bias

    def setGamma(self, gamma):
        self.weight.data = gamma

    def setBeta(self, beta):
        self.bias.data = beta


class CIN(nn.Module):
    def __init__(self, style_num, in_channels):
        super(CIN, self).__init__()
        self.norms = nn.ModuleList([myInstanceNorm2d(in_channels, affine=True) for i in range(style_num)])
        self.in_channels = in_channels
        self.mixNorm = myInstanceNorm2d(in_channels, affine=True)
        # self.instance_norm = nn.InstanceNorm2d(in_channels, affine=False)
        # self.weights = nn.Parameter(torch.ones(style_num))
        # self.bias = nn.Parameter(torch.zeros(style_num))

        # self.mixNorm = myInstanceNorm2d(in_channels, affine=False)

    def forward(self, x, style_weights, mix=True):
        device = x.device
        if not mix:
            out = torch.stack([self.norms[torch.nonzero(style_weights)[i][1]](x[i]) for i in range(x.size(0))])
        else:
            beta_mix = torch.zeros(x.size(0), self.in_channels).to(device)
            gamma_mix = torch.zeros(x.size(0), self.in_channels).to(device)
            for i in range(style_weights.size(0)):
                for j in range(style_weights.size(1)):
                    beta_mix[i] = beta_mix[i].clone() + self.norms[j].getBeta() * style_weights[i, j]
                    gamma_mix[i] = gamma_mix[i].clone() + self.norms[j].getGamma() * style_weights[i, j]

            out = nn.InstanceNorm2d(self.in_channels, affine=False)(x)
            # print(out.size())
            for i in range(out.size(0)):
                for j in range(out.size(1)):
                    out[i, j, :, :] = gamma_mix[i, j] * out[i, j, :, :].clone() + beta_mix[i, j]
        return out
    #     mix_weights = 0
    #     mix_bias = 0
    #     out = self.instance_norm(x)
    #     for i in range(style_weights.size(0)):
    #         mix_weights += style_weights[i] * self.weights[i]
    #         mix_bias += style_weights[i] * self.bias[i]
    #     out = mix_weights * out + mix_bias
    #     return out

    # def addStyle(self, num_styles, in_channels):
    #     num_norms = len(self.norms)
    #     for i in range(num_styles):
    #         self.norms.append(myInstanceNorm2d(in_channels, affine=True))
    #         self.norms[num_norms + i].weight.data.normal_(0.0, 0.04 ** 2)
    #         self.norms[num_norms + i].bias.data.fill_(0)


class StyleModel(nn.Module):
    def __init__(self, num_styles=10):
        super(StyleModel, self).__init__()
        self.pad0 = nn.ReflectionPad2d(4)
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=9, stride=1, padding=0)
        self.CIN0 = CIN(num_styles, in_channels=32)

        self.pad1 = nn.ReflectionPad2d((0, 1, 0, 1))
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.CIN1 = CIN(num_styles, in_channels=64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0)
        self.CIN2 = CIN(num_styles, in_channels=128)

        self.pad2 = nn.ReflectionPad2d(1)
        self.resBlock1 = nn.ModuleList(
            [nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0) for i in range(5)])
        self.CINres1 = nn.ModuleList([CIN(num_styles, in_channels=128) for i in range(5)])
        self.resBlock2 = nn.ModuleList(
            [nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0) for i in range(5)])
        self.CINres2 = nn.ModuleList([CIN(num_styles, in_channels=128) for i in range(5)])

        self.upsample0 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.CINup0 = CIN(num_styles, in_channels=64)
        self.upsample1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.CINup1 = CIN(num_styles, in_channels=32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=9, stride=1, padding=0)
        self.CIN3 = CIN(num_styles, in_channels=3)

    def forward(self, x, style_weights=torch.ones((10,)), mix=False):
        out = self.pad0(x)
        out = self.conv0(out)
        out = self.CIN0(out, style_weights, mix)
        out = F.relu(out)

        out = self.pad1(out)
        out = self.conv1(out)
        out = self.CIN1(out, style_weights, mix)
        out = F.relu(out)

        out = self.pad1(out)
        out = self.conv2(out)
        out = self.CIN2(out, style_weights, mix)
        out = F.relu(out)

        for i in range(5):
            res_in = out
            out = self.pad2(out)
            out = self.resBlock1[i](out)
            out = self.CINres1[i](out, style_weights, mix)
            out = F.relu(out)

            out = self.pad2(out)
            out = self.resBlock2[i](out)
            out = self.CINres2[i](out, style_weights, mix)
            out = res_in + out

        out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = self.pad2(out)
        out = self.upsample0(out)
        out = self.CINup0(out, style_weights, mix)
        out = F.relu(out)

        out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = self.pad2(out)
        out = self.upsample1(out)
        out = self.CINup1(out, style_weights, mix)
        out = F.relu(out)

        out = self.pad0(out)
        out = self.conv3(out)
        out = self.CIN3(out, style_weights, mix)
        out = torch.sigmoid(out)

        return out

    # def addStyle(self, num_styles):
    #     for child in model.children():
    #         if isinstance(child, CIN):
    #             in_channels = child.norms[0].num_features
    #             child.addStyle(num_styles, in_channels)
    #         if isinstance(child, nn.ModuleList):
    #             for grandchild in child:
    #                 if isinstance(grandchild, CIN):
    #                     in_channels = grandchild.norms[0].num_features
    #                     grandchild.addStyle(num_styles, in_channels)
