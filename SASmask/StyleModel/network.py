import torch
import torch.nn.functional as F
from torch import nn


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, myInstanceNorm2d):
        m.weight.data.normal_(0.0, 0.04 ** 2)
        m.bias.data.fill_(0)


def norm(image):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    output = torch.stack([(image[0][i] - mean[i]) / std[i] for i in range(3)])
    return output.unsqueeze(0)


def gram_matrix(input):
    '''
    from: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html#style-loss
    '''
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


def loss_calc(content, style, pastiche, lambda_c, lambda_s):
    c = F.mse_loss(pastiche.relu4_3, content.relu4_3)

    s1 = F.mse_loss(gram_matrix(pastiche.relu1_2), gram_matrix(style.relu1_2))
    s2 = F.mse_loss(gram_matrix(pastiche.relu2_2), gram_matrix(style.relu2_2))
    s3 = F.mse_loss(gram_matrix(pastiche.relu3_3), gram_matrix(style.relu3_3))
    s4 = F.mse_loss(gram_matrix(pastiche.relu4_3), gram_matrix(style.relu4_3))
    #    print("Content:",c.data.item())
    #    print("Style:",s1.data.item()+s2.data.item()+s3.data.item()+s4.data.item())

    loss = lambda_c * c + lambda_s * (s1 + s2 + s3 + s4)

    return loss


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
        self.mixNorm = myInstanceNorm2d(in_channels, affine=True);

    def forward(self, x, style_ids=0, style_weights=[0], mix=False):
        if self.training or not mix:
            out = self.norms[style_ids](x)
        else:
            beta_mix = 0
            gamma_mix = 0
            for i in range(len(style_weights)):
                beta_mix += self.norms[i].getBeta() * style_weights[i]
                gamma_mix += self.norms[i].getGamma() * style_weights[i]
            self.mixNorm.setGamma(gamma_mix)
            self.mixNorm.setBeta(beta_mix)
            out = self.mixNorm(x)

        #         print(self.norms[0].getGamma())
        return out

    # def addStyle(self, num_styles, in_channels):
    #     num_norms = len(self.norms)
    #     for i in range(num_styles):
    #         self.norms.append(myInstanceNorm2d(in_channels, affine=True))
    #         self.norms[num_norms + i].weight.data.normal_(0.0, 0.04 ** 2)
    #         self.norms[num_norms + i].bias.data.fill_(0)


class StyleModel(nn.Module):
    def __init__(self, num_styles):
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

    def forward(self, x, style_ids=0, style_weights=[0], mix=False):
        out = self.pad0(x)
        out = self.conv0(out)
        out = self.CIN0(out, style_ids, style_weights, mix)
        out = F.relu(out)

        out = self.pad1(out)
        out = self.conv1(out)
        out = self.CIN1(out, style_ids, style_weights, mix)
        out = F.relu(out)

        out = self.pad1(out)
        out = self.conv2(out)
        out = self.CIN2(out, style_ids, style_weights, mix)
        out = F.relu(out)

        for i in range(5):
            res_in = out
            out = self.pad2(out)
            out = self.resBlock1[i](out)
            out = self.CINres1[i](out, style_ids, style_weights, mix)
            out = F.relu(out)

            out = self.pad2(out)
            out = self.resBlock2[i](out)
            out = self.CINres2[i](out, style_ids, style_weights, mix)
            out = res_in + out

        out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = self.pad2(out)
        out = self.upsample0(out)
        out = self.CINup0(out, style_ids, style_weights, mix)
        out = F.relu(out)

        out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = self.pad2(out)
        out = self.upsample1(out)
        out = self.CINup1(out, style_ids, style_weights, mix)
        out = F.relu(out)

        out = self.pad0(out)
        out = self.conv3(out)
        out = self.CIN3(out, style_ids, style_weights, mix)
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
