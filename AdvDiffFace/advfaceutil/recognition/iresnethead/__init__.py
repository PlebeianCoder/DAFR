import math

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms

from advfaceutil.datasets import FaceDatasets
from advfaceutil.datasets import FaceDatasetSize
from advfaceutil.recognition.base import RecognitionArchitecture

# __all__ = ['iresnet18', 'iresnet34', 'iresnet50', 'iresnet100']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


# https://arxiv.org/pdf/1801.07698v1.pdf
class ArcFace(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 32.0,
        m: float = 0.8,
        easy_margin: bool = False,
    ):
        """
        Initialise the ArcFace layer.

        :param in_features: The number of input features.
        :param out_features: The number of output features.
        :param s: The norm of the input feature.
        :param m: The margin.
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    # noinspection PyTypeChecker
    def forward(self, x, label):
        # cos(theta) and phi(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))

        if label is None:
            return cosine

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        output = (label * phi) + ((1.0 - label) * cosine)
        output *= self.s
        return output


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
    ):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(
            inplanes,
            eps=1e-05,
        )
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(
            planes,
            eps=1e-05,
        )
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(
            planes,
            eps=1e-05,
        )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class IResNetHead(RecognitionArchitecture):
    fc_scale = 7 * 7

    @staticmethod
    def construct(
        dataset: FaceDatasets,
        size: FaceDatasetSize,
        weights_directory=None,
        training: bool = False,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> "IResNetHead":
        # Embedder is the model
        # backbone, head = embedder_name.split('_')
        # weights_path = embedders_dict[backbone]['heads'][head]['weights_path']

        wd = str(weights_directory)
        sd = torch.load(wd + "/backbone.pth", map_location=device)
        # ALWAYS R100
        layers = [3, 13, 30, 3]
        embedder = (
            IResNetHead(IBasicBlock, layers=layers, classes=100).to(device).eval()
        )
        embedder.load_state_dict(sd, strict=False)

        embedder = embedder.to(device)

        if not training:
            embedder.eval()

        return embedder

    def __init__(
        self,
        block,
        layers,
        dropout=0,
        num_features=512,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        fp16=False,
        withHead=False,
        classes=0,
    ):
        super(IResNetHead, self).__init__()
        self.fp16 = fp16
        self.noBGR = True
        self.batched = True
        mean = [0.5] * 3
        std = [0.5 * 256 / 255] * 3
        self.normalizeProcess = transforms.Compose([transforms.Normalize(mean, std)])
        self.crop_size = 112
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.bn2 = nn.BatchNorm2d(
            512 * block.expansion,
            eps=1e-05,
        )
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        self.head = ArcFace(512, classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(
                    planes * block.expansion,
                    eps=1e-05,
                ),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x, label=None):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.normalizeProcess(x)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.bn2(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
        x = self.fc(x.float() if self.fp16 else x)
        if x.size()[0] == 1:
            # single batch
            ftr = x
        else:
            ftr = self.features(x)

        x = self.head(ftr, label)
        return x

    def returnEmbedding(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.normalizeProcess(x)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.bn2(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
        x = self.fc(x.float() if self.fp16 else x)
        x = l2_norm(self.features(x))
        return x

    def save_transfer_data(
        self, save_directory, dataset: FaceDatasets, size: FaceDatasetSize
    ) -> None:
        save_directory.mkdir(exist_ok=True)
        torch.save(
            self.state_dict(),
            save_directory / f"backbone.pth",
        )

    def load_transfer_data(
        self,
        weights_directory,
        dataset: FaceDatasets,
        size: FaceDatasetSize,
        device: torch.device,
    ) -> None:
        self.load_state_dict(
            torch.load(weights_directory / "backbone.pth", map_location=device)
        )
