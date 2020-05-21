import torch.nn as nn
import math
import torch

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

K_1st_CONV = 3


class ResNet(nn.Module):
    def __init__(self, list_block, layers,
                 **kwargs):
        self.inplanes = 64
        self.input_dim = 4
        super(ResNet, self).__init__()
        self._first_conv()
        self.relu = nn.ReLU(inplace=True)
        self.list_channels = [64, 128, 256, 512]
        self.layer1 = self._make_layer(list_block[0], self.list_channels[0], layers[0])
        self.layer2 = self._make_layer(list_block[1], self.list_channels[1], layers[1], stride=2)
        self.layer3 = self._make_layer(list_block[2], self.list_channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(list_block[3], self.list_channels[3], layers[3], stride=2)
        self.out_dim = 5

        # Init of the weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _first_conv(self):
        self.conv1 = nn.Conv2d(3, 64,
                               kernel_size=(7, 7),
                               stride=(2, 2),
                               padding=(3, 3),
                               bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.input_dim = 4

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None

        # Upgrade the stride is spatio-temporal kernel
        stride = (1, stride, stride)

        if stride != 1 or self.inplanes != planes * block.expansion:
            conv, batchnorm = nn.Conv3d, nn.BatchNorm3d

            downsample = nn.Sequential(
                conv(self.inplanes, planes * block.expansion,
                     kernel_size=1, stride=stride, bias=False, dilation=dilation),
                batchnorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def get_feature_maps(self, x):

        B, C, T, W, H = x.size()

        # 5D -> 4D if 2D conv at the beginning
        x = transform_input(x, self.input_dim, T=T)

        # 1st conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 1st residual block
        x = transform_input(x, self.layer1[0].input_dim, T=T)
        x = self.layer1(x)
        fm1 = x

        # 2nd residual block
        x = transform_input(x, self.layer2[0].input_dim, T=T)
        x = self.layer2(x)
        fm2 = x

        # 3rd residual block
        x = transform_input(x, self.layer3[0].input_dim, T=T)
        x = self.layer3(x)
        fm3 = x

        # 4th residual block
        x = transform_input(x, self.layer4[0].input_dim, T=T)
        x = self.layer4(x)
        final_fm = transform_input(x, self.out_dim, T=T)

        return fm1, fm2, fm3, final_fm


def transform_input(x, dim, T=12):
    diff = len(x.size()) - dim

    if diff > 0:
        B, C, T, W, H = x.size()
        x = x.transpose(1, 2)
        x = x.contiguous()
        x = x.view(-1, C, W, H)
    elif diff < 0:
        _, C, W, H = x.size()
        x = x.view(-1, T, C, W, H)
        x = x.transpose(1, 2)

    return x
