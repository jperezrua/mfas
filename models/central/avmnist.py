#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: juanma
"""

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as A
import models.auxiliary.aux_models as aux


# %%

class GP_LeNet(nn.Module):
    def __init__(self, args, in_channels):
        super(GP_LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, args.channels, kernel_size=5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(int(args.channels))
        self.gp1 = aux.GlobalPooling2D()

        self.conv2 = nn.Conv2d(args.channels, 2 * args.channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(int(2 * args.channels))
        self.gp2 = aux.GlobalPooling2D()

        self.conv3 = nn.Conv2d(2 * args.channels, 4 * args.channels, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(int(4 * args.channels))
        self.gp3 = aux.GlobalPooling2D()

        self.classifier = nn.Sequential(
            nn.Linear(int(4 * args.channels), args.num_outputs)
        )

        # initialization of weights
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out1, 2)
        gp1 = self.gp1(out1)

        out2 = F.relu(self.bn2(self.conv2(out)))
        out = F.max_pool2d(out2, 2)
        gp2 = self.gp2(out2)

        out3 = F.relu(self.bn3(self.conv3(out)))
        out = F.max_pool2d(out3, 2)
        gp3 = self.gp3(out3)

        out = self.classifier(gp3)

        return out, gp1, gp2, gp3


class GP_LeNet_Deeper(nn.Module):
    def __init__(self, args, in_channels):
        super(GP_LeNet_Deeper, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, args.channels, kernel_size=5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(int(args.channels))
        self.gp1 = aux.GlobalPooling2D()

        self.conv2 = nn.Conv2d(args.channels, 2 * args.channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(int(2 * args.channels))
        self.gp2 = aux.GlobalPooling2D()

        self.conv3 = nn.Conv2d(2 * args.channels, 4 * args.channels, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(int(4 * args.channels))
        self.gp3 = aux.GlobalPooling2D()

        self.conv4 = nn.Conv2d(4 * args.channels, 8 * args.channels, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(int(8 * args.channels))
        self.gp4 = aux.GlobalPooling2D()

        self.conv5 = nn.Conv2d(8 * args.channels, 16 * args.channels, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(int(16 * args.channels))
        self.gp5 = aux.GlobalPooling2D()

        self.classifier = nn.Sequential(
            nn.Linear(int(16 * args.channels), args.num_outputs)
        )

        # initialization of weights
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out1, 2)
        gp1 = self.gp1(out)

        out2 = F.relu(self.bn2(self.conv2(out)))
        out = F.max_pool2d(out2, 2)
        gp2 = self.gp2(out2)

        out3 = F.relu(self.bn3(self.conv3(out)))
        out = F.max_pool2d(out3, 2)
        gp3 = self.gp3(out3)

        out4 = F.relu(self.bn4(self.conv4(out)))
        out = F.max_pool2d(out4, 2)
        gp4 = self.gp4(out4)

        out5 = F.relu(self.bn5(self.conv5(out)))
        out = F.max_pool2d(out5, 2)
        gp5 = self.gp5(out5)

        out = self.classifier(gp5)

        return out, gp1, gp2, gp3, gp4, gp5


class SimpleAVNet(nn.Module):
    def __init__(self, args, audio_channels, image_channels):
        super(SimpleAVNet, self).__init__()

        self.audio_net = GP_LeNet(args, audio_channels)
        self.image_net = GP_LeNet(args, image_channels)

        self.classifier = nn.Linear(int(2 * 4 * args.channels), args.num_outputs)

    def forward(self, audio, image):
        audio_out, audio_gp1, audio_gp2, audio_gp3 = self.audio_net(audio)
        image_out, image_gp1, image_gp2, image_gp3 = self.image_net(image)

        multimodal_feat = torch.cat((audio_gp3, image_gp3), 1)
        out = self.classifier(multimodal_feat)

        return out


class SimpleAVNet_Deeper(nn.Module):
    def __init__(self, args, audio_channels, image_channels):
        super(SimpleAVNet_Deeper, self).__init__()

        self.audio_net = GP_LeNet_Deeper(args, audio_channels)
        self.image_net = GP_LeNet(args, image_channels)

        self.classifier = nn.Linear(int(20 * args.channels), args.num_outputs)

    def forward(self, audio, image):
        audio_out, audio_gp1, audio_gp2, audio_gp3, audio_gp4, audio_gp5 = self.audio_net(audio)
        image_out, image_gp1, image_gp2, image_gp3 = self.image_net(image)

        multimodal_feat = torch.cat((audio_gp5, image_gp3), 1)
        out = self.classifier(multimodal_feat)

        return out


class SimpleAV_CentralNet(nn.Module):
    def __init__(self, args, audio_channels, image_channels):
        super(SimpleAV_CentralNet, self).__init__()

        self.args = args
        self.audio_net = GP_LeNet_Deeper(args, audio_channels)
        self.image_net = GP_LeNet(args, image_channels)

        self.alpha1_feat1 = nn.Parameter(torch.rand(1))
        self.alpha2_feat1 = nn.Parameter(torch.rand(1))
        self.alpha3_feat1 = nn.Parameter(torch.rand(1))

        self.alpha1_feat2 = nn.Parameter(torch.rand(1))
        self.alpha2_feat2 = nn.Parameter(torch.rand(1))
        self.alpha3_feat2 = nn.Parameter(torch.rand(1))

        self.alpha_conv1 = nn.Parameter(torch.rand(1))
        self.alpha_conv2 = nn.Parameter(torch.rand(1))

        self.central_conv1 = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.central_conv2 = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.central_conv3 = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)

        if self.args.fusingmix == '11,32,53' or self.args.fusingmix == '31,42,53':
            nodes = 384

        if self.args.fusingmix == '11,22,33':
            nodes = 96

        if self.args.fusetype == 'cat':
            nodes *= 2

        self.central_classifier = nn.Linear(nodes, args.num_outputs)

    def central_params(self):

        central_parameters = [
            {'params': self.central_conv1.parameters()},
            {'params': self.central_conv2.parameters()},
            {'params': self.central_conv3.parameters()},
            {'params': self.alpha1_feat1},
            {'params': self.alpha2_feat1},
            {'params': self.alpha3_feat1},
            {'params': self.alpha1_feat2},
            {'params': self.alpha2_feat2},
            {'params': self.alpha3_feat2},
            {'params': self.alpha_conv1},
            {'params': self.alpha_conv2},
            {'params': self.central_classifier.parameters()}]

        return central_parameters

    def forward(self, audio, image):

        audio_out, audio_gp1, audio_gp2, audio_gp3, audio_gp4, audio_gp5 = self.audio_net(audio)
        image_out, image_gp1, image_gp2, image_gp3 = self.image_net(image)

        if self.args.fusingmix == '11,32,53':
            fuse1 = self._fuse_features(audio_gp1, image_gp1, self.alpha1_feat1, self.alpha1_feat2, self.args.fusetype)
            fuse2 = self._fuse_features(audio_gp3, image_gp2, self.alpha2_feat1, self.alpha2_feat2, self.args.fusetype)
            fuse3 = self._fuse_features(audio_gp5, image_gp3, self.alpha3_feat1, self.alpha3_feat2, self.args.fusetype)
        elif self.args.fusingmix == '11,22,33':
            fuse1 = self._fuse_features(audio_gp1, image_gp1, self.alpha1_feat1, self.alpha1_feat2, self.args.fusetype)
            fuse2 = self._fuse_features(audio_gp2, image_gp2, self.alpha2_feat1, self.alpha2_feat2, self.args.fusetype)
            fuse3 = self._fuse_features(audio_gp3, image_gp3, self.alpha3_feat1, self.alpha3_feat2, self.args.fusetype)
        elif self.args.fusingmix == '31,42,53':
            fuse1 = self._fuse_features(audio_gp3, image_gp1, self.alpha1_feat1, self.alpha1_feat2, self.args.fusetype)
            fuse2 = self._fuse_features(audio_gp4, image_gp2, self.alpha2_feat1, self.alpha2_feat2, self.args.fusetype)
            fuse3 = self._fuse_features(audio_gp5, image_gp3, self.alpha3_feat1, self.alpha3_feat2, self.args.fusetype)
        else:
            raise ValueError(
                'self.args.fusingmix {} fusion combinantion is not implemented'.format(self.args.fusingmix))

        one = A.Variable(torch.ones(1))  # ugly hack to improve
        if image.is_cuda:
            one = one.cuda()

        cc1 = F.relu(self.central_conv1(fuse1.unsqueeze(1)))
        cc1 = self._fuse_features(cc1[:, 0, :], fuse2, self.alpha_conv1, one, 'wsum')

        cc2 = F.relu(self.central_conv2(cc1.unsqueeze(1)))
        cc2 = self._fuse_features(cc2[:, 0, :], fuse3, self.alpha_conv2, one, 'wsum')

        cc3 = F.relu(self.central_conv3(cc2.unsqueeze(1)))
        # print(cc3.size())

        fusion_out = self.central_classifier(cc3[:, 0, :])

        return audio_out, image_out, fusion_out

    def _fuse_features(self, f1, f2, a1, a2, fusetype):

        f1sz = f1.size()
        f2sz = f2.size()

        dif = f1sz[1] - f2sz[1]

        if fusetype == 'cat':
            if dif > 0:
                fuse = torch.cat((f1, self._lateral_padding(f2, dif)), 1)
            elif dif < 0:
                fuse = torch.cat((self._lateral_padding(f1, -dif), f2), 1)
            else:
                fuse = torch.cat((f1, f2), 1)

        elif fusetype == 'wsum':
            if dif > 0:
                fuse = f1 * a1.expand_as(f1) + self._lateral_padding(f2, dif) * a2.expand_as(f1)
            elif dif < 0:
                fuse = self._lateral_padding(f1, -dif) * a1.expand_as(f2) + f2 * a2.expand_as(f2)
            else:
                fuse = f1 * a1.expand_as(f1) + f2 * a1.expand_as(f2)

        return fuse

    def _lateral_padding(self, inputs, pad=0):
        sz = inputs.size()
        padding = A.Variable(torch.zeros(sz[0], pad), requires_grad=False)
        if inputs.is_cuda:
            padding = padding.cuda()

        padded = torch.cat((inputs, padding), 1)
        return padded
