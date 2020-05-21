#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: juanma
"""

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.auxiliary.inflated_resnet as resnet
import models.utils as utils


# %%
class Visual(nn.Module):
    def __init__(self, args):
        super(Visual, self).__init__()
        self.cnn = resnet.inflated_resnet()
        # self.avgpool_1x7x7 = nn.AvgPool3d((1, 7, 7))
        self.avgpool_Tx7x7 = nn.AvgPool3d((args.vid_len[0], 7, 7))
        self.D = 2048
        self.classifier = nn.Linear(self.D, args.num_outputs)

    def temporal_pooling(self, x):
        B, D, T, W, H = x.size()
        if self.D == D:
            final_representation = self.avgpool_Tx7x7(x)
            final_representation = final_representation.view(B, self.D)
            return final_representation
        else:
            print("Temporal pooling is not possible due to invalid channels dimensions:", self.D, D)

    def forward(self, x):
        # Changing temporal and channel dim to fit the inflated resnet input requirements
        B, T, W, H, C = x.size()
        x = x.view(B, 1, T, W, H, C)
        x = x.transpose(1, -1)
        x = x.view(B, C, T, W, H)
        x = x.contiguous()

        # Inflated ResNet
        out_1, out_2, out_3, out_4 = self.cnn.get_feature_maps(x)

        # Temporal pooling
        out_5 = self.temporal_pooling(out_4)
        out_6 = self.classifier(out_5)

        return out_1, out_2, out_3, out_4, out_5, out_6


class Skeleton(
    nn.Module):  # https://arxiv.org/pdf/1804.06055.pdf  https://github.com/huguyuehuhu/HCN-pytorch/blob/master/model/HCN.py
    def __init__(self, args):
        super(Skeleton, self).__init__()
        in_channel = 3
        num_joint = 25
        num_person = 2
        out_channel = 64
        window_size = args.vid_len[1]
        drpt = args.drpt
        self.num_person = num_person
        num_classes = args.num_outputs
        # position
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=window_size, kernel_size=(3, 1), stride=1,
                               padding=(1, 0))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=num_joint, out_channels=out_channel // 2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=drpt),
            nn.MaxPool2d(2))
        # motion
        self.conv1m = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        self.conv2m = nn.Conv2d(in_channels=out_channel, out_channels=window_size, kernel_size=(3, 1), stride=1,
                                padding=(1, 0))

        self.conv3m = nn.Sequential(
            nn.Conv2d(in_channels=num_joint, out_channels=out_channel // 2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2))
        self.conv4m = nn.Sequential(
            nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=drpt),
            nn.MaxPool2d(2))

        # concatenate motion & position
        if window_size == 8:
            self.conv5 = nn.Sequential(
                nn.Conv2d(in_channels=out_channel * 2, out_channels=out_channel * 2, kernel_size=3, stride=1,
                          padding=1),
                nn.ReLU(),
                nn.Dropout2d(p=drpt)
            )
        else:
            self.conv5 = nn.Sequential(
                nn.Conv2d(in_channels=out_channel * 2, out_channels=out_channel * 2, kernel_size=3, stride=1,
                          padding=1),
                nn.ReLU(),
                nn.Dropout2d(p=drpt),
                nn.MaxPool2d(2)
            )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel * 2, out_channels=out_channel * 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=drpt),
            nn.MaxPool2d(2)
        )

        lin = (out_channel * 4) * max((window_size // 16) * (window_size // 16), 1)
        self.fc7 = nn.Sequential(
            nn.Linear(lin, 256 * 2),  # 4*4 for window=64; 8*8 for window=128
            nn.ReLU(),
            nn.Dropout2d(p=drpt))
        self.fc8 = nn.Linear(256 * 2, num_classes)

        utils.initial_model_weight(layers=list(self.children()))

    def forward(self, x):
        N, C, T, V, M = x.size()  # N0, C1, T2, V3, M4
        motion = x[:, :, 1::, :, :] - x[:, :, 0:-1, :, :]
        motion = motion.permute(0, 1, 4, 2, 3).contiguous().view(N, C * M, T - 1, V)
        motion = F.interpolate(motion, size=(T, V), mode='bilinear', align_corners=False).contiguous().view(N, C, M, T,
                                                                                                            V).permute(
            0, 1, 3, 4, 2)

        logits = []
        hidden = []
        for i in range(self.num_person):
            # position
            # N0,C1,T2,V3 point-level
            out1 = self.conv1(x[:, :, :, :, i])
            out2 = self.conv2(out1)
            # N0,V1,T2,C3, global level
            out2 = out2.permute(0, 3, 2, 1).contiguous()
            out3 = self.conv3(out2)
            out_p = self.conv4(out3)

            # motion
            # N0,T1,V2,C3 point-level
            out1m = self.conv1m(motion[:, :, :, :, i])
            out2m = self.conv2m(out1m)
            # N0,V1,T2,C3, global level
            out2m = out2m.permute(0, 3, 2, 1).contiguous()
            out3m = self.conv3m(out2m)
            out_m = self.conv4m(out3m)

            # concat
            out4 = torch.cat((out_p, out_m), dim=1)
            out5 = self.conv5(out4)
            out6 = self.conv6(out5)

            hidden.append([out1, out2, out3, out4, out5, out6])
            logits.append(out6)

        # max out logits
        out7 = torch.max(logits[0], logits[1])
        out7 = out7.view(out7.size(0), -1)
        out8 = self.fc7(out7)
        outf = self.fc8(out8)

        # clean hidden representations
        new_hidden = []
        for h1, h2 in zip(hidden[0], hidden[1]):
            new_hidden.append(torch.max(h1, h2))
        new_hidden.append(out7)
        new_hidden.append(out8)

        t = outf
        assert not ((t != t).any())  # find out nan in tensor
        # assert not (t.abs().sum() == 0) # find out 0 tensor

        return new_hidden, outf


class LateFusion(nn.Module):
    def __init__(self, args):
        super(LateFusion, self).__init__()
        self.skeleton = Skeleton(args)
        self.visual = Visual(args)
        self.final_pred = nn.Linear(args.num_classes * 2, args.num_classes)

    def forward(self, input):
        frames, skeleton = input
        skeleton = self.skeleton(skeleton)
        skeleton = skeleton[-1]
        vis = self.visual(frames)
        vis = vis[-1]
        pred = self.final_pred(torch.cat([skeleton, vis], dim=-1))
        return pred


class GMU(nn.Module):
    def __init__(self, args):
        super(GMU, self).__init__()
        self.skeleton = Skeleton(args)
        self.visual = Visual(args)

        self.skel_redu = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout2d(p=args.drpt))
        self.vis_redu = nn.Sequential(nn.Linear(2048, 128), nn.ReLU(), nn.Dropout2d(p=args.drpt))
        self.ponderation = nn.Sequential(nn.Linear(256 + 2048, 1), nn.Sigmoid())

        self.final_pred = nn.Linear(128, args.num_classes)

    def forward(self, input):
        frames, skeleton = input
        skeleton = self.skeleton(skeleton)
        skeleton = skeleton[0][-2]
        vis = self.visual(frames)
        vis = vis[-2]

        z = self.ponderation(torch.cat([vis, skeleton], 1))
        skeleton = self.skel_redu(skeleton)
        vis = self.vis_redu(vis)

        h = z * skeleton + (1.0 - z) * vis
        pred = self.final_pred(h)
        return pred


class CentralNet(nn.Module):
    def __init__(self, args):
        super(CentralNet, self).__init__()

        self.skeleton = Skeleton(args)
        self.visual = Visual(args)

        # Central
        central_list = [nn.Sequential(
            *[nn.Conv2d(conv_dim, 2 * conv_dim, kernel_size=4, stride=2, padding=1), torch.nn.BatchNorm2d(2 * conv_dim),
              nn.ReLU(args)]) for conv_dim in [512]]
        central_list.append(nn.Sequential(
            *[nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1), torch.nn.BatchNorm2d(2048), nn.ReLU(args),
              nn.AvgPool2d((7, 7))]))
        central_list.append(nn.Linear(2048, args.num_classes))
        self.central_conv = nn.ModuleList(central_list)

        self.alphas_a = nn.ParameterList([torch.nn.Parameter(torch.rand(1)) for i in range(4)])
        self.alphas_v = nn.ParameterList([torch.nn.Parameter(torch.rand(1)) for i in range(4)])
        self.alphas_c = nn.ParameterList([torch.nn.Parameter(torch.rand(1)) for i in range(4)])

    def _lateral_padding(self, inputs, pad=0):
        sz = inputs.size()
        if len(sz) > 2:
            padding = torch.zeros(sz[0], pad, sz[2], sz[3]).cuda()
        elif len(sz) == 2:
            padding = torch.zeros(sz[0], pad).cuda()
        padded = torch.cat((inputs, padding), 1)

        return padded

    def _fuse_features(self, m1, m2, central, alpha1, alpha2, alpha_central):
        m1sz = m1.size()
        m2sz = m2.size()
        centralsz = central.size()

        # In the case one of the modality (often visual) is splitted into frames, first perform averaging across frames before fusing.
        batch_size = m1sz[0]
        if len(m1.size()) > 4:
            m1 = torch.mean(m1, 2).view(batch_size, m1sz[1], m1sz[3], m1sz[4])
        if len(m2.size()) > 4:
            m2 = torch.mean(m2, 2).view(batch_size, m2sz[1], m2sz[3], m2sz[4])
        if len(central.size()) > 4:
            central = torch.mean(central, 2).view(batch_size, centralsz[1], centralsz[3], centralsz[4])
        if central.size(-1) == 1:
            central = central.view(batch_size, -1)
        m2 = self._lateral_padding(m2, pad=m1.size(1) - m2.size(1))  # zero-padding the channel
        return central * alpha_central.expand_as(central) + m1 * alpha1.expand_as(m1) + m2 * alpha2.expand_as(m2)

    def forward(self, input):
        frames, skeleton = input
        self.visual.load_state_dict(torch.load("checkpoints/best_rgb"))
        for p in self.visual.parameters():
            p.requires_grad = False
        self.skeleton.load_state_dict(torch.load("checkpoints/best_skeleton"))
        for p in self.skeleton.parameters():
            p.requires_grad = False
        out_1, out_2, out_3, out_4, out_5, visual_pred = self.visual(frames)
        hidden, skel_pred = self.skeleton(skeleton)
        central_rep = torch.zeros_like(out_2).cuda()
        for c_conv, mv, ma, aa, av, ac in zip(self.central_conv, [out_2, out_3, out_5, visual_pred],
                                              [hidden[1], hidden[2], hidden[-1], skel_pred], self.alphas_a,
                                              self.alphas_v, self.alphas_c):
            central_rep = self._fuse_features(mv, ma, central_rep, F.sigmoid(aa), F.sigmoid(av), F.sigmoid(ac))
            central_rep = c_conv(central_rep)

        return central_rep
