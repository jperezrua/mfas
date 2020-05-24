#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: juanma
"""

# %%
import torch
import torch.nn as nn
import torch.optim as op
import os

import models.auxiliary.scheduler as sc
import models.auxiliary.aux_models as aux
import models.central.ntu as ntu
import models.search.train_searchable.ntu as tr


# %%


def train_sampled_models(sampled_configurations, searchable_type, dataloaders,
                         args, device,
                         return_model=[], premodels=[], preaccuracies=[],
                         train_only_central_params=True,
                         state_dict=dict()):
    use_weightsharing = args.weightsharing
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'dev']}
    num_batches_per_epoch = dataset_sizes['train'] / args.batchsize
    criterion = torch.nn.CrossEntropyLoss()

    real_accuracies = []

    if return_model:
        models = []

    for idx, configuration in enumerate(sampled_configurations):

        if not return_model or idx in return_model:

            # model to train
            if not premodels:
                rmode = searchable_type(args, configuration)
                # loading pretrained weights
                skemodel_filename = os.path.join(args.checkpointdir, args.ske_cp)
                rgbmodel_filename = os.path.join(args.checkpointdir, args.rgb_cp)
                rmode.skenet.load_state_dict(torch.load(skemodel_filename))
                rmode.rgbnet.load_state_dict(torch.load(rgbmodel_filename))
            else:
                rmode = searchable_type(args, configuration)

                if args.use_dataparallel:
                    rmode.load_state_dict(premodels[idx].module.state_dict())
                else:
                    rmode.load_state_dict(premodels[idx].state_dict())

                    # parameters to update during training
            if train_only_central_params:
                params = rmode.central_params()
            else:
                params = rmode.parameters()

            # optimizer and scheduler
            optimizer = op.Adam(params, lr=args.eta_max, weight_decay=1e-4)
            scheduler = sc.LRCosineAnnealingScheduler(args.eta_max, args.eta_min, args.Ti, args.Tm,
                                                      num_batches_per_epoch)

            # hardware tuning
            if torch.cuda.device_count() > 1 and args.use_dataparallel:
                rmode = torch.nn.DataParallel(rmode)
            rmode.to(device)

            if use_weightsharing:
                set_central_states(rmode, state_dict, args.use_dataparallel)

            if args.verbose:
                print('Now training: ')
                print(configuration)

            if not preaccuracies:
                best_model_acc = tr.train_ntu_track_acc(rmode, criterion, optimizer, scheduler, dataloaders,
                                                        dataset_sizes,
                                                        device=device, num_epochs=args.epochs, verbose=args.verbose)
            else:
                best_model_acc = tr.train_ntu_track_acc(rmode, criterion, optimizer, scheduler, dataloaders,
                                                        dataset_sizes,
                                                        device=device, num_epochs=args.epochs, verbose=args.verbose,
                                                        init_f1=preaccuracies[idx])

            if use_weightsharing:
                state_dict = get_central_states(rmode, state_dict, args.use_dataparallel)

            real_accuracies.append(best_model_acc)

            if return_model:
                models.append(rmode)

    if return_model:
        return real_accuracies, models
    else:
        return real_accuracies


def get_possible_layer_configurations(progression_index):
    def get_max_labels():
        # return (8, 5, 2)
        return (4, 4, 2)

    list_conf = []
    max_labels = get_max_labels()

    for ti in range(max_labels[0]):
        for vi in range(max_labels[1]):
            for ni in range(max_labels[2]):
                conf = [ti, vi, ni]
                list_conf.append(conf)

    return list_conf


# %%
def get_central_states(model, state_dict, using_dataparallel=False):
    if using_dataparallel:
        fusion_layers = model.module.fusion_layers
        configuration = model.module.conf
    else:
        fusion_layers = model.fusion_layers
        configuration = model.conf

    for idx_layer, layer in enumerate(fusion_layers):
        name = str(idx_layer)
        name += '.L_' + str(layer[0].in_features) + '_' + str(layer[0].out_features)

        if configuration[idx_layer][2] == 0:
            name += '.A_relu'
        if configuration[idx_layer][2] == 1:
            name += '.A_sigmoid'
        if configuration[idx_layer][2] == 2:
            name += '.A_lrelu'

        if name in state_dict:
            print('Updating shared weight with ID: {}'.format(name))
        else:
            print('Creating shared weight with ID: {}'.format(name))

        state_dict[name] = layer.state_dict()

    return state_dict


def set_central_states(model, state_dict, using_dataparallel=False):
    if using_dataparallel:
        fusion_layers = model.module.fusion_layers
        configuration = model.module.conf
    else:
        fusion_layers = model.fusion_layers
        configuration = model.conf

    for idx_layer, layer in enumerate(fusion_layers):

        name = str(idx_layer)
        name += '.L_' + str(layer[0].in_features) + '_' + str(layer[0].out_features)

        if configuration[idx_layer][2] == 0:
            name += '.A_relu'
        if configuration[idx_layer][2] == 1:
            name += '.A_sigmoid'
        if configuration[idx_layer][2] == 2:
            name += '.A_lrelu'

        if name in state_dict:
            layer.load_state_dict(state_dict[name])
            print('Loaded shared weight with ID: {}'.format(name))


# %%
class Searchable_Skeleton_Image_Net(nn.Module):
    def __init__(self, args, conf):
        super(Searchable_Skeleton_Image_Net, self).__init__()

        """
            conf: description of network numpy array
            
            1 row per fusion
            1st column [0] = ske feature
            2nd column [1] = rgb feature
            3rd column [2] = nonlinearity
        """

        self.conf = conf
        self.args = args

        self.rgbnet = ntu.Visual(args)
        self.skenet = ntu.Skeleton(args)

        self.alphas = self._create_alphas()
        self.gp_v, self.gp_s = self._create_global_poolings()
        self.fusion_layers = self._create_fc_layers()
        self.central_classifier = nn.Linear(self.args.inner_representation_size, args.num_outputs)

        for m in self.modules():
            if isinstance(m, aux.AlphaScalarMultiplication):
                nn.init.normal_(m.alpha_x, 0.0, 0.1)

    def forward(self, tensor_tuple):

        skeleton, image = tensor_tuple[1], tensor_tuple[0]

        # apply net on input image        
        visual_features = self.rgbnet(image)
        visual_classifier = visual_features[-1]
        visual_features = visual_features[-5:-1]

        # apply net on input skeleton
        skel_features, skel_classifier = self.skenet(skeleton)
        skel_features = skel_features[-4:]

        # only keep the features we care about
        visual_features = [visual_features[idx] for idx in self.conf[:, 1]]
        skel_features = [skel_features[idx] for idx in self.conf[:, 0]]

        # apply global pooling and batchnorms on visual features
        visual_features = [self.gp_v[idx](feat) for idx, feat in enumerate(visual_features)]
        skel_features = [self.gp_s[idx](feat) for idx, feat in enumerate(skel_features)]

        # fuse features
        for fusion_idx, conf in enumerate(self.conf):
            ske_feat = skel_features[fusion_idx]
            vis_feat = visual_features[fusion_idx]

            if self.args.alphas:
                ske_feat, vis_feat = self.alphas[fusion_idx](ske_feat, vis_feat)

            if fusion_idx == 0:
                fused = torch.cat((ske_feat, vis_feat), 1)
                out = self.fusion_layers[fusion_idx](fused)
            else:
                fused = torch.cat((ske_feat, vis_feat, out), 1)
                out = self.fusion_layers[fusion_idx](fused)

        out = self.central_classifier(out)

        if not self.args.multitask:
            return out
        else:
            return out, visual_classifier, skel_classifier

    def central_params(self):
        central_parameters = [
            {'params': self.alphas.parameters()},
            {'params': self.fusion_layers.parameters()},
            {'params': self.central_classifier.parameters()}
        ]

        return central_parameters

    def _create_fc_layers(self):
        fusion_layers = []
        for i, conf in enumerate(self.conf):
            in_size = self.alphas[i].size_alpha_x + self.alphas[i].size_alpha_y

            if i > 0:
                in_size += self.args.inner_representation_size

            out_size = self.args.inner_representation_size
            if conf[2] == 0:
                nl = nn.ReLU()
            elif conf[2] == 1:
                nl = nn.Sigmoid()
            elif conf[2] == 2:
                nl = nn.LeakyReLU()

            if self.args.drpt > 1e-10 and self.args.batchnorm:
                op = nn.Sequential(nn.Linear(in_size, out_size), nl, nn.BatchNorm1d(out_size),
                                   nn.Dropout(self.args.drpt))

            if self.args.drpt > 1e-10 and not self.args.batchnorm:
                op = nn.Sequential(nn.Linear(in_size, out_size), nl, nn.Dropout(self.args.drpt))

            if self.args.drpt < 1e-10 and self.args.batchnorm:
                op = nn.Sequential(nn.Linear(in_size, out_size), nl, nn.BatchNorm1d(out_size))

            fusion_layers.append(op)

        return nn.ModuleList(fusion_layers)

    def _create_alphas(self):
        # sizes_ske = [64,25,32,128,128,256,256,512] #real sizes
        # sizes_ims = [256,512,1024,2048,2048] #real sizes
        sizes_ske = [128, 256, self.args.vid_len[1] * 32, 512]  # sizes last 4
        sizes_ims = [512, 1024, 2048, 2048]  # sizes last 4 (-1)
        # todo: check if I actually need the sizes in AlphaScalarMultiplication (maybe point-wise multiplication in torch can infer size)

        alphas = [aux.AlphaScalarMultiplication(sizes_ske[conf[0]], sizes_ims[conf[1]]) for conf in self.conf]
        return nn.ModuleList(alphas)

    def _create_global_poolings(self):
        gp_list_v = [aux.GlobalPooling2D() for i in range(len(self.conf))]
        gp_list_s = [aux.GlobalPooling2D() for i in range(len(self.conf))]
        return nn.ModuleList(gp_list_v), nn.ModuleList(gp_list_s)
