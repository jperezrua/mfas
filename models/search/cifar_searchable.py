#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: juanma
"""

# %%
import torch
import torch.nn as nn
import models.auxiliary.aux_models as aux
import numpy as np
import models.auxiliary.scheduler as sc
import models.search.train_searchable.cifar as tr
import torch.optim as op


# %%


def train_sampled_models(sampled_configurations, searchable_type, dataloaders,
                         args, device, state_dict=dict()):
    use_weightsharing = args.weightsharing
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'dev']}
    num_batches_per_epoch = dataset_sizes['train'] / args.batchsize
    criterion = torch.nn.CrossEntropyLoss()

    real_accuracies = []

    for idx, configuration in enumerate(sampled_configurations):

        # model to train
        rmode = searchable_type(args, configuration)

        # parameters to update during training
        params = rmode.parameters()

        # optimizer and scheduler
        optimizer = op.Adam(params, lr=args.eta_max, weight_decay=1e-4)
        scheduler = sc.LRCosineAnnealingScheduler(args.eta_max, args.eta_min, args.Ti, args.Tm, num_batches_per_epoch)

        # hardware tuning
        if torch.cuda.device_count() > 1 and args.use_dataparallel:
            rmode = torch.nn.DataParallel(rmode)
        rmode.to(device)

        if use_weightsharing:
            set_states(rmode, state_dict, args.use_dataparallel)

        if args.verbose:
            print('Now training: ')
            print(configuration)

        best_model_acc = tr.train_cifar_track_acc(rmode, criterion, optimizer, scheduler, dataloaders, dataset_sizes,
                                                  device=device, num_epochs=args.epochs, verbose=args.verbose)

        if use_weightsharing:
            state_dict = get_states(rmode, state_dict, args.use_dataparallel)

        real_accuracies.append(best_model_acc)

    return real_accuracies


def get_possible_layer_configurations(progression_index):
    num_ops_per_block = 5

    label_list = []  # (seq_len=1, batch=num_options, input_size=1+size_connections):

    for op1i in range(num_ops_per_block):
        for op2i in range(num_ops_per_block):
            for bi1 in range(-2, progression_index):
                for bi2 in range(-2, progression_index):
                    if op1i == op2i:
                        continue

                    label_list.append([op1i, op2i, bi1, bi2])

    return label_list


# %%
def get_states(model, state_dict, using_dataparallel=False):
    # deliver ALL possible states (even unused ones)
    state_dict = {}

    print('getting states')
    if not using_dataparallel:
        for index_cell, cell in enumerate(model.cell_array):
            for index_block, block in enumerate(cell.blocks):
                name1 = 'op1.{0}.block{1}.cell{2}'.format(block.op1_type, index_block, index_cell)
                name2 = 'op2.{0}.block{1}.cell{2}'.format(block.op2_type, index_block, index_cell)

                state_dict[name1] = block.op1.state_dict()
                state_dict[name2] = block.op2.state_dict()

        state_dict['input_conv'] = model.input_conv.state_dict()
        state_dict['classifier'] = model.classifier.state_dict()
        state_dict['aux_classifier'] = model.aux_classifier.state_dict()

    else:
        for index_cell, cell in enumerate(model.module.cell_array):
            for index_block, block in enumerate(cell.blocks):
                name1 = 'op1.{0}.block{1}.cell{2}'.format(block.op1_type, index_block, index_cell)
                name2 = 'op2.{0}.block{1}.cell{2}'.format(block.op2_type, index_block, index_cell)

                state_dict[name1] = block.op1.state_dict()
                state_dict[name2] = block.op2.state_dict()

        state_dict['input_conv'] = model.module.input_conv.state_dict()
        state_dict['classifier'] = model.module.classifier.state_dict()
        state_dict['aux_classifier'] = model.module.aux_classifier.state_dict()

    return state_dict


def set_states(model, state_dict, using_dataparallel=False):
    # loop over current model and update the states
    if not using_dataparallel:
        for index_cell, cell in enumerate(model.cell_array):
            for index_block, block in enumerate(cell.blocks):
                key1 = 'op1.{0}.block{1}.cell{2}'.format(block.op1_type, index_block, index_cell)
                key2 = 'op2.{0}.block{1}.cell{2}'.format(block.op2_type, index_block, index_cell)

                if key1 in state_dict:
                    block.op1.load_state_dict(state_dict[key1])
                if key2 in state_dict:
                    block.op2.load_state_dict(state_dict[key2])

        if 'classifier' in state_dict:
            model.classifier.load_state_dict(state_dict['classifier'])

        if 'aux_classifier' in state_dict:
            model.aux_classifier.load_state_dict(state_dict['aux_classifier'])

        if 'input_conv' in state_dict:
            model.input_conv.load_state_dict(state_dict['input_conv'])
    else:
        for index_cell, cell in enumerate(model.module.cell_array):
            for index_block, block in enumerate(cell.blocks):
                key1 = 'op1.{0}.block{1}.cell{2}'.format(block.op1_type, index_block, index_cell)
                key2 = 'op2.{0}.block{1}.cell{2}'.format(block.op2_type, index_block, index_cell)

                if key1 in state_dict:
                    block.op1.load_state_dict(state_dict[key1])
                if key2 in state_dict:
                    block.op2.load_state_dict(state_dict[key2])

        if 'classifier' in state_dict:
            model.module.classifier.load_state_dict(state_dict['classifier'])

        if 'aux_classifier' in state_dict:
            model.module.aux_classifier.load_state_dict(state_dict['aux_classifier'])

        if 'input_conv' in state_dict:
            model.module.input_conv.load_state_dict(state_dict['input_conv'])

        # %%


class Searchable_MicroCNN(nn.Module):

    def __init__(self, args, configuration,
                 operation_labels=['I', '3x3 conv', '5x5 conv', '3x3 depthconv', '5x5 depthconv', '7x7 depthconv',
                                   '3x3 maxpool', '3x3 avgpool'],
                 fixed=False):

        super(Searchable_MicroCNN, self).__init__()

        self.args = args
        self.fixed = fixed  # fixed=True is used for training of a found architecture

        if isinstance(configuration, np.ndarray):
            _configuration = configuration
        else:
            _configuration = np.asarray(configuration, np.int)

        """
         conf = [[op1 op2 con1 con2], %b0
                 [op1 op2 con1 con2], %b1
                 [op1 op2 con1 con2], %b2
                    ...,
                 [op1 op2 con1 con2]] %bB
            
         conX \in [-2,b# - 1]
         opX  \in [0, len(operation_labels)]
        """
        self._configuration_indexes = _configuration[:, 0:2]
        self._connections = _configuration[:, 2:]

        self._spatial_dim = self.args.img_size

        self._operation_labels = operation_labels
        self._num_ops = len(self._operation_labels)

        self._network_shape = args.net_str

        self.input_conv = nn.Sequential(
            nn.Conv2d(3, int(self.args.planes), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(self.args.planes), eps=1e-3)
        )

        self.cell_array, self.pooled_layers = self._create_cell_array()

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(int(self.args.planes), self.args.num_outputs)

        self.dropout_cla = nn.Dropout(p=self.args.drop_prob)

        # if self.args.aux:
        self.aux_head = aux.AuxiliaryHead(self.args.num_outputs, self.args.planes)

        # initialization of weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):

        cell_outputs = [self.input_conv(x)]

        pool_layer_id = 0
        for index_cell, cell in enumerate(self.cell_array):

            if index_cell == 0:
                cell_out = cell(cell_outputs[0], cell_outputs[0])
                cell_outputs.append(cell_out)
            else:
                cell_out = cell(cell_outputs[-2], cell_outputs[-1])
                cell_outputs.append(cell_out)

            if self._network_shape[index_cell] == 2:
                ## this is a pool cell
                for idx, cell_out in enumerate(cell_outputs):
                    # downsample previous cell outputs
                    cell_outputs[idx] = self.pooled_layers[pool_layer_id](cell_out)
                    pool_layer_id += 1

        out = self._output_(cell_outputs[-1], False)
        iout = self._output_(cell_outputs[int(index_cell * 0.666)], True)

        return out, iout

    def _output_(self, x, intermediate=False):

        if not intermediate:
            out = self.global_avg_pool(x)
            out = out.view(out.size(0), -1)
            out = self.dropout_cla(out)
            out = self.classifier(out)
        else:
            out = self.aux_head(x)

        return out

    def _create_cell_array(self):

        cell_array = nn.ModuleList()
        pooled_layers = nn.ModuleList([])

        for layer_red in self._network_shape:

            if self.fixed:
                cell = aux.FixedCell(self._operation_labels,
                                     self._configuration_indexes,
                                     self._connections,
                                     self.args)
            else:
                cell = aux.Cell(self._operation_labels,
                                self._configuration_indexes,
                                self._connections,
                                self.args)
            cell_array.append(cell)

            if layer_red == 2:
                for i in range(len(cell_array) + 1):
                    if self.fixed:
                        pooled_layers.append(aux.FactorizedReduction(self.args.planes, self.args.planes * 2))
                    else:
                        pooled_layers.append(aux.FactorizedReduction(self.args.planes, self.args.planes))
                if self.fixed:
                    self.args.planes *= 2

        return cell_array, pooled_layers
