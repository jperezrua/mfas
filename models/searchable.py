#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 12:06:40 2018

@author: juanma
"""

# general use
import torch
import torch.optim as op
import numpy as np
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Subset

# avmnist
import models.search.avmnist_searchable as avmnist
from datasets import avmnist as avmnist_data

# ntu
import models.search.ntu_searchable as ntu
from datasets import ntu as ntu_data

# cifar
import models.search.cifar_searchable as cifar

# surrogate related
import models.search.surrogate as surr

# gen tools
import models.search.tools as tools

# %%
"""
 Base class for NAS
"""


class ModelSearcher():
    def __init__(self, args):
        self.args = args

    def search(self):
        pass

    def _epnas(self, model_type, surrogate_dict, dataloaders, dataset_searchmethods, device):

        # surrogate
        surrogate = surrogate_dict['model']
        s_crite = surrogate_dict['criterion']
        s_data = surr.SurrogateDataloader()
        s_optim = op.Adam(surrogate.parameters(), lr=self.args.lr_surrogate)

        # search functions that are specific to the dataset
        train_sampled_models = dataset_searchmethods['train_sampled_fun']
        get_possible_layer_configurations = dataset_searchmethods['get_layer_confs']

        temperature = self.args.initial_temperature

        sampled_k_confs = []

        shared_weights = dict()

        # repeat process search_iterations times
        for si in range(self.args.search_iterations):

            if self.args.verbose:
                print(50 * "=")
                print("Search iteration {}/{} ".format(si, self.args.search_iterations))

            # for each fusion
            for progression_index in range(self.args.max_progression_levels):

                if self.args.verbose:
                    print(25 * "-")
                    print("Progressive step {}/{} ".format(progression_index, self.args.max_progression_levels))

                    # Step 1: unfold layer (fusion index)
                list_possible_layer_confs = get_possible_layer_configurations(progression_index)

                # Step 2: merge previous top with unfolded configurations
                all_configurations = tools.merge_unfolded_with_sampled(sampled_k_confs, list_possible_layer_confs,
                                                                       progression_index)

                # Step 3: obtain accuracies for all possible unfolded configurations
                # if first execution, just train all, if not, use surrogate to predict them
                if si + progression_index == 0:
                    all_accuracies = train_sampled_models(all_configurations, model_type, dataloaders, self.args,
                                                          device, state_dict=shared_weights)
                    tools.update_surrogate_dataloader(s_data, all_configurations, all_accuracies)
                    tools.train_surrogate(surrogate, s_data, s_optim, s_crite, self.args, device)

                    if self.args.verbose:
                        print("Trained architectures: ")
                        print(list(zip(all_configurations, all_accuracies)))
                else:
                    all_accuracies = tools.predict_accuracies_with_surrogate(all_configurations, surrogate, device)
                    if self.args.verbose:
                        print("Predicted accuracies: ")
                        print(list(zip(all_configurations, all_accuracies)))

                # Step 4: sample K architectures and train them. 
                # this should happen only if not first iteration because in that case, 
                # all confs were trained in step 3
                if si + progression_index == 0:
                    sampled_k_confs = tools.sample_k_configurations(all_configurations, all_accuracies,
                                                                    self.args.num_samples, temperature)

                    if self.args.verbose:
                        estimated_accuracies = tools.predict_accuracies_with_surrogate(all_configurations, surrogate,
                                                                                       device)
                        diff = np.abs(np.array(estimated_accuracies) - np.array(all_accuracies))
                        print("Error on accuracies = {}".format(diff))

                else:
                    sampled_k_confs = tools.sample_k_configurations(all_configurations, all_accuracies,
                                                                    self.args.num_samples, temperature)
                    sampled_k_accs = train_sampled_models(sampled_k_confs, model_type, dataloaders, self.args, device,
                                                          state_dict=shared_weights)

                    tools.update_surrogate_dataloader(s_data, sampled_k_confs, sampled_k_accs)
                    err = tools.train_surrogate(surrogate, s_data, s_optim, s_crite, self.args, device)

                    if self.args.verbose:
                        print("Trained architectures: ")
                        print(list(zip(sampled_k_confs, sampled_k_accs)))
                        print("with surrogate error: {}".format(err))

                # temperature decays at each step
                iteration = si * self.args.search_iterations + progression_index
                temperature = tools.compute_temperature(iteration, self.args)
                if self.args.verbose:
                    print("Temperature is being set to {}".format(temperature))

        return s_data

    def _randsearch(self, model_type, dataloaders, dataset_searchmethods, device):

        # surrogate (in here, we only use the dataloader as means to keep track of real accuracies during exploration)
        s_data = surr.SurrogateDataloader()

        # search functions that are specific to the dataset
        train_sampled_models = dataset_searchmethods['train_sampled_fun']
        get_possible_layer_configurations = dataset_searchmethods['get_layer_confs']

        sampled_k_confs = []

        shared_weights = dict()

        # repeat process search_iterations times
        for si in range(self.args.search_iterations * self.args.max_progression_levels):

            if self.args.verbose:
                print(50 * "=")
                print("Random Search iteration {}/{} ".format(si,
                                                              self.args.search_iterations * self.args.max_progression_levels))

            # Step 1: sample
            sampled_k_confs = tools.sample_k_configurations_directly(self.args.num_samples,
                                                                     self.args.max_progression_levels,
                                                                     get_possible_layer_configurations)
            sampled_k_accs = train_sampled_models(sampled_k_confs, model_type, dataloaders, self.args, device,
                                                  state_dict=shared_weights)

            # Step 2: keep accuracy measure
            tools.update_surrogate_dataloader(s_data, sampled_k_confs, sampled_k_accs)

            if self.args.verbose:
                print("Trained architectures: ")
                print(list(zip(sampled_k_confs, sampled_k_accs)))

        return s_data

    # %%


"""
    Specialization for the AVMNIST dataset.
"""


class AVMNISTSearcher(ModelSearcher):
    def __init__(self, args, device):
        super(AVMNISTSearcher, self).__init__(args)

        self.device = device

        # Handle data
        transformer = transforms.Compose([
            avmnist_data.ToTensor(),
            avmnist_data.Normalize((0.1307,), (0.3081,))
        ])

        dataset_training = avmnist_data.AVMnist(args.datadir, transform=transformer, stage='train')
        dataset_validate = avmnist_data.AVMnist(args.datadir, transform=transformer, stage='train')

        train_indices = list(range(0, 50000))
        valid_indices = list(range(50000, 55000))

        train_subset = Subset(dataset_training, train_indices)
        valid_subset = Subset(dataset_validate, valid_indices)

        trainloader = torch.utils.data.DataLoader(train_subset, batch_size=args.batchsize, shuffle=True,
                                                  num_workers=args.num_workers)
        devloader = torch.utils.data.DataLoader(valid_subset, batch_size=args.batchsize, shuffle=False,
                                                num_workers=args.num_workers)

        self.dataloaders = {'train': trainloader, 'dev': devloader}

    def search(self):
        avmnist_searchmethods = {'train_sampled_fun': avmnist.train_sampled_models,
                                 'get_layer_confs': avmnist.get_possible_layer_configurations}

        if not self.args.randsearch:
            surrogate = surr.SimpleRecurrentSurrogate(100, 3, 100)
            surrogate.to(self.device)
            surrogate_dict = {'model': surrogate, 'criterion': torch.nn.MSELoss()}
            return self._epnas(avmnist.Searchable_Audio_Image_Net, surrogate_dict, self.dataloaders,
                               avmnist_searchmethods, self.device)
        else:
            return self._randsearch(avmnist.Searchable_Audio_Image_Net, self.dataloaders, avmnist_searchmethods,
                                    self.device)


# %%
"""
    Specialization for the NTU dataset.
"""


class NTUSearcher(ModelSearcher):
    def __init__(self, args, device):
        super(NTUSearcher, self).__init__(args)

        self.device = device

        # Handle data
        transformer_val = transforms.Compose([ntu_data.NormalizeLen(args.vid_len), ntu_data.ToTensor()])
        transformer_tra = transforms.Compose(
            [ntu_data.AugCrop(), ntu_data.NormalizeLen(args.vid_len), ntu_data.ToTensor()])

        dataset_training = ntu_data.NTU(args.datadir, transform=transformer_tra, stage='trainexp', args=args)
        dataset_dev = ntu_data.NTU(args.datadir, transform=transformer_val, stage='dev', args=args)

        datasets = {'train': dataset_training, 'dev': dataset_dev}
        self.dataloaders = {
            x: DataLoader(datasets[x], batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers,
                          drop_last=False) for x in ['train', 'dev']}

    def search(self):
        surrogate = surr.SimpleRecurrentSurrogate(100, 3, 100)
        surrogate.to(self.device)
        surrogate_dict = {'model': surrogate, 'criterion': torch.nn.MSELoss()}
        ntu_searchmethods = {'train_sampled_fun': ntu.train_sampled_models,
                             'get_layer_confs': ntu.get_possible_layer_configurations}

        return self._epnas(ntu.Searchable_Skeleton_Image_Net, surrogate_dict, self.dataloaders, ntu_searchmethods,
                           self.device)

    # %%


"""
    Specialization for the CIFAR-10 dataset.
"""


class CifarSearcher(ModelSearcher):
    def __init__(self, args, device):
        super(CifarSearcher, self).__init__(args)

        self.device = device

        train_indices = list(range(0, 45000))
        valid_indices = list(range(45000, 50000))

        # Handle data
        transformer_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transformer_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transformers = {'train': transformer_train, 'test': transformer_val}

        dataset_training = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True,
                                                        transform=transformers['train'])
        dataset_validate = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True,
                                                        transform=transformers['train'])

        train_subset = Subset(dataset_training, train_indices)
        valid_subset = Subset(dataset_validate, valid_indices)

        trainloader = torch.utils.data.DataLoader(train_subset, batch_size=args.batchsize, shuffle=True,
                                                  num_workers=args.num_workers)
        devloader = torch.utils.data.DataLoader(valid_subset, batch_size=args.batchsize, shuffle=False,
                                                num_workers=args.num_workers)

        self.dataloaders = {'train': trainloader, 'dev': devloader}

    def search(self):
        surrogate = surr.SimpleRecurrentSurrogate(100, 4, 100)
        surrogate.to(self.device)
        surrogate_dict = {'model': surrogate, 'criterion': torch.nn.MSELoss()}
        cifar_searchmethods = {'train_sampled_fun': cifar.train_sampled_models,
                               'get_layer_confs': cifar.get_possible_layer_configurations}

        return self._epnas(cifar.Searchable_MicroCNN, surrogate_dict, self.dataloaders, cifar_searchmethods,
                           self.device)
