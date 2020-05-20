#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 17:00:28 2018

@author: juanma
"""

import models.search.ntu_searchable as ntu
import numpy as np
import torch
import argparse
import time
import os
import re

import torch.optim as op
import models.auxiliary.scheduler as sc
import models.search.train_searchable.ntu as tr


# %% Parse inputs

def parse_args():
    parser = argparse.ArgumentParser(description='Modality optimization.')
    parser.add_argument('--checkpointdir', type=str, help='output base dir',
                        default='/home/juanma/Documents/Checkpoints/NTU/')
    parser.add_argument('--datadir', type=str, help='data directory',
                        default='/home/juanma/Documents/Data/ROSE_Action/')
    parser.add_argument('--ske_cp', type=str, help='Skeleton net checkpoint (assuming is contained in checkpointdir)',
                        default='skeleton_32frames_85.24.checkpoint')
    parser.add_argument('--rgb_cp', type=str, help='RGB net checkpoint (assuming is contained in checkpointdir)',
                        default='rgb_8frames_83.91.checkpoint')
    parser.add_argument('--test_cp', type=str, help='Full net checkpoint (assuming is contained in checkpointdir)',
                        default='')
    parser.add_argument('--num_outputs', type=int, help='output dimension', default=60)
    parser.add_argument('--batchsize', type=int, help='batch size', default=20)
    parser.add_argument('--inner_representation_size', type=int, help='output size of mixing linear layers',
                        default=256)
    parser.add_argument('--epochs', type=int, help='training epochs', default=70)
    parser.add_argument('--eta_max', type=float, help='eta max', default=0.001)
    parser.add_argument('--eta_min', type=float, help='eta min', default=0.000001)
    parser.add_argument('--Ti', type=int, help='epochs Ti', default=5)
    parser.add_argument('--Tm', type=int, help='epochs multiplier Tm', default=2)
    parser.add_argument('--use_dataparallel', help='Use several GPUs', action='store_true', dest='use_dataparallel',
                        default=False)
    parser.add_argument('--j', dest='num_workers', type=int, help='Dataloader CPUS', default=16)
    parser.add_argument('--modality', type=str, help='', default='both')
    parser.add_argument('--no-verbose', help='verbose', action='store_false', dest='verbose', default=True)
    parser.add_argument('--weightsharing', help='Weight sharing', action='store_true', default=False)
    parser.add_argument('--no-multitask', dest='multitask', help='Multitask loss', action='store_false', default=True)
    parser.add_argument('--alphas', help='Use alphas', action='store_true', default=False)
    parser.add_argument('--batchnorm', help='Use batch norm', action='store_true', dest='batchnorm', default=False)

    parser.add_argument("--vid_dim", action="store", default=256, dest="vid_dim",
                        help="frame side dimension (square image assumed) ")
    parser.add_argument("--vid_fr", action="store", default=30, dest="vi_fr", help="video frame rate")
    parser.add_argument("--vid_len", action="store", default=(8, 32), dest="vid_len", type=int, nargs='+',
                        help="length of video, as a tuple of two lengths, (rgb len, skel len)")
    parser.add_argument("--drpt", action="store", default=0.4, dest="drpt", type=float, help="dropout")

    parser.add_argument('--no_bad_skel', action="store_true",
                        help='Remove the 300 bad samples, espec. useful to evaluate', default=False)
    parser.add_argument("--no_norm", action="store_true", default=False, dest="no_norm",
                        help="Not normalizing the skeleton")

    parser.add_argument('--conf', type=int, help='conf to train', default=1)

    return parser.parse_args()


# %%
def get_dataloaders(args):
    import torchvision.transforms as transforms
    from datasets import ntu as d
    from torch.utils.data import DataLoader

    # Handle data
    transformer_val = transforms.Compose([d.NormalizeLen(args.vid_len), d.ToTensor()])
    transformer_tra = transforms.Compose([d.AugCrop(), d.NormalizeLen(args.vid_len), d.ToTensor()])

    dataset_training = d.NTU(args.datadir, transform=transformer_tra, stage='train', args=args)
    dataset_testing = d.NTU(args.datadir, transform=transformer_val, stage='test', args=args)
    dataset_validation = d.NTU(args.datadir, transform=transformer_val, stage='dev', args=args)

    datasets = {'train': dataset_training, 'dev': dataset_validation, 'test': dataset_testing}

    dataloaders = {x: DataLoader(datasets[x], batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers,
                                 drop_last=False, pin_memory=True) for x in ['train', 'dev', 'test']}

    return dataloaders


def train_model(rmode, configuration, dataloaders, args, device):
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'test', 'dev']}

    if args.test_cp == '':
        num_batches_per_epoch = dataset_sizes['train'] / args.batchsize
        criteria = [torch.nn.CrossEntropyLoss(), torch.nn.CrossEntropyLoss(), torch.nn.CrossEntropyLoss()]

        # loading pretrained weights
        skemodel_filename = os.path.join(args.checkpointdir, args.ske_cp)
        rgbmodel_filename = os.path.join(args.checkpointdir, args.rgb_cp)
        rmode.skenet.load_state_dict(torch.load(skemodel_filename))
        rmode.rgbnet.load_state_dict(torch.load(rgbmodel_filename))

        # optimizer and scheduler
        params = rmode.central_params()
        optimizer = op.Adam(params, lr=args.eta_max / 10, weight_decay=1e-4)
        scheduler = sc.LRCosineAnnealingScheduler(args.eta_max, args.eta_min, args.Ti, args.Tm, num_batches_per_epoch)

        # hardware tuning
        if torch.cuda.device_count() > 1 and args.use_dataparallel:
            rmode = torch.nn.DataParallel(rmode)
        rmode.to(device)

        if args.verbose:
            print('Pretraining central weights: ')
            print(configuration)

        interm_model_acc = tr.train_ntu_track_acc(rmode, criteria, optimizer, scheduler, dataloaders, dataset_sizes,
                                                  device=device, num_epochs=1, verbose=args.verbose,
                                                  multitask=args.multitask)

        if args.verbose:
            print('Intermediate val accuracy: ' + str(interm_model_acc))

        if torch.cuda.device_count() > 1 and args.use_dataparallel:
            params = rmode.module.parameters()
        else:
            params = rmode.parameters()

        optimizer = op.Adam(params, lr=args.eta_max, weight_decay=1e-4)
        scheduler = sc.LRCosineAnnealingScheduler(args.eta_max, args.eta_min, args.Ti, args.Tm, num_batches_per_epoch)
        best_model_acc = tr.train_ntu_track_acc(rmode, criteria, optimizer, scheduler, dataloaders, dataset_sizes,
                                                device=device, num_epochs=args.epochs, verbose=args.verbose,
                                                multitask=args.multitask)

        if args.verbose:
            print('Final val accuracy: ' + str(best_model_acc))

    else:
        # perform test only (load weights then)
        fullmodel_filename = os.path.join(args.checkpointdir, args.test_cp)
        rmode.load_state_dict(torch.load(fullmodel_filename))

        # hardware tuning
        if torch.cuda.device_count() > 1 and args.use_dataparallel:
            rmode = torch.nn.DataParallel(rmode)
        rmode.to(device)

    test_model_acc = tr.test_ntu_track_acc(rmode, dataloaders, dataset_sizes, device=device, multitask=args.multitask)

    if args.verbose:
        print('Final test accuracy: ' + str(test_model_acc))

    return test_model_acc


# %%
if __name__ == "__main__":
    # %%
    print("Training found NTU network")
    args = parse_args()
    print("The configuration of this run is:")
    print(args)

    # %% hardware
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")

    # %% Train found
    if args.conf == 0:
        configuration = np.array([[2, 2, 0], [1, 0, 1], [3, 2, 0], [3, 1, 1]])
    elif args.conf == 1:
        configuration = np.array([[3, 0, 0], [1, 3, 0], [1, 1, 1], [3, 3, 0]])
    elif args.conf == 2:
        configuration = np.array([[3, 2, 0], [2, 3, 1], [0, 1, 1], [3, 0, 0]])
    elif args.conf == 3:
        configuration = np.array([[1, 1, 1], [3, 2, 0], [0, 1, 1], [3, 0, 0]])
    elif args.conf == 4:
        configuration = np.array([[3, 1, 1], [1, 3, 0], [1, 1, 1], [3, 3, 0]])

    rmode = ntu.Searchable_Skeleton_Image_Net(args, configuration)
    dataloaders = get_dataloaders(args)
    start_time = time.time()
    modelacc = train_model(rmode, configuration, dataloaders, args, device)
    time_elapsed = time.time() - start_time
    print('Training in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Model Acc: {}'.format(modelacc))

    # %%
    confstr = np.array2string(configuration, precision=1, separator='_', suppress_small=True)
    confstr = re.sub(r"_\n ", "_", confstr)

    # filename = args.checkpointdir+"/final_conf_" + confstr + "_" + str(modelacc.item())+'.checkpoint'
    # torch.save(rmode.state_dict(), filename)
