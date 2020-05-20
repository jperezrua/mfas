#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 17:00:28 2018

@author: juanma
"""

import models.searchable as S
import torch
import argparse
import time


# %%
def parse_args():
    parser = argparse.ArgumentParser(description='Modality optimization.')
    parser.add_argument('--checkpointdir', type=str, help='output base dir',
                        default='/home/juanma/Documents/Checkpoints/NTU/')
    parser.add_argument('--datadir', type=str, help='data directory',
                        default='/home/juanma/Documents/Data/ROSE_Action/')
    parser.add_argument('--ske_cp', type=str, help='Skeleton net checkpoint (assuming is contained in checkpointdir)',
                        default='skeleton_32frames_83.42')
    parser.add_argument('--rgb_cp', type=str, help='RGB net checkpoint (assuming is contained in checkpointdir)',
                        default='rgb_8frames_82.14')
    parser.add_argument('--num_outputs', type=int, help='output dimension', default=60)
    parser.add_argument('--batchsize', type=int, help='batch size', default=20)
    parser.add_argument('--inner_representation_size', type=int, help='output size of mixing linear layers', default=16)
    parser.add_argument('--epochs', type=int, help='training epochs', default=3)
    parser.add_argument('--lr_surrogate', type=float, help='learning rate surrogate', default=0.001)
    parser.add_argument('--epochs_surrogate', type=int, help='num of epochs for surrogate', default=50)
    parser.add_argument('--eta_max', type=float, help='eta max', default=0.001)
    parser.add_argument('--eta_min', type=float, help='eta min', default=0.000001)
    parser.add_argument('--Ti', type=int, help='epochs Ti', default=1)
    parser.add_argument('--Tm', type=int, help='epochs multiplier Tm', default=2)
    parser.add_argument('--use_dataparallel', help='Use several GPUs', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, help='Dataloader CPUS', default=16)
    parser.add_argument('--modality', type=str, help='', default='both')
    parser.add_argument('--max_fusions', type=int, dest="max_progression_levels", help='max fusions', default=4)
    parser.add_argument('--search_iterations', type=int, help='epnas iterations', default=3)
    parser.add_argument('--num_samples', type=int, help='number of samples to train at each explo step (K)', default=15)
    parser.add_argument('--initial_temperature', type=float, help='initial sampling temperature', default=10.0)
    parser.add_argument('--final_temperature', type=float, help='final sampling temperature', default=0.2)
    parser.add_argument('--temperature_decay', type=float, help='temperature decay (sigma)', default=4.0)
    parser.add_argument('--no-verbose', help='verbose', dest='verbose', action='store_false', default=True)
    parser.add_argument('--weightsharing', help='Weight sharing', action='store_true', default=False)
    parser.add_argument('--alphas', help='Use alphas', action='store_true', default=False)
    parser.add_argument('--batchnorm', help='Use batch norm', action='store_true', default=False)
    parser.add_argument('--multitask', help='Multitask loss', action='store_true', default=False)

    parser.add_argument("--vid_dim", action="store", default=256, dest="vid_dim",
                        help="frame side dimension (square image assumed) ")
    parser.add_argument("--vid_fr", action="store", default=30, dest="vi_fr", help="video frame rate")
    parser.add_argument("--vid_len", action="store", default=(8, 32), dest="vid_len", type=int, nargs='+',
                        help="length of video, as a tuple of two lengths, (rgb len, skel len)")
    parser.add_argument("--drpt", action="store", default=0.5, dest="drpt", type=float, help="dropout")

    parser.add_argument('--no_bad_skel', action="store_true",
                        help='Remove the 300 bad samples, espec. useful to evaluate', default=False)
    parser.add_argument("--no_norm", action="store_true", default=False, dest="no_norm",
                        help="Not normalizing the skeleton")

    return parser.parse_args()


# %%
if __name__ == "__main__":
    # %% parse args
    args = parse_args()
    # %% hardware
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")

    # %% Searcher
    ntu_searcher = S.NTUSearcher(args, device)

    # %% Do the search
    print("MFAS for NTU Started!!!!")
    start_time = time.time()
    surrogate_data = ntu_searcher.search()
    time_elapsed = time.time() - start_time
    print('Search complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # %% Get best K=5
    k_best, k_accs, idx = surrogate_data.get_k_best(5)

    print('Now listing best architectures')
    print(zip(k_best, k_accs))
