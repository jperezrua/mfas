#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: juanma
"""

#%%
import torch
import models.aux.scheduler as sc
import copy

#%% for simple multimodal
def train_cifar_track_acc(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, 
                device, num_epochs=200, verbose=False, use_intermediate=False):
    
    best_model_sd = copy.deepcopy(model.state_dict())
    best_error = 1e100
    criterion2 = torch.nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        
        if verbose:
            print()
        
               # Each epoch has a training and validation phase
        for phase in ['train', 'dev']:

            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data.
            for data in dataloaders[phase]:
                               
                # get the inputs
                rgb, gt_label = data[0], data[1]

                # device
                rgb = rgb.to(device)
                gt_label = gt_label.to(device)                

                # zero the parameter gradients
                optimizer.zero_grad()
               
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    
                    output, output_i = model(rgb)
                    
                    if not use_intermediate:
                        loss = criterion(output, gt_label)
                    else:
                        loss = criterion(output, gt_label) + 0.4*criterion2(output_i, gt_label)

                    _, preds = torch.max(output, 1)
                
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        scheduler.step()
                        if isinstance(scheduler, sc.LRCosineAnnealingScheduler):
                            scheduler.update_optimizer(optimizer)                    
                        loss.backward()
                        
                        #_ = torch.nn.utils.clip_grad_norm_(model.parameters(), 25.0)
                        optimizer.step()
                      
                # statistics
                running_loss += loss.item() * rgb.size(0)
                running_corrects += torch.sum(preds == gt_label.data)
                
            epoch_error = 1.0 - running_corrects.double() / dataset_sizes[phase]
            
            # deep copy the model
            if phase == 'dev':
                if epoch_error < best_error:
                    best_error = epoch_error
                    best_model_sd = copy.deepcopy(model.state_dict())                    


                if verbose:
                    print('Epoch #{} val error: {}'.format(epoch, epoch_error))

    model.load_state_dict(best_model_sd)
    model.train(False)
    
    if verbose:
        print('Best val error: {}'.format(best_error))
    
    return 1.0 - best_error


#%% for simple multimodal
def test_cifar_track_acc(model, dataloaders, dataset_sizes, device):
    

    phase = 'test'
    model.train(False)  # Set model to evaluate mode
    
    
    running_corrects = 0
            
    # Iterate over data.
    for data in dataloaders[phase]:
                       
        # get the inputs
        rgb, gt_label = data[0], data[1]

        # device
        rgb = rgb.to(device)
        gt_label = gt_label.to(device)                
    
        # forward            
        output, _ = model(rgb)
        _, preds = torch.max(output, 1)
                      
        # statistics
        running_corrects += torch.sum(preds == gt_label.data)
        
    acc = running_corrects.double() / dataset_sizes[phase]
        
    return acc