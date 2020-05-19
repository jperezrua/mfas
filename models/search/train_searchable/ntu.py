#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: juanma
"""

#%%
import torch
import models.auxiliary.scheduler as sc
import copy

#%% for simple multimodal
def train_ntu_track_acc(model, criteria, optimizer, scheduler, dataloaders, dataset_sizes, 
                device=None, num_epochs=200, verbose=False, multitask=False):
    
    best_model_sd = copy.deepcopy(model.state_dict())
    best_acc = 0
    
    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'dev']: 

            if phase == 'train':
                if not isinstance(scheduler, sc.LRCosineAnnealingScheduler):
                    scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data.
            for data in dataloaders[phase]:     
                
                # get the inputs
                rgb, ske, label = data['rgb'], data['ske'], data['label']                
                
                # device
                rgb = rgb.to(device)
                ske = ske.to(device)                
                label = label.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()     
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    output = model((rgb, ske))
                    
                    if not multitask:
                        _, preds = torch.max(output, 1)
                        if isinstance(criteria, list):
                            loss = criteria[0](output, label)
                        else:
                            loss = criteria(output, label)
                    else:
                        _, preds = torch.max(sum(output), 1)
                        loss = criteria[0](output[0], label) + criteria[1](output[1], label) + criteria[2](output[2], label)
                
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        if isinstance(scheduler, sc.LRCosineAnnealingScheduler):
                            scheduler.step()
                            scheduler.update_optimizer(optimizer)
                        loss.backward()
                        optimizer.step()
                        
                # statistics
                running_loss += loss.item() * rgb.size(0)
                running_corrects += torch.sum(preds == label.data)
        
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc  = running_corrects.double() / dataset_sizes[phase]
    
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            # deep copy the model
            if phase == 'dev' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_sd = copy.deepcopy(model.state_dict())
                
    model.load_state_dict(best_model_sd)
    model.train(False)            
    
    return best_acc

#%% for simple multimodal
def test_ntu_track_acc(model, dataloaders, dataset_sizes, 
                device=None, multitask=False):
    
    model.train(False)
    phase = 'test'
    

    running_corrects = 0
            
    # Iterate over data.
    for data in dataloaders[phase]:     
        
        # get the inputs
        rgb, ske, label = data['rgb'], data['ske'], data['label']                
        
        # device
        rgb = rgb.to(device)
        ske = ske.to(device)                
        label = label.to(device)
        
       
        output = model((rgb, ske))
            
        if not multitask:
            _, preds = torch.max(output, 1)
        else:
            _, preds = torch.max(sum(output), 1)       
                
        # statistics
        running_corrects += torch.sum(preds == label.data)

    acc  = running_corrects.double() / dataset_sizes[phase]                    
    
    return acc