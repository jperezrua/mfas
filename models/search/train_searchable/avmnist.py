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
def train_avmnist_track_acc(model, criteria, optimizer, scheduler, dataloaders, dataset_sizes, 
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
                rgb, snd, label = data['image'], data['audio'], data['label']               
                
                # device
                rgb = rgb.to(device)
                snd = snd.to(device)                
                label = label.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()     
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    output = model((rgb, snd))
                    
                    if not multitask:
                        _, preds = torch.max(output, 1)
                        loss = criteria[0](output, label)
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
        
            epoch_acc  = running_corrects.double() / dataset_sizes[phase]
    
            print('{} Acc: {:.4f}'.format(phase, epoch_acc))
            
            # deep copy the model
            if phase == 'dev' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_sd = copy.deepcopy(model.state_dict())
                
    model.load_state_dict(best_model_sd)
    model.train(False)            
    
    return best_acc

#%% for simple multimodal
def test_avmnist_track_acc(model, dataloaders, dataset_sizes, 
                device=None, multitask=False):
    
    model.train(False)
    phase = 'test'
    

    running_corrects = 0
            
    # Iterate over data.
    for data in dataloaders[phase]:     
        
        # get the inputs
        rgb, snd, label = data['image'], data['audio'], data['label']                 
        
        # device
        rgb = rgb.to(device)
        snd = snd.to(device)                
        label = label.to(device)
        
       
        output = model((rgb, snd))
            
        if not multitask:
            _, preds = torch.max(output, 1)
        else:
            _, preds = torch.max(sum(output), 1)       
                
        # statistics
        running_corrects += torch.sum(preds == label.data)

    acc  = running_corrects.double() / dataset_sizes[phase]                    
    
    return acc