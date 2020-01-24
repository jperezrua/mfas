#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: juanma
"""

#%%
import torch
import models.train.scheduler as sc
import copy
from sklearn.metrics import f1_score 

#%% for simple multimodal
def train_mmimdb_track_f1(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, 
                device=None, num_epochs=200, verbose=False, init_f1=0.0, th_fscore=0.3):
    
        
    best_model_sd = copy.deepcopy(model.state_dict())
    best_f1 = init_f1

    failsafe = True
    cont_overloop = 0
    while failsafe:
        for epoch in range(num_epochs):
            
            #if verbose:
            #    print()
            #    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            #    print('-' * 10)
            
           
            # Each epoch has a training and validation phase
            for phase in ['train', 'dev']:
    
                if phase == 'train':
                    if not isinstance(scheduler, sc.LRCosineAnnealingScheduler):
                        scheduler.step()
                    model.train(True)  # Set model to training mode
                else:
                    model.train(False)  # Set model to evaluate mode
                    list_preds = [] # ReSet predictions for F score
                    list_label = []                    
    
                running_loss = 0.0
                           
                # Iterate over data.
                for data in dataloaders[phase]:
                                   
                    # get the inputs
                    image, text, label = data['image'], data['text'], data['label']
    
                    # device
                    image = image.to(device)
                    text = text.to(device)                
                    label = label.to(device)
                    
                    #text = torch.nn.utils.rnn.pad_packed_sequence(text,textlen, batch_first=True)
    
                    # zero the parameter gradients
                    optimizer.zero_grad()
    
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        output = model(text, image)
                        #print(model.classifier.weight)
                        
                        if isinstance(output, tuple):
                            output = output[-1]
    
                        _, preds = torch.max(output, 1)
                        loss = criterion(output, label)
                    
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            if isinstance(scheduler, sc.LRCosineAnnealingScheduler):
                                scheduler.step()
                                scheduler.update_optimizer(optimizer)
                            loss.backward()
                            optimizer.step()
                        
                        if phase == 'dev':
                            preds_th = torch.nn.functional.sigmoid(output) > th_fscore
                            
                            list_preds.append(preds_th.cpu())
                            list_label.append(label.cpu()) 

    
                    # statistics
                    running_loss += loss.item() * image.size(0)
                    #running_corrects += torch.sum(preds == label.data)
                        
                epoch_loss = running_loss / dataset_sizes[phase]
                #epoch_acc  = running_corrects.double() / dataset_sizes[phase]                   
                
                if phase == 'dev':
                    y_pred = torch.cat(list_preds, dim=0).numpy()
                    y_true = torch.cat(list_label, dim=0).numpy()
                
                    curr_f1 = f1_score(y_true, y_pred, average='samples')  
                    #curr_f1 = f1_score(y_true, y_pred, average='micro')  
                    #curr_f1 = f1_score(y_true, y_pred, average='macro')  
                    #curr_f1 = f1_score(y_true, y_pred, average='weighted')                  
                
                    if verbose:
                        print('epoch #{} {} F1: {:.4f} '.format(epoch, phase, curr_f1))
                
                # deep copy the model
                if phase == 'train' and epoch_loss != epoch_loss:
                    print("Nan loss during training, escaping")
                    model.load_state_dict(best_model_sd)
                    model.train(False)                
                    return best_f1
                    
                    
                if phase == 'dev':
                    if curr_f1 > best_f1:
                        best_f1 = curr_f1
                        best_model_sd = copy.deepcopy(model.state_dict())
                    
        
        if best_f1 != best_f1 and num_epochs == 1 and cont_overloop < 1:
            failsafe = True
            print('Recording a NaN F1, training for one more epoch.')
        else:
            failsafe = False
            
        cont_overloop += 1
    
    model.load_state_dict(best_model_sd)
    model.train(False)
    
    if best_f1!=best_f1:
        best_f1 = 0.0
    
    return best_f1
