#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 22:19:13 2023

@author: bovey
"""

import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from net import VNet
from preprocess import *
from performance import *

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


train_directory = "/home/bovey/Desktop/MVSEG2023/train"
val_directory = "/home/bovey/Desktop/MVSEG2023/val"

train_dataset = datareader(data_directory = train_directory, train=True)
val_dataset = datareader(data_directory = val_directory, train=False)

#Hyperparameters
num_epochs = 200

for trial in range(5):
    ##########################
    ### HYPERPARAMETERS to randomize
    ##########################
    name_label = str(int(time.time() % 10000000000))
    learning_rate = random.randint(1, 10000)/10000000
    gamma_ex = random.randint(900, 989)/1000
    scheduler_rand = random.randint(0,2)
    optimizer_choose = random.randint(0,3)
    batch_size = 2
    
    
    print(batch_size, learning_rate, gamma_ex, scheduler_rand, optimizer_choose)
    epoch_break_tracker = 0

    train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True, num_workers=2, pin_memory=True)
    
    val_loader = DataLoader(dataset=val_dataset, 
                          batch_size=1, 
                          shuffle=True, num_workers=2, pin_memory=True)
    
    model = VNet()
    model = model.to(device)
    
    if optimizer_choose == 0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if optimizer_choose == 1:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if optimizer_choose == 2:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    if optimizer_choose == 3:
        optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
    schedulerCOSAN = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs*len(train_loader))
    schedulerReduceLROnPlateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, mode='max', patience=10, threshold=0.01)
    
    if scheduler_rand == 0:
        scheduler = schedulerCOSAN
    if scheduler_rand == 1:
        scheduler = schedulerReduceLROnPlateau
    
    loss_function = focal()
    
    path = "/home/bovey/Desktop/MVSEG2023/trial"
    os.mkdir(os.path.join(path, name_label))
    save_dir = '/home/bovey/Desktop/MVSEG2023/trial/' + name_label + '/'
    
    for epoch in range(30):
        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.float)
            logits = model(features)
            loss = loss_function(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if not batch_idx % 50:
                print ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' 
                       %(epoch+1, num_epochs, batch_idx, 
                         len(train_loader), loss), 'time is', time.time())
            
        model.eval()
        dice_tracker = 0
        with torch.no_grad():
            for batch_idx, (features, targets) in enumerate(val_loader):
                features = features.to(device, dtype=torch.float)
                targets = targets.to(device, dtype=torch.float)
                logits = model(features)
                dice_val = dice()
                dice_tracker += dice_val(logits, targets)
        print(dice_tracker / len(val_loader))
            
            
# 2 0.0008827 0.909 1 3 got dice of .9931
# 2 6.28e-05 0.953 1 2 looks like it's getting dice of .9619

















