#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 00:32:49 2023

@author: bovey
"""

import os
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.ndimage import rotate

'''
image pre-processing and data augmentation

pre-processing strategies:
    normalization
    
augmentation strategies:
    angle tilt (x3 dimensions) at -5, -4...4, 5
    additive noise
    crop (non-central, biases will be determined in the future***)
'''

image_path = "/home/bovey/Desktop/MVSEG2023/val/val_004-US.nii"
label_path = "/home/bovey/Desktop/MVSEG2023/val/val_004-label.nii"
img = nib.load(image_path).get_fdata(dtype=np.float64)
label = nib.load(label_path).get_fdata()

class datareader(Dataset):
    def __init__(self, data_directory, train = True, size = (336,256,208), transform=None):
        '''data_dir is location; dir is divided by train and label so will just be all within that dir
        size is max size of each dimension
        transform if true will apply'''
        US_file_names = []
        label_file_names = []
        self.train = train
        if self.train:
            for i in range(1, 106):
                US_file_names += ["/home/bovey/Desktop/MVSEG2023/train/train_" + '%03.0f' % i + "-US.nii"]
                label_file_names += ["/home/bovey/Desktop/MVSEG2023/train/train_" + '%03.0f' % i + "-label.nii"]
        else:
            for i in range(1, 31):
                US_file_names += ["/home/bovey/Desktop/MVSEG2023/val/val_" + '%03.0f' % i + "-US.nii"]
                label_file_names += ["/home/bovey/Desktop/MVSEG2023/val/val_" + '%03.0f' % i + "-label.nii"]
        self.US_file_names = US_file_names
        self.label_file_names = label_file_names
        
    def __getitem__(self, index):
        '''index of item
        returns image and label'''
        random_seed = torch.randint(0,101, (3,))
        image_name = self.US_file_names[index]
        label_name = self.label_file_names[index]
        
        img = nib.load(image_name).get_fdata(dtype=np.float64)
        label_temp = nib.load(label_name).get_fdata(dtype=np.float64)
        label = np.where(label_temp > 0.5, 1, 0)
        i, j, k = img.shape
        
        img2 = np.pad(img, ((round((336-i)/2), round((336-i)/2)), (round((256-j)/2), round((256-j)/2)), (0, 0)), mode='edge')
        label2 = np.pad(label, ((round((336-i)/2), round((336-i)/2)), (round((256-j)/2), round((256-j)/2)), (0, 0)), mode='edge')
                
        # create an extra channel for label split; img will be identical
        
        if 0 < random_seed[0] < 30 and self.train:
            self.normalization_totensor = normalization_totensor()
            self.gaussian = gaussian()
            img2 = self.normalization_totensor(img2, label = True)
            label2 = self.normalization_totensor(label2)
            img2 = self.gaussian(img2)
            return img2, label2
        
        if 30 < random_seed[0] < 50 and self.train:
            rotate_tuple = (round(random_seed[1].item()/10), round(random_seed[2].item()))
            self.rotation = rotation(degrees = rotate_tuple[0] - 5)
            self.normalization_totensor = normalization_totensor()
            
            if 0 < rotate_tuple[1] <= 33:
                img2 = self.rotation(img2, direction = 'z')
                labels2 = self.rotation(label2, direction = 'z')
            elif 33 < rotate_tuple[1] <= 67:
                img2 = self.rotation(img2, direction = 'y')
                labels2 = self.rotation(label2, direction = 'y')
            else:
                img2 = self.rotation(img2, direction = 'x')
                labels2 = self.rotation(label2, direction = 'x')
            img2 = self.normalization_totensor(img2)
            label2 = self.normalization_totensor(label2, label = False)
            return img2, label2
            
        else:
            self.normalization_totensor = normalization_totensor()
            img2 = self.normalization_totensor(img2, label = True)
            label2 = self.normalization_totensor(label2)
            return img2, label2
        
    def __len__(self):
        return len(self.US_file_names)
            
class normalization_totensor:
    '''normalization of 3D echo objects'''
    def __init__(self, mean = 38.1, std = 29.4, relu = False):
        self.mean = mean
        self.std = std
        self.relu = relu
    
    def __call__(self, echo, label = False):
        '''self-designate mean and std'''
        if label:
            norm_echo = (echo - self.mean) / self.std
            tor_echo = torch.from_numpy(norm_echo)
        else:
            tor_echo = torch.from_numpy(echo)
        '''for upper and lower threshold, can subtract the numbers then relu'''
        return tor_echo.view((1,336,256,208))

class gaussian:
    '''gaussian noise for data augmentation'''
    def __init__(self, variance = 0.01 ** 0.5, probs = 0.25):
        self.variance = variance
        self.probs = probs
    
    def __call__(self, echo):
        if torch.randint(0, 100, (1,)).item()/100 <= self.probs:
            X = echo + self.variance * torch.randn(echo.shape)
            return X
        else:
            return echo

class rotation:
    '''augmentation strat +/- degrees along given axis
    accepts a tuple of size dim 3 which will rotate along (z,y,x) axis'''
    def __init__(self, degrees = 5):
        self.rotate = degrees
    
    def __call__(self, echo, label = False, direction = 'z'):
        if direction == 'z':
            rotation1 = rotate(echo, self.rotate, mode='nearest', axes=(0,1), reshape=False)
        elif direction == 'y':
            rotation1 = rotate(echo, self.rotate, mode='nearest', axes=(0,2), reshape=False)
        elif direction == 'x':
            rotation1 = rotate(echo, self.rotate, mode='nearest', axes=(1,2), reshape=False)
        return rotation1
            
class shift_crop:
    '''identification of likely region of mitral valve
    where l, w, h are tuples of respective dimension'''
    def __init__(self, z, y, x):
        self.z = z
        self.y = y
        self.x = x
    
    def __call__(self, echo, label = False):
        return echo[self.z[0]:self.z[1],
                    self.y[0]:self.y[1],
                    self.x[0]:self.x[1]]





