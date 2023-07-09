#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 00:21:37 2023

@author: bovey
"""

'''module will contain loss functions including DICE, focal loss
Haus distance, hybrid'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss
from scipy.ndimage import distance_transform_edt

class dice(nn.Module):
    """
    Calculates the dice loss between prediction and ground truth label tensors. Prediction tensor must be normalised using sigmoid function before
    calculation. 
    """
    def __init__(self):
        super(dice, self).__init__()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, predicted_output, label):
        assert predicted_output.size() == label.size(), 'predicted output and label must have the same dimensions'
        predicted_output = self.sigmoid(predicted_output)
        # Resizes or flattens the predicted and label tensors to calculate intersect between them
        predicted_output = predicted_output.view(1, -1)
        label = label.view(1, -1).float()
        intersect = (predicted_output * label).sum(-1)
        denominator = (predicted_output).sum(-1) + (label).sum(-1)
        dice_score = 2 * (intersect / denominator.clamp(min = 1e-6))
        
        return 1.0 - dice_score
    
class focal(nn.Module):
    '''focal loss (BCE)'''
    def __init__(self, alpha = 0.25, gamma = 2):
        super(focal, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, y_pred, y):
        p = torch.sigmoid(y_pred)
        ce_loss = F.binary_cross_entropy_with_logits(y_pred, y, reduction="none")
        p_t = p * y + (1 - p) * (1 - y)
        loss = ce_loss * ((1 - p_t)) ** self.gamma

        return loss.mean()


# class hausdorff(nn.Module):
    # dist_lst = []
    # for idx in range(len(vol_a)):
    #     dist_min = 1000.0        
    #     for idx2 in range(len(vol_b)):
    #         dist= np.linalg.norm(vol_a[idx]-vol_b[idx2])
    #         if dist_min > dist:
    #             dist_min = dist
    #     dist_lst.append(dist_min)
    # return np.max(dist_lst)
    '''95% hausdorff distance'''
    '''may be too expensive to calculate'''

def distance_map(labels) :
    labels = labels.numpy().astype(np.int16)
    assert set(np.unique(labels)).issubset([0,1]), 'Groundtruth labels must only have values 0 or 1'
    result = np.zeros_like(labels) # container to fill in distance values
    # for x in range(len(labels)):
    posmask = labels.astype(bool)
    negmask = ~posmask
    result = distance_transform_edt(negmask) * negmask - (distance_transform_edt(posmask) - 1) * posmask # Level set representation 

    return torch.Tensor(result).to(dtype = torch.int16)

class surfaceloss(nn.Module):
    """
    Object to calculate the Surface Loss between a prediction and ground truth image. Based on https://github.com/LIVIAETS/boundary-loss/blob/master/utils.py
    specified in "Kervadec, H. et al. (2018) ‘Boundary loss for highly unbalanced segmentation’, pp. 1–21. doi: 10.1016/j.media.2020.101851."
    Predicted tensor must be normalised using sigmoid function before loss calculation.
    """
    def __init__(self):
        super(surfaceloss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, predicted_output, distance_maps) :
        assert predicted_output.shape == distance_maps.shape
        predicted_output = self.sigmoid(predicted_output)
        predicted_output = predicted_output.type(torch.float32)
        surface_loss = predicted_output * distance_maps
        loss = surface_loss.mean()

        return loss

class hybridloss_ds(nn.Module):
    """
    Object uses both Dice Loss and Surface Loss in proportions defined in specified parameter alpha to calculate resultant loss to be used for model
    optimisation. (Note: Focal Loss has not been tested but should work.)
    """
    def __init__(self, loss_type = 'Dice'):
        super(hybridloss_ds, self).__init__()
        self.loss_1 = dice()
        self.surface_loss = surfaceloss()
        
    def forward(self, predicted_output, label, distance_map, alpha):
        self.alpha = alpha
        error = self.loss_1(predicted_output, label)
        self.dsc = self.alpha * error
        self.surface  = (1 - self.alpha) * self.surface_loss(predicted_output, distance_map) 
        return error, self.dsc + self.surface
        #return error, self.alpha * self.loss_1(predicted_output, label) + (1 - self.alpha) * self.surface_loss(predicted_output, distance_map) 
    
class hybridloss_fds(nn.Module):
    """
    Object uses both Dice Loss and Surface Loss in proportions defined in specified parameter alpha to calculate resultant loss to be used for model
    optimisation. (Note: Focal Loss has not been tested but should work.)
    """
    def __init__(self, alpha = 0.33, beta = 0.33, gamma = 0.33):
        super(hybridloss_fds, self).__init__()
        self.dice = dice()
        self.surface_loss = surfaceloss()
        self.focal = focal()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def forward(self, predicted_output, label, distance_map):
        dice_error = self.dice(predicted_output, label)
        surface_error = self.surface_loss(predicted_output, distance_map)
        focal_error = self.focal(predicted_output, label)
        
        self.fd = (dice_error * self.alpha + focal_error * self.beta) / (self.alpha + self.beta)
        self.sd = (dice_error * self.alpha + surface_error * self.gamma) / (self.alpha + self.gamma)
        self.fs = (focal_error * self.beta + surface_error * self.gamma) / (self.beta + self.gamma)
        
        self.fds = (focal_error * self.beta + surface_error * self.gamma + dice_error * self.alpha) / (self.alpha + self.beta + self.gamma)
        
        metrics = dice_error, surface_error, focal_error, self.fd, self.sd, self.fs, self.fds
        
        return metrics

