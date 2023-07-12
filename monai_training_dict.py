#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 21:14:08 2023

@author: bovey
"""

import logging
import os
import sys
import tempfile
from glob import glob

import nibabel as nib
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import ImageDataset, create_test_image_3d, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    RandGaussianNoise,
    Compose,
    SpatialPad,
    RandRotate90,
    RandSpatialCrop,
    RandRotate,
    ScaleIntensity,
)
from monai.visualize import plot_2d_or_3d_image

train_path = "/home/bovey/Desktop/MVSEG2023/train"
val_path = "/home/bovey/Desktop/MVSEG2023/val"


##parameters to consider changing



def main(train_dir, val_dir):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    train_images = sorted(glob(os.path.join(train_dir, "*-US.nii")))
    train_segs = sorted(glob(os.path.join(train_dir, "*-label.nii")))

    val_images = sorted(glob(os.path.join(val_dir, "*-US.nii")))
    val_segs = sorted(glob(os.path.join(val_dir, "*-label.nii")))

    train_imtrans = Compose(
        [
            ScaleIntensity(),
            EnsureChannelFirst(),
            SpatialPad((336,256,208)),
            RandGaussianNoise(prob=0.1, mean=0.0, std=0.1),
            RandRotate(range_x=0.1, range_y=0.1, range_z=0.1, prob=0.1, keep_size=True),
            
        ]
    )

    train_segtrans = Compose(
        [
            EnsureChannelFirst(),
            SpatialPad((336,256,208)), 
            RandRotate(range_x=0.1, range_y=0.1, range_z=0.1, prob=0.1, keep_size=True),
            
        ]
    )

    val_imtrans = Compose(
        [
            ScaleIntensity(),
            EnsureChannelFirst(),
            SpatialPad((336,256,208))
        ]
    )

    val_segtrans = Compose(
        [
            EnsureChannelFirst(),
            SpatialPad((336,256,208))
        ]
    )


    # check_ds = ImageDataset(train_images, train_segs, transform=train_imtrans, seg_transform=train_segtrans)
    # check_loader = DataLoader(check_ds, batch_size=2, num_workers=2, pin_memory=torch.cuda.is_available())
    # im, seg = monai.utils.misc.first(check_loader)
    # print(im.shape, seg.shape)

    train_ds = ImageDataset(train_images[:], train_segs[:], transform=train_imtrans, seg_transform=train_segtrans)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())
    # create a validation data loader
    val_ds = ImageDataset(val_images[:], val_segs[:], transform=val_imtrans, seg_transform=val_segtrans)
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=4, pin_memory=torch.cuda.is_available())
    
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_trans = Compose([Activations(softmax=True), AsDiscrete(threshold=0.5)])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    loss_function = monai.losses.DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)

    
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter()
    for epoch in range(5):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{5}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in val_loader:
                    val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                    roi_size = None
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)
                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), "best_metric_model_segmentation3d_array.pth")
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_mean_dice", metric, epoch + 1)
                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
                plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
                plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tempdir:
        main(train_path, val_path)
























