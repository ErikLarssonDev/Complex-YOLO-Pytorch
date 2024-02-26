import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.append('../')

from data_process.ZOD_dataset import ZOD
from data_process.transformation import Compose, OneOf, Random_Rotation, Random_Scaling, Horizontal_Flip, Cutout


def create_train_dataloader(configs):
    """Create dataloader for training"""

    train_lidar_transforms = OneOf([
        Random_Rotation(limit_angle=20., p=1.0),
        Random_Scaling(scaling_range=(0.95, 1.05), p=1.0)
    ], p=0.66)

    train_aug_transforms = Compose([
        Horizontal_Flip(p=configs.hflip_prob),
        Cutout(n_holes=configs.cutout_nholes, ratio=configs.cutout_ratio, fill_value=configs.cutout_fill_value,
               p=configs.cutout_prob)
    ], p=1.)

    train_dataset = ZOD(configs.dataset_dir, mode='train', lidar_transforms=train_lidar_transforms,
                                 aug_transforms=train_aug_transforms, multiscale=configs.multiscale_training,
                                 num_samples=configs.num_samples, mosaic=configs.mosaic,
                                 random_padding=configs.random_padding)
    train_sampler = None
    if configs.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=(train_sampler is None),
                                  pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=train_sampler,
                                  collate_fn=train_dataset.collate_fn)

    return train_dataloader, train_sampler


def create_val_dataloader(configs):
    """Create dataloader for validation"""
    val_sampler = None
    val_dataset = ZOD(configs.dataset_dir, mode='val', lidar_transforms=None, aug_transforms=None,
                               multiscale=False, num_samples=configs.num_samples, mosaic=False, random_padding=False)
    if configs.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False,
                                pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=val_sampler,
                                collate_fn=val_dataset.collate_fn)

    return val_dataloader


def create_test_dataloader(configs):
    """Create dataloader for testing phase"""

    test_dataset = ZOD(configs.dataset_dir, mode='test', lidar_transforms=None, aug_transforms=None,
                                multiscale=False, num_samples=configs.num_samples, mosaic=False, random_padding=False)
    test_sampler = None
    if configs.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=False,
                                 pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=test_sampler)

    return test_dataloader

# TODO: Fix so that this depends on the config, move to dataset
def transform_coordinates(x, y): 
    # Input range: [-250, 250], center at (0, 0)
    # Output range: [0, 608], center at (304, 304)

    # Shift the coordinates to have the center at (0, 0)
    x_centered = x + 250
    y_centered = y + 250

    # Scale the coordinates to the new range
    x_transformed = (x_centered / 500) * 608
    y_transformed = (y_centered / 500) * 608

    return x_transformed, y_transformed


if __name__ == '__main__':
    import argparse
    import os

    import cv2
    import numpy as np
    from easydict import EasyDict as edict

    import data_process.kitti_bev_utils as bev_utils
    from data_process import kitti_data_utils
    from utils.visualization_utils import show_image_with_boxes, merge_rgb_to_bev, invert_target
    import config.ZOD_config as cnf

    parser = argparse.ArgumentParser(description='Complexer YOLO Implementation')

    parser.add_argument('--img_size', type=int, default=608,
                        help='the size of input image')
    parser.add_argument('--hflip_prob', type=float, default=0.,
                        help='The probability of horizontal flip')
    parser.add_argument('--cutout_prob', type=float, default=0.,
                        help='The probability of cutout augmentation')
    parser.add_argument('--cutout_nholes', type=int, default=1,
                        help='The number of cutout area')
    parser.add_argument('--cutout_ratio', type=float, default=0.3,
                        help='The max ratio of the cutout area')
    parser.add_argument('--cutout_fill_value', type=float, default=0.,
                        help='The fill value in the cut out area, default 0. (black)')
    parser.add_argument('--multiscale_training', action='store_true',
                        help='If true, use scaling data for training')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 1)')
    parser.add_argument('--mosaic', action='store_true',
                        help='If true, compose training samples as mosaics')
    parser.add_argument('--random-padding', action='store_true',
                        help='If true, random padding if using mosaic augmentation')
    parser.add_argument('--show-train-data', action='store_true',
                        help='If true, random padding if using mosaic augmentation')
    parser.add_argument('--output-width', type=int, default=608,
                        help='the width of showing output, the height maybe vary')
    parser.add_argument('--save_img', action='store_true',
                        help='If true, save the images')

    configs = edict(vars(parser.parse_args()))
    configs.distributed = False  # For testing
    configs.pin_memory = False
    configs.dataset_dir = os.path.join('../../', 'minzod_mmdet3d')

    if configs.save_img:
        print('saving validation images')
        configs.saved_dir = os.path.join(configs.dataset_dir, 'validation_data')
        if not os.path.isdir(configs.saved_dir):
            os.makedirs(configs.saved_dir)

    if configs.show_train_data:
        dataloader, _ = create_train_dataloader(configs)
        print('len train dataloader: {}'.format(len(dataloader)))
    else:
        dataloader = create_val_dataloader(configs)
        print('len val dataloader: {}'.format(len(dataloader)))

    show_next_image = True
    print(configs)
    print('\n\nPress n to see the next sample >>> Press Esc to quit...')

    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        # Rescale target
        # We go from coord x, y in [-250, 250] and center is (0,0) --> x, y in [0, 608] and center is (304, 304), TODO: Don't we need to do something more?
        
        targets[:, 2:6] *= configs.img_size

        # Get yaw angle
        targets[:, 6] = torch.atan2(targets[:, 6], targets[:, 7])

        img_bev = imgs.squeeze() * 255
        img_bev = img_bev.permute(1, 2, 0).numpy().astype(np.uint8)
        img_bev = cv2.resize(img_bev, (configs.img_size, configs.img_size)) # TODO: Whe should resize but keep the aspect ratio
        
        for c, x, y, w, l, yaw in targets[:, 1:7].numpy(): # targets = [cl, y1, x1, w1, l1, math.sin(float(yaw)), math.cos(float(yaw))]
            # Draw rotated box
            bev_utils.drawRotatedBox(img_bev, x, y, w, l, yaw, cnf.colors[int(c)])

        img_bev = cv2.rotate(img_bev, cv2.ROTATE_180)

        if configs.mosaic and configs.show_train_data:
            if configs.save_img:
                fn = os.path.basename(img_file)
                cv2.imwrite(os.path.join(configs.saved_dir, fn), img_bev)
            else:
                cv2.imshow('mosaic_sample', img_bev)
        else:
            # out_img = merge_rgb_to_bev(img_rgb, img_bev, output_width=configs.output_width)
            out_img = img_bev
            if configs.save_img:
                fn = os.path.basename(img_file)
                cv2.imwrite(os.path.join(configs.saved_dir, fn), out_img)
            else:
                cv2.imshow('single_sample', out_img)

        if not configs.save_img:
                if show_next_image:
                    key = cv2.waitKey(0) & 0xFF  # Ensure the result is an 8-bit integer
                    if key == 27:  # Check if 'Esc' key is pressed
                        cv2.destroyAllWindows() 
                        break
                    elif key == 110:
                        print(f"\nShowing image {batch_i}\n")
                        show_next_image = True  # Set the flag to False to avoid showing the same image again
                        continue  # Skip the rest of the loop and go to the next iteration

        show_next_image = True  # Set the flag to True for the next iteration

