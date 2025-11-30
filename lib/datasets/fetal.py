# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# Modified by Netanell Avisdris and Chiara Di Vece (chiara.divece.20@ucl.ac.uk)
# ------------------------------------------------------------------------------

import os
import random

import torch
import torch.utils.data as data
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import cv2
from sklearn.mixture import GaussianMixture

from ..utils.transforms import fliplr_joints, crop, generate_target, transform_pixel


class FetalLandmarks(data.Dataset):
    """
    PyTorch Dataset for fetal biometry landmark detection.
    
    Loads ultrasound images and corresponding anatomical landmark annotations
    for training and evaluation of landmark detection models. Supports multiple
    anatomies (brain/head, abdomen, femur) and metrics (BPD, OFD, TAD, APAD, FL).
    
    Args:
        cfg: Configuration object containing dataset and model parameters
        is_train (bool): Whether this is training data (applies augmentation)
        transform: Optional transform to be applied on samples (not currently used)
    
    Attributes:
        anatomy: Type of anatomy ('brain', 'abdomen', 'femur')
        metrics: Type of measurement ('BPD', 'OFD', 'TAD', 'APAD', 'FL')
        reassign: Whether to apply landmark reassignment based on learned direction
    """

    def __init__(self, cfg, is_train=True, transform=None):
        # specify annotation file for dataset
        if is_train:
            self.csv_file = cfg.DATASET.TRAINSET
        else:
            self.csv_file = cfg.DATASET.TESTSET

        self.is_train = is_train
        self.transform = transform
        self.data_root = cfg.DATASET.ROOT
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.sigma = cfg.MODEL.SIGMA
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rot_factor = cfg.DATASET.ROT_FACTOR
        self.label_type = cfg.MODEL.TARGET_TYPE
        self.flip = cfg.DATASET.FLIP
        self.reassign = cfg.TRAIN.REASSIGN
        self.anatomy = cfg.DATASET.ANATOMY
        self.metrics = cfg.DATASET.METRICS

        # load annotations
        self.landmarks_frame = pd.read_csv(self.csv_file, header=0, sep=',')
        # Drop 'index' column
        self.landmarks_frame.drop(self.landmarks_frame.columns[0], axis=1, inplace=True)

        # Remove rows where any landmarks are negative
        if self.anatomy == 'brain':
            if self.metrics == 'OFD':
                landmark_cols = self.landmarks_frame.columns[4:8]  # OFD
            elif self.metrics == 'BPD':
                landmark_cols = self.landmarks_frame.columns[8:12]  # BPD
            else:
                raise ValueError(f"Metrics {self.metrics} not supported")
        elif self.anatomy == 'abdomen':
            if self.metrics == 'TAD':
                landmark_cols = self.landmarks_frame.columns[4:8]   # TAD
            elif self.metrics == 'APAD':
                landmark_cols = self.landmarks_frame.columns[8:12]   # APAD
            else:
                raise ValueError(f"Metrics {self.metrics} not supported")
        elif self.anatomy == 'femur':
            if self.metrics == 'FL':
                landmark_cols = self.landmarks_frame.columns[4:8]   # FL
            else:
                raise ValueError(f"Metrics {self.metrics} not supported")
        else:
            raise ValueError(f"Anatomy {self.anatomy} not supported")

        # mask = (self.landmarks_frame[landmark_cols] < 0).any(axis=1)
        mask = (self.landmarks_frame[landmark_cols] < 0) | (self.landmarks_frame[landmark_cols].isna())
        mask = mask.any(axis=1)
        self.landmarks_frame = self.landmarks_frame[~mask].reset_index(drop=True)

        if is_train:
            if self.anatomy == 'brain':
                if self.metrics == 'OFD':
                    landmarks = np.array(self.landmarks_frame.iloc[:, 4:8].values, dtype=np.float32)  # OFD
                elif self.metrics == 'BPD':
                    landmarks = np.array(self.landmarks_frame.iloc[:, 8:12].values, dtype=np.float32)  # BPD
                else:
                    raise ValueError(f"Metrics {self.metrics} not supported")
            elif self.anatomy == 'abdomen':
                if self.metrics == 'TAD':
                    landmarks = np.array(self.landmarks_frame.iloc[:, 4:8].values, dtype=np.float32)  # TAD
                elif self.metrics == 'APAD':
                    landmarks = np.array(self.landmarks_frame.iloc[:, 8:12].values, dtype=np.float32)  # APAD
                else:
                    raise ValueError(f"Metrics {self.metrics} not supported") 
            elif self.anatomy == 'femur':
                if self.metrics == 'FL':
                    landmarks = np.array(self.landmarks_frame.iloc[:, 4:8].values, dtype=np.float32)  # FL
                else:
                    raise ValueError(f"Metrics {self.metrics} not supported")
            else:
                raise ValueError(f"Anatomy {self.anatomy} not supported")

            landmarks = landmarks.reshape(-1, 2)
            self.d_vect = determine_direction(landmarks)

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (img, target, meta) where:
                - img: Preprocessed and normalized image tensor (3, H, W)
                - target: Heatmap target tensor (num_landmarks, H_out, W_out)
                - meta: Dictionary containing metadata (index, center, scale, pts, tpts)
        """
        
        image_path = os.path.join(self.data_root,
                                  self.landmarks_frame.iloc[idx, 0])
        scale = self.landmarks_frame.iloc[idx, 1]

        center_w = self.landmarks_frame.iloc[idx, 2]
        center_h = self.landmarks_frame.iloc[idx, 3]
        center = torch.Tensor([center_w, center_h])

        if self.anatomy == 'brain':
            if self.metrics == 'OFD':
                pts = self.landmarks_frame.iloc[idx, 4:8].values  # OFD
            elif self.metrics == 'BPD':
                pts = self.landmarks_frame.iloc[idx, 8:12].values  # BPD
            else:
                raise ValueError(f"Metrics {self.metrics} not supported")
        elif self.anatomy == 'abdomen':
            if self.metrics == 'TAD':
                pts = self.landmarks_frame.iloc[idx, 4:8].values   # TAD
            elif self.metrics == 'APAD':
                pts = self.landmarks_frame.iloc[idx, 8:12].values   # APAD
            else:
                raise ValueError(f"Metrics {self.metrics} not supported")
        elif self.anatomy == 'femur':
            if self.metrics == 'FL':
                pts = self.landmarks_frame.iloc[idx, 4:8].values   # FL
            else:
                raise ValueError(f"Metrics {self.metrics} not supported")
        else:
            raise ValueError(f"Anatomy {self.anatomy} not supported")
        
        pts = pts.astype('float').reshape(-1, 2)

        scale *= 1.7
        nparts = pts.shape[0]
        img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)

        r = 0
        if self.is_train:
            scale = scale * (random.uniform(1 - self.scale_factor,
                                            1 + self.scale_factor))
            r = random.uniform(-self.rot_factor, self.rot_factor) \
                if random.random() <= 0.6 else 0
            if random.random() <= 0.5 and self.flip:
                img = np.fliplr(img)
                pts = fliplr_joints(pts, width=img.shape[1], dataset='FETAL')
                center[0] = img.shape[1] - center[0]

        img = crop(img, center, scale, self.input_size, rot=r)

        target = np.zeros((nparts, self.output_size[0], self.output_size[1]))
        tpts = pts.copy()

        for i in range(nparts):
            if tpts[i, 1] >= 0:
                tpts[i, 0:2] = transform_pixel(tpts[i, 0:2]+1, center,
                                               scale, self.output_size, rot=r)
                target[i] = generate_target(target[i], tpts[i]-1, self.sigma,
                                            label_type=self.label_type)
            else:
                raise ValueError(f"Invalid landmark coordinate at index {i}: {tpts[i]}")
        
        # Reassign landmarks based on learned direction (if enabled during training)
        if self.is_train:
            if self.reassign:
                tpts = classify_direction(tpts, self.d_vect)
                tpts = tpts.reshape(-1, 2)
                target = np.zeros((nparts, self.output_size[0], self.output_size[1])) 
                for i in range(nparts):
                    target[i] = generate_target(target[i], tpts[i] - 1, self.sigma,
                                                label_type=self.label_type)
        
        img = img.astype(np.float32)

        # Normalize image
        img = (img / 255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        target = torch.Tensor(target)
        tpts = torch.Tensor(tpts)
        center = torch.Tensor(center)

        meta = {'index': idx, 'center': center, 'scale': scale,
                'pts': torch.Tensor(pts), 'tpts': tpts}

        return img, target, meta

def determine_direction(pts_arr, do_plot=True):
    """
    Determine landmark direction using Gaussian Mixture Model.
    
    Args:
        pts_arr: Array of landmark points
        do_plot: If True, save diagnostic plot to 'plots/' directory
    
    Returns:
        GMM cluster means representing landmark direction
    """
    if np.isnan(pts_arr).any():
        print("Warning: The array contains NaN values.")
        nan_positions = np.argwhere(np.isnan(pts_arr))
        nan_count = np.isnan(pts_arr).sum()
        print(f"NaN positions: {nan_positions}")
        print(f"Number of NaN values: {nan_count}")
    
    gmm = GaussianMixture(n_components=2)
    gmm.fit(pts_arr)
    
    if do_plot:
        plt.scatter(pts_arr[::2, 0], pts_arr[::2, 1], alpha=1)
        plt.scatter(pts_arr[1::2, 0], pts_arr[1::2, 1], color='r', alpha=1)
        plt.plot(gmm.means_[:, 0], gmm.means_[:, 1], color='k')
        # Save diagnostic plot (requires 'plots/' directory to exist)
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/fetal_landmark_hrnet_w18_determine_direction_plots.png', dpi=600)
        plt.close()
    
    return gmm.means_

def classify_direction(pts_arr, d_pts):
    """
    Classify and reorder landmark points based on learned direction.
    
    Args:
        pts_arr: Array of landmark points to classify
        d_pts: Direction vector from GMM cluster means
    
    Returns:
        Reordered landmark points consistent with learned direction
    """
    d_vector = d_pts[1, :] - d_pts[0, :]
    ap = pts_arr
    pt_part1 = np.dot(d_vector[np.newaxis, :], ap.T) / np.linalg.norm(d_vector)
    pts_class = pt_part1.flatten()
    pts_class = pts_class.reshape(-1, 2)
    pts_class = np.stack((np.argmin(pts_class, axis=1), np.argmax(pts_class, axis=1)), axis=1)
    
    pts_remap = pts_class + (np.arange(pts_class.shape[0]) * 2)[:, np.newaxis]
    ret_pts = pts_arr[pts_remap]
    
    return ret_pts

if __name__ == '__main__':

    pass
