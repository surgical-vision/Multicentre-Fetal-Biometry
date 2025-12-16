# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# Modified by Netanell Avisdris and Chiara Di Vece (chiara.divece.20@ucl.ac.uk)
# ------------------------------------------------------------------------------
"""
Fetal Biometry Dataset for Landmark Detection.

This module implements the FetalLandmarks dataset class for training and testing
fetal biometry landmark detection models. It supports:

- Multiple anatomical structures (brain, abdomen, femur)
- Multiple metrics per anatomy (BPD, OFD, TAD, APAD, FL)
- Direction-of-Diameter (DOD) reassignment for consistent landmark ordering
- Data augmentation (flipping, rotation, scaling)
- Heatmap generation for landmark localization

The dataset handles landmark identity ambiguity by using a learned direction vector
(d_vect) to consistently order landmark pairs across all samples.
"""

import os
import random
import torch
import torch.utils.data as data
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.mixture import GaussianMixture

from ..utils.transforms import fliplr_joints, crop, generate_target, transform_pixel, get_transform


def _flip_x_only(pts: np.ndarray, width: int) -> np.ndarray:
    """
    Horizontal flip for points WITHOUT swapping landmark identities.
    When DOD reassign is enabled, we want DOD to be the only authority for pt1/pt2.
    pts: (N, 2) [x, y]
    """
    out = pts.copy()
    out[:, 0] = width - out[:, 0]
    return out


def _default_ref_for_metric(metric: str) -> np.ndarray:
    """
    Choose a reference axis for consistent sign of the learned direction.
    You train each metric separately, so this is safe and meaningful.

    - BPD, TAD, FL are typically "horizontal" diameters -> x-axis
    - OFD, APAD are typically "vertical" diameters -> y-axis
    """
    m = (metric or "").upper()
    if m in {"OFD", "APAD"}:
        return np.array([0.0, 1.0], dtype=np.float32)
    return np.array([1.0, 0.0], dtype=np.float32)


def _transform_pixel_float(pt, center, scale, output_size, invert=0, rot=0) -> np.ndarray:
    """
    Float-precision version of transform_pixel (keeps subpixel coords).
    This is ONLY used for DOD reassignment decisions to avoid ties caused by integer rounding.
    It does NOT change the heatmap generation path.
    """
    t = get_transform(center, scale, output_size, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1.0, pt[1] - 1.0, 1.0], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(np.float32) + 1.0


class FetalLandmarks(data.Dataset):
    """
    Dataset class for fetal biometry landmark detection.
    
    This dataset loads fetal ultrasound images and their corresponding landmark
    annotations. It supports Direction-of-Diameter (DOD) reassignment to ensure
    consistent landmark ordering across samples.
    
    Args:
        cfg: Configuration object containing dataset parameters
        is_train: Whether this is training (True) or test (False) dataset
        transform: Optional transform to apply to images (not currently used)
        d_vect: Direction vector for DOD reassignment. If None and is_train=True,
                it will be learned from the training data. If None and is_train=False,
                it must be provided (raises ValueError if REASSIGN=True).
    """
    def __init__(self, cfg, is_train=True, transform=None, d_vect=None):
        # Specify annotation file for dataset (CSV format)
        self.csv_file = cfg.DATASET.TRAINSET if is_train else cfg.DATASET.TESTSET

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
        self.landmarks_frame = pd.read_csv(self.csv_file, header=0, sep=',', index_col=False)

        # Drop common index columns (robust to CSVs saved with an extra index)
        if 'index' in self.landmarks_frame.columns:
            self.landmarks_frame.drop('index', axis=1, inplace=True)
        elif len(self.landmarks_frame.columns) > 0 and str(self.landmarks_frame.columns[0]).lower() == 'index':
            self.landmarks_frame.drop(self.landmarks_frame.columns[0], axis=1, inplace=True)
        elif len(self.landmarks_frame.columns) > 0 and str(self.landmarks_frame.columns[0]).startswith('Unnamed'):
            self.landmarks_frame.drop(self.landmarks_frame.columns[0], axis=1, inplace=True)

        # Define columns based on anatomy/metrics
        # CSV format: image_name, scale, center_w, center_h, [landmark columns...]
        # Landmark columns are organized as: metric1_x1, metric1_y1, metric1_x2, metric1_y2, ...
        # Store column indices so __getitem__ uses the same columns
        start_col, end_col = 4, 8  # Default (first metric)
        if self.anatomy == 'brain':
            if self.metrics == 'OFD':
                start_col, end_col = 4, 8  # OFD is first metric for brain
            elif self.metrics == 'BPD':
                start_col, end_col = 8, 12  # BPD is second metric for brain
        elif self.anatomy == 'abdomen':
            if self.metrics == 'TAD':
                start_col, end_col = 4, 8  # TAD is first metric for abdomen
            elif self.metrics == 'APAD':
                start_col, end_col = 8, 12  # APAD is second metric for abdomen
        elif self.anatomy == 'femur':
            start_col, end_col = 4, 8  # FL is the only metric for femur

        self.start_col = start_col
        self.end_col = end_col

        # Clean data
        image_name_mask = self.landmarks_frame.iloc[:, 0].isna()
        landmark_cols = self.landmarks_frame.columns[self.start_col:self.end_col]
        landmark_mask = (self.landmarks_frame[landmark_cols] < 0) | (self.landmarks_frame[landmark_cols].isna())
        landmark_mask = landmark_mask.any(axis=1)

        mask = image_name_mask | landmark_mask
        self.landmarks_frame = self.landmarks_frame[~mask].reset_index(drop=True)

        # Ensure image_name column is string type
        self.landmarks_frame.iloc[:, 0] = self.landmarks_frame.iloc[:, 0].astype(str)

        # --- DIRECTION VECTOR LOGIC (DOD Reassignment) ---
        # Goal: Learn direction ONLY from training data, then reuse the SAME direction for val/test.
        # The direction vector (d_vect) is used to consistently order landmark pairs
        # by projecting landmarks onto a learned direction and ordering them accordingly.
        # This handles the ambiguity where landmark identities might be swapped.
        if self.reassign:
            if d_vect is not None:
                # IMPORTANT: Do NOT canonicalize here; preserve exactly what was learned/saved.
                # This ensures consistency between training and testing.
                self.d_vect = np.array(d_vect, dtype=np.float32)
            elif is_train:
                # Learn direction vector from training data
                landmarks = np.array(
                    self.landmarks_frame.iloc[:, self.start_col:self.end_col].values,
                    dtype=np.float32
                ).reshape(-1, 2)  # Reshape to [N, 2] where N is total number of landmarks

                # Stabilize sign/orientation of the learned direction using a reference axis
                # This ensures consistent orientation across different training runs
                ref = _default_ref_for_metric(self.metrics)
                self.d_vect = determine_direction(landmarks, ref=ref, do_plot=False)
            else:
                raise ValueError(
                    "REASSIGN=True but no d_vect provided for val/test. "
                    "Pass the training-learned d_vect into this dataset."
                )
        else:
            # DOD reassignment is disabled
            self.d_vect = None

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        image_name_val = self.landmarks_frame.iloc[idx, 0]
        if pd.isna(image_name_val):
            raise ValueError(f"Invalid image name at index {idx}: NaN value found")
        image_name = str(image_name_val)

        image_path = os.path.join(self.data_root, image_name)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        scale = float(self.landmarks_frame.iloc[idx, 1])
        center_w = float(self.landmarks_frame.iloc[idx, 2])
        center_h = float(self.landmarks_frame.iloc[idx, 3])
        center = torch.Tensor([center_w, center_h])

        # Extract points using the SAME columns as in __init__
        pts = self.landmarks_frame.iloc[idx, self.start_col:self.end_col].values
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)

        scale *= 1.7
        nparts = pts.shape[0]

        img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)

        r = 0.0
        if self.is_train:
            scale = scale * random.uniform(1 - self.scale_factor, 1 + self.scale_factor)
            r = random.uniform(-self.rot_factor, self.rot_factor) if random.random() <= 0.6 else 0.0

            if random.random() <= 0.5 and self.flip:
                img = np.fliplr(img)

                # If DOD is enabled: do NOT swap indices during flip.
                # Let DOD be the only identity canonicalizer.
                if self.reassign:
                    pts = _flip_x_only(pts, width=img.shape[1])
                else:
                    pts = fliplr_joints(pts, width=img.shape[1], dataset='FETAL')

                center[0] = img.shape[1] - center[0]

        # Image crop/warp (points are transformed separately via transform_pixel)
        img = crop(img, center, scale, self.input_size, rot=r)

        # ---- Compute heatmap-space points FIRST (rotation-aware) ----
        # Keep the existing +1 / -1 convention exactly as in the original code.
        # We compute:
        #   - tpts_int: int coords for target heatmap generation (unchanged behaviour)
        #   - tpts_flt: float coords for DOD ordering (prevents ties from integer rounding)
        tpts_int = pts.copy()
        tpts_flt = pts.copy()

        for i in range(nparts):
            if tpts_int[i, 1] >= 0:
                tpts_int[i, 0:2] = transform_pixel(
                    tpts_int[i, 0:2] + 1, center, scale, self.output_size, rot=r
                )
                tpts_flt[i, 0:2] = _transform_pixel_float(
                    tpts_flt[i, 0:2] + 1, center, scale, self.output_size, rot=r
                )

        # ---- DOD reassignment in HEATMAP space (rot-aware), tie-safe ----
        if self.reassign:
            if self.d_vect is None:
                raise ValueError("REASSIGN=True but d_vect is None")

            # Transform the two prototype points into the SAME heatmap coordinate system (float precision)
            d0 = _transform_pixel_float(self.d_vect[0] + 1, center, scale, self.output_size, rot=r)
            d1 = _transform_pixel_float(self.d_vect[1] + 1, center, scale, self.output_size, rot=r)
            d_vec = (d1 - d0).astype(np.float32)
            denom = float(np.linalg.norm(d_vec) + 1e-12)

            # Project float heatmap-space points and reorder within each pair.
            # IMPORTANT: use a deterministic tie-breaker (<=) so we never map both points to the same index.
            proj = (tpts_flt @ d_vec) / denom  # (N,)

            if proj.shape[0] % 2 != 0:
                raise ValueError(f"Expected even number of points (pairs). Got N={proj.shape[0]}")

            proj_pairs = proj.reshape(-1, 2)  # (num_pairs, 2)

            # Tie-safe ordering:
            # - if proj0 <= proj1 -> keep [0,1]
            # - else -> swap [1,0]
            keep = (proj_pairs[:, 0] <= proj_pairs[:, 1])

            order_in_pair = np.zeros((proj_pairs.shape[0], 2), dtype=np.int64)
            order_in_pair[keep, 0] = 0
            order_in_pair[keep, 1] = 1
            order_in_pair[~keep, 0] = 1
            order_in_pair[~keep, 1] = 0

            base = (np.arange(order_in_pair.shape[0]) * 2)[:, None]
            remap = (order_in_pair + base).astype(np.int64).flatten()

            # Apply the SAME remap to:
            #  - image-space pts (used as GT in evaluation)
            #  - heatmap-space tpts_int (used for targets)
            #  - float heatmap-space tpts_flt (debug/consistency)
            pts = pts[remap].reshape(-1, 2)
            tpts_int = tpts_int[remap].reshape(-1, 2)
            tpts_flt = tpts_flt[remap].reshape(-1, 2)

        # ---- Now generate target heatmaps from reordered tpts_int ----
        target = np.zeros((nparts, self.output_size[0], self.output_size[1]), dtype=np.float32)
        for i in range(nparts):
            if tpts_int[i, 1] >= 0:
                target[i] = generate_target(
                    target[i], tpts_int[i] - 1, self.sigma, label_type=self.label_type
                )

        img = img.astype(np.float32)
        img = (img / 255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1])

        target_t = torch.Tensor(target)
        tpts_t = torch.Tensor(tpts_int)
        center_t = torch.Tensor(center)

        meta = {
            'index': idx,
            'center': center_t,
            'scale': scale,
            'pts': torch.Tensor(pts),   # image-space pts, reordered
            'tpts': tpts_t              # heatmap-space pts (same convention as before), reordered
        }

        return img, target_t, meta


def determine_direction(pts_arr: np.ndarray, ref: np.ndarray = None, do_plot=True) -> np.ndarray:
    """
    Determine direction vector for DOD reassignment using Gaussian Mixture Model.
    
    Fits a 2-component GMM to landmark positions to find two clusters representing
    the two landmark types (e.g., left and right endpoints of a diameter).
    Returns the means of these clusters as the direction vector.
    
    Args:
        pts_arr: Array of landmark coordinates [N, 2] where N is even (pairs of landmarks)
        ref: Reference vector for consistent orientation [2]. If provided, ensures
             the direction vector has a consistent sign by checking dot product.
        do_plot: Whether to plot the GMM fit (for debugging, not currently used)
    
    Returns:
        np.ndarray: Direction vector as two prototype points [2, 2]
                   representing the two landmark clusters
    """
    pts_arr = np.asarray(pts_arr, dtype=np.float32)

    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(pts_arr)
    means = gmm.means_.astype(np.float32)

    if ref is not None:
        ref = np.asarray(ref, dtype=np.float32)
        d = means[1] - means[0]
        if float(np.dot(d, ref)) < 0.0:
            means = means[::-1].copy()

    return means


# NOTE:
# classify_direction() is kept for backward compatibility / debugging,
# but with the current implementation (DOD in heatmap space), we no longer use it inside __getitem__.
def classify_direction(pts_arr: np.ndarray, d_pts: np.ndarray) -> np.ndarray:
    """
    Canonicalize identity of point pairs using a direction defined by two prototype points.
    
    This function reorders landmarks within each pair based on their projection onto
    a learned direction vector. It's kept for backward compatibility but is not
    currently used in the main dataset pipeline (which uses heatmap-space reassignment).
    
    Args:
        pts_arr: Landmark coordinates (N, 2) where N is even and pairs are consecutive
        d_pts: Direction vector defined by two prototype points (2, 2)
    
    Returns:
        np.ndarray: Reordered landmarks with consistent identity within each pair
    """
    pts_arr = np.asarray(pts_arr, dtype=np.float32)
    d_pts = np.asarray(d_pts, dtype=np.float32)

    if d_pts.shape != (2, 2):
        raise ValueError(f"d_pts must have shape (2,2), got {d_pts.shape}")
    if pts_arr.ndim != 2 or pts_arr.shape[1] != 2:
        raise ValueError(f"pts_arr must have shape (N,2), got {pts_arr.shape}")
    if pts_arr.shape[0] % 2 != 0:
        raise ValueError(f"pts_arr must contain an even number of points (pairs). Got N={pts_arr.shape[0]}")

    d_vector = d_pts[1, :] - d_pts[0, :]
    denom = float(np.linalg.norm(d_vector) + 1e-12)

    proj = np.dot(d_vector[np.newaxis, :], pts_arr.T) / denom  # (1, N)
    proj = proj.flatten()                                      # (N,)

    proj_pairs = proj.reshape(-1, 2)  # (num_pairs, 2)

    order_in_pair = np.stack(
        (np.argmin(proj_pairs, axis=1), np.argmax(proj_pairs, axis=1)),
        axis=1
    )  # (num_pairs, 2)

    base = (np.arange(order_in_pair.shape[0]) * 2)[:, np.newaxis]
    remap = order_in_pair + base

    ret_pts = pts_arr[remap.flatten()].reshape(pts_arr.shape)
    return ret_pts
