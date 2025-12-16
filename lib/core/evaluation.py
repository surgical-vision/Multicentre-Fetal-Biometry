# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# Modified by Netanell Avisdris and Chiara Di Vece (chiara.divece.20@ucl.ac.uk)
# ------------------------------------------------------------------------------

import math
import torch
import numpy as np
from ..utils.transforms import transform_preds

EPS_INTEROCULAR = 1e-6  # avoids division-by-zero producing inf in NME


def get_preds(scores):
    """
    Extract landmark predictions from heatmap score maps.
    
    Finds the location of maximum activation in each heatmap channel,
    which corresponds to the predicted landmark location.
    
    Args:
        scores: Heatmap score maps [batch, num_joints, H, W]
    
    Returns:
        torch.Tensor: Predicted landmark coordinates [batch, num_joints, 2]
                     in heatmap space (1-indexed)
    """
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % scores.size(3) + 1
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / scores.size(3)) + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds


def compute_nme(preds, meta):
    """
    Compute Normalized Mean Error (NME) with Direction Invariance for Fetal Biometry.
    
    NME is computed as the average Euclidean distance between predicted and ground truth
    landmarks, normalized by a reference distance (interocular distance for face datasets,
    or the distance between the two landmarks for fetal biometry).
    
    For fetal biometry (2 landmarks), the metric is made direction-invariant by computing
    both standard and swapped errors and taking the minimum. This handles cases where
    landmark identities might be flipped.
    
    Args:
        preds: Predicted landmark coordinates [N, L, 2] where N is batch size, L is num landmarks
        meta: Dictionary containing:
            - 'pts': Ground truth landmarks [N, L, 2]
            - 'box_size': (for AFLW dataset) bounding box size
    
    Returns:
        np.ndarray: NME values for each sample in the batch [N]
    """
    targets = meta['pts']
    preds = preds.numpy()
    target = targets.cpu().numpy()

    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = preds[i, ], target[i, ]
        
        # --- Normalization Factor ---
        if L == 19:  # aflw
            interocular = meta['box_size'][i]
        elif L == 29:  # cofw
            interocular = np.linalg.norm(pts_gt[8, ] - pts_gt[9, ])
        elif L == 68:  # 300w
            interocular = np.linalg.norm(pts_gt[36, ] - pts_gt[45, ])
        elif L == 98:
            interocular = np.linalg.norm(pts_gt[60, ] - pts_gt[72, ])
        elif L == 2: # Fetal Biometry (Diameter)
            # Distance between the two Ground Truth points
            interocular = np.linalg.norm(pts_gt[0, ] - pts_gt[1, ])
            interocular = max(float(interocular), EPS_INTEROCULAR)
        else:
            raise ValueError('Number of landmarks is wrong')

        # --- Error Calculation ---
        if L == 2:
            # Calculate Standard Error (1->1, 2->2)
            dist_standard = np.linalg.norm(pts_pred[0] - pts_gt[0]) + \
                            np.linalg.norm(pts_pred[1] - pts_gt[1])
            
            # Calculate Swapped Error (1->2, 2->1)
            # This makes the metric invariant to left/right flipping
            dist_swapped = np.linalg.norm(pts_pred[0] - pts_gt[1]) + \
                           np.linalg.norm(pts_pred[1] - pts_gt[0])
            
            # Use the minimum distance (Best Match)
            rmse[i] = min(dist_standard, dist_swapped) / (interocular * L)
        else:
            # Standard calculation for face datasets
            rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (interocular * L)

    return rmse


def compute_l1dist(preds, meta):
    """
    Compute L1 distance (absolute difference) between predicted and ground truth measurements.
    
    For fetal biometry, this computes the absolute difference between the predicted
    and ground truth diameters/lengths. This metric is naturally direction-invariant
    since it compares lengths rather than individual landmark positions.
    
    Args:
        preds: Predicted landmark coordinates [N, L, 2]
        meta: Dictionary containing:
            - 'pts': Ground truth landmarks [N, L, 2]
    
    Returns:
        np.ndarray: L1 distance values for each sample [N] (in pixels)
    """
    targets = meta['pts']
    preds = preds.numpy()
    target = targets.cpu().numpy()

    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = preds[i, ], target[i, ]

        if L == 2:  # Fetal biometry (2 landmarks per measurement)
            # L1 is naturally invariant because it compares length vs length
            # Compute ground truth measurement length
            interocular = np.linalg.norm(pts_gt[0, ] - pts_gt[1, ])
            interocular = max(float(interocular), EPS_INTEROCULAR)
            # Compute predicted measurement length
            pred_length = np.linalg.norm(pts_pred[0, ] - pts_pred[1, ])
            # Absolute difference
            rmse[i] = np.abs(interocular - pred_length)
        else:
            raise ValueError('Number of landmarks is wrong')

    return rmse

def decode_preds(output, center, scale, res):
    """
    Decode landmark predictions from heatmaps to original image coordinates.
    
    This function:
    1. Extracts peak locations from heatmaps
    2. Refines predictions using sub-pixel localization (checking neighboring pixels)
    3. Transforms coordinates from heatmap space back to original image space
    
    Args:
        output: Heatmap predictions [batch, num_joints, H, W]
        center: Center coordinates for each image [batch, 2]
        scale: Scale factors for each image [batch]
        res: Resolution of heatmaps [H, W]
    
    Returns:
        torch.Tensor: Predicted landmarks in original image coordinates [batch, num_joints, 2]
    """
    coords = get_preds(output)  # Get initial predictions from heatmaps (float type)

    coords = coords.cpu()
    # Sub-pixel refinement: check neighboring pixels to refine landmark location
    # This improves accuracy by using gradient information from the heatmap
    for n in range(coords.size(0)):
        for p in range(coords.size(1)):
            hm = output[n][p]
            px = int(math.floor(coords[n][p][0]))
            py = int(math.floor(coords[n][p][1]))
            # Check if within valid bounds
            if (px > 1) and (px < res[0]) and (py > 1) and (py < res[1]):
                # Compute gradient from neighboring pixels
                diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1]-hm[py - 2][px - 1]])
                # Adjust prediction by 0.25 pixels in the direction of higher activation
                coords[n][p] += diff.sign() * .25
    coords += 0.5  # Convert from 1-indexed to 0-indexed
    
    preds = coords.clone()

    # Transform predictions from heatmap space back to original image space
    for i in range(coords.size(0)):
        preds[i] = transform_preds(coords[i], center[i], scale[i], res)

    # Ensure correct dimensionality
    if preds.dim() < 3:
        preds = preds.view(1, preds.size())

    return preds
