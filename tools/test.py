# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# Modified by Netanell Avisdris and Chiara Di Vece (chiara.divece.20@ucl.ac.uk)
# ------------------------------------------------------------------------------
"""
Test script for Fetal Biometry Landmark Detection.

This script loads a trained model and evaluates it on a test dataset.
It handles model weight loading (with support for DataParallel-wrapped models),
direction vector (d_vect) loading for Direction-of-Diameter (DOD) reassignment,
and computes evaluation metrics including Normalized Mean Error (NME).

Usage:
    python tools/test.py --cfg <config_file> --model-file <model_file>

The script saves predictions to the output directory for further analysis.
"""

import os
import pprint
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import lib.models as models
from lib.config import config, update_config
from lib.utils import utils
from lib.datasets import get_dataset
from lib.core import function


def parse_args():
    """
    Parse command-line arguments for testing.
    
    Returns:
        argparse.Namespace: Parsed arguments containing:
            - cfg: Path to experiment configuration YAML file
            - model-file: Path to trained model checkpoint file
    """
    parser = argparse.ArgumentParser(description='Test Fetal Biometry Landmark Detection')
    parser.add_argument('--cfg', help='experiment configuration filename', required=True, type=str)
    parser.add_argument('--model-file', help='model parameters', required=True, type=str)
    args = parser.parse_args()
    update_config(config, args)
    return args


def _load_weights_into_model(model_dp: nn.DataParallel, weights: dict, logger):
    """
    Loads weights into a DataParallel-wrapped model robustly whether keys are
    prefixed with 'module.' or not.
    """
    if not isinstance(weights, dict):
        raise TypeError(f"Expected weights dict/OrderedDict, got {type(weights)}")

    has_module_prefix = any(k.startswith("module.") for k in weights.keys())

    try:
        if has_module_prefix:
            logger.info("Detected 'module.'-prefixed keys. Loading into DataParallel wrapper (model.load_state_dict).")
            model_dp.load_state_dict(weights, strict=True)
        else:
            logger.info("Detected non-prefixed keys. Loading into underlying module (model.module.load_state_dict).")
            model_dp.module.load_state_dict(weights, strict=True)
    except RuntimeError as e:
        # If strict load fails, report missing/unexpected keys clearly.
        logger.error(f"Weight loading failed: {e}")
        raise


def main():
    """
    Main testing function.
    
    This function:
    1. Parses command-line arguments and loads configuration
    2. Initializes the model and loads trained weights
    3. Handles direction vector (d_vect) for DOD reassignment
    4. Creates test data loader
    5. Runs inference and computes evaluation metrics
    6. Saves predictions to disk
    """
    args = parse_args()
    logger, final_output_dir, tb_log_dir = utils.create_logger(config, args.cfg, 'test')
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # Configure CuDNN for optimal performance
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # Disable weight initialization since we're loading a trained model
    config.defrost()
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()

    # Initialize model and wrap with DataParallel for multi-GPU support
    model = models.get_face_alignment_net(config)
    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # Load model checkpoint
    payload = torch.load(args.model_file, map_location="cpu")

    # ---- Extract weights + d_vect if present ----
    # The checkpoint may contain:
    # - A full checkpoint dict with 'state_dict' and optional 'd_vect'
    # - Or just the weights (OrderedDict)
    learned_d_vect = None

    if isinstance(payload, dict) and "state_dict" in payload:
        weights = payload["state_dict"]
        learned_d_vect = payload.get("d_vect", None)
        if learned_d_vect is not None:
            logger.info("Loaded d_vect from checkpoint/final_state payload.")
        else:
            logger.info("No d_vect found in checkpoint payload.")
    else:
        # weights-only file
        weights = payload
        logger.info("Loaded weights-only file (no checkpoint keys). No d_vect available here by definition.")

    # Load weights robustly (handles both DataParallel and non-DataParallel formats)
    _load_weights_into_model(model, weights, logger)

    dataset_type = get_dataset(config)

    # ---- Determine which d_vect to use ----
    # The direction vector (d_vect) is used for Direction-of-Diameter (DOD) reassignment
    # to ensure consistent landmark ordering. It should be learned during training
    # and reused during testing for consistency.
    if learned_d_vect is None:
        # Fallback: compute from training annotations if not in checkpoint
        logger.info("Computing DOD direction from training annotations (fallback)...")
        dummy_train_dataset = dataset_type(config, is_train=True)
        learned_d_vect = dummy_train_dataset.d_vect
        logger.info("Computed d_vect from training set.")
    else:
        # Ensure d_vect is a plain numpy array, not a GPU tensor
        if isinstance(learned_d_vect, torch.Tensor):
            learned_d_vect = learned_d_vect.detach().cpu().numpy()

    logger.info(f"Using Direction Vector: {learned_d_vect}")

    # Initialize test dataset with the chosen direction vector
    # This ensures consistent landmark ordering between training and testing
    test_loader = DataLoader(
        dataset=dataset_type(config, is_train=False, d_vect=learned_d_vect),
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,  # Don't shuffle test data
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    # Run inference and compute evaluation metrics
    nme, nme_mean, nme_std, predictions = function.inference(config, test_loader, model)
    
    # Save predictions for further analysis
    torch.save(predictions, os.path.join(final_output_dir, 'predictions.pth'))


if __name__ == '__main__':
    main()