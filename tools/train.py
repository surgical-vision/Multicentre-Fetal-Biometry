# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# Modified by Netanell Avisdris and Chiara Di Vece (chiara.divece.20@ucl.ac.uk)
# ------------------------------------------------------------------------------
"""
Training script for Fetal Biometry Landmark Detection.

This script trains HRNet models for fetal biometry landmark detection.
It supports:
- Multiple loss functions (MSE, FUSE, FUSEV2)
- Direction-of-Diameter (DOD) reassignment for consistent landmark ordering
- Checkpoint saving and resuming
- TensorBoard logging
- Learning rate scheduling

Usage:
    python tools/train.py --cfg <config_file>

The script saves checkpoints, best models, and final model states to the output directory.
"""

import os
import warnings
warnings.filterwarnings("ignore")
# Set default GPU if not specified in environment
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pprint
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lib.models as models
from lib.config import config, update_config
from lib.datasets import get_dataset
from lib.core import function
from lib.utils import utils
import torch.nn.functional as F

def parse_args():
    """
    Parse command-line arguments for training.
    
    Returns:
        argparse.Namespace: Parsed arguments containing:
            - cfg: Path to experiment configuration YAML file
    """
    parser = argparse.ArgumentParser(description='Train Fetal Biometry Landmark Detection')
    parser.add_argument('--cfg', help='experiment configuration filename', required=True, type=str)
    args = parser.parse_args()
    update_config(config, args)
    return args

def FuseLoss(output, target):
    """
    Fused loss function combining heatmap MSE and summed heatmap MSE.
    
    This loss encourages the model to predict accurate individual heatmaps
    as well as accurate summed heatmaps (which represent the overall measurement).
    
    Args:
        output: Predicted heatmaps [batch, num_joints, H, W]
        target: Target heatmaps [batch, num_joints, H, W]
    
    Returns:
        torch.Tensor: Combined loss value
    """
    output_fl = torch.sum(output, axis=1)  # Sum across joints dimension
    target_fl = torch.sum(target, axis=1)
    return F.mse_loss(output, target, reduction='mean') + F.mse_loss(output_fl, target_fl, reduction='mean')

def FuseLossV2(output, target):
    """
    Variant of fused loss with reduced weight on summed heatmap term.
    
    This is a modified version of FuseLoss with a 0.5 weight on the summed
    heatmap term, giving less emphasis to the global constraint.
    
    Args:
        output: Predicted heatmaps [batch, num_joints, H, W]
        target: Target heatmaps [batch, num_joints, H, W]
    
    Returns:
        torch.Tensor: Combined loss value
    """
    output_fl = torch.sum(output, axis=1)  # Sum across joints dimension
    target_fl = torch.sum(target, axis=1)
    return F.mse_loss(output, target, reduction='mean') + 0.5*F.mse_loss(output_fl, target_fl, reduction='mean')

def main():
    """
    Main training function.
    
    This function:
    1. Initializes model, optimizer, and loss function
    2. Handles checkpoint resuming
    3. Initializes datasets with direction vector (d_vect) for DOD reassignment
    4. Trains the model for specified number of epochs
    5. Validates after each epoch and saves best model
    6. Saves final model state with d_vect for inference
    """
    args = parse_args()
    logger, final_output_dir, tb_log_dir = utils.create_logger(config, args.cfg, 'train')
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # Configure CuDNN for optimal performance
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # Initialize model and TensorBoard writer
    model = models.get_face_alignment_net(config)
    writer_dict = {'writer': SummaryWriter(log_dir=tb_log_dir), 'train_global_steps': 0, 'valid_global_steps': 0}

    # Wrap model with DataParallel for multi-GPU training
    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # Select loss function based on configuration
    if config.TRAIN.CRITERION == 'MSE':
        criterion = torch.nn.MSELoss(reduction='mean').cuda()
    elif config.TRAIN.CRITERION == 'FUSE':
        criterion = FuseLoss
    elif config.TRAIN.CRITERION == 'FUSEV2':
        criterion = FuseLossV2
    else:
        raise ValueError('Criterion not defined')

    # Initialize optimizer and training state
    optimizer = utils.get_optimizer(config, model)
    best_nme = 100  # Track best validation NME (lower is better)
    last_epoch = config.TRAIN.BEGIN_EPOCH
    
    # Resume from checkpoint if enabled
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir, 'latest.pth')
        if os.path.islink(model_state_file) or os.path.exists(model_state_file):
            checkpoint = torch.load(model_state_file)
            last_epoch = checkpoint['epoch']
            best_nme = checkpoint['best_nme']
            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            learned_d_vect = checkpoint.get('d_vect', None)
            if learned_d_vect is not None:
                logger.info(f"Loaded DOD direction from checkpoint: {learned_d_vect}")
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found")

    # Initialize learning rate scheduler
    if isinstance(config.TRAIN.LR_STEP, list):
        # Multi-step scheduler: reduce LR at specified epochs
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR, last_epoch-1)
    else:
        # Step scheduler: reduce LR every N epochs
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR, last_epoch-1)
    
    # --- DATASET INITIALIZATION WITH DIRECTION PASSING ---
    # The direction vector (d_vect) ensures consistent landmark ordering
    # across training and validation sets using Direction-of-Diameter (DOD) reassignment
    dataset_type = get_dataset(config)

    # 1) Initialize Training Set:
    # - If resuming and checkpoint contains d_vect, reuse it for consistency
    # - Otherwise, learn it from the training set
    if 'learned_d_vect' in locals() and learned_d_vect is not None:
        train_dataset = dataset_type(config, is_train=True, d_vect=learned_d_vect)
    else:
        train_dataset = dataset_type(config, is_train=True)
        learned_d_vect = train_dataset.d_vect
    logger.info(f"Using DOD Direction Vector: {learned_d_vect}")
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY)

    # 2) Initialize Validation Set with the learned direction
    # This ensures validation uses the same landmark ordering convention as training
    val_dataset = dataset_type(config, is_train=False, d_vect=learned_d_vect)
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=False,  # Don't shuffle validation data
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    # Training loop
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        # Train for one epoch
        function.train(config, train_loader, model, criterion, optimizer, epoch, writer_dict)
        lr_scheduler.step()  # Update learning rate
        
        # Validate on validation set
        nme, predictions = function.validate(config, val_loader, model, criterion, epoch, writer_dict)

        # Track best model based on validation NME
        is_best = nme < best_nme
        best_nme = min(nme, best_nme)

        # Save checkpoint
        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        print("best:", is_best)
        utils.save_checkpoint(
            {"state_dict": model.module.state_dict(),
             "epoch": epoch + 1,
             "best_nme": best_nme,
             "optimizer": optimizer.state_dict(),
              "d_vect": learned_d_vect,  # Save d_vect for consistent inference
             }, predictions, is_best, final_output_dir, 'checkpoint_{}.pth'.format(epoch))

    # Save final model state (used for inference)
    # This includes model weights and d_vect in a portable format
    final_model_state_file = os.path.join(final_output_dir, 'final_state.pth')
    logger.info('saving final model state + d_vect to {}'.format(final_model_state_file))

    final_payload = {
        # Weights only (clean, not DataParallel wrapper)
        "state_dict": model.module.state_dict(),
        # Store d_vect in a portable form (numpy array, not tensor)
        "d_vect": (learned_d_vect.detach().cpu().numpy()
                if isinstance(learned_d_vect, torch.Tensor)
                else learned_d_vect),
        "epoch": epoch + 1,
        "best_nme": best_nme,
    }

    torch.save(final_payload, final_model_state_file)

    # Close TensorBoard writer
    writer_dict['writer'].close()

if __name__ == '__main__':
    main()