# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn), Tianheng Cheng(tianhengcheng@gmail.com)
# Further modified by Netanell Avisdris and Chiara Di Vece (chiara.divece.20@ucl.ac.uk)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path
import shutil

import torch
import torch.optim as optim


def create_logger(cfg, cfg_name, phase='train'):
    """
    Create logger and output directories for training/testing.
    
    Sets up:
    - Output directory for checkpoints and predictions
    - Log file for text logging
    - TensorBoard log directory for visualization
    
    Args:
        cfg: Configuration object
        cfg_name: Configuration filename (used to name output directories)
        phase: 'train' or 'test' (used in log filename)
    
    Returns:
        tuple: (logger, output_dir, tensorboard_log_dir)
            - logger: Python logging.Logger instance
            - output_dir: Path to output directory for checkpoints/predictions
            - tensorboard_log_dir: Path to TensorBoard log directory
    """
    root_output_dir = Path(cfg.OUTPUT_DIR)
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]  # Remove path and extension

    # Create experiment-specific output directory
    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    # Create log file with timestamp
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file

    # Configure logging to both file and console
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file), format=head)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Also log to console
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    # Create TensorBoard log directory
    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def get_optimizer(cfg, model):
    """
    Create optimizer based on configuration.
    
    Supports SGD, Adam, and RMSprop optimizers. Only parameters that require
    gradients are included in the optimizer.
    
    Args:
        cfg: Configuration object with TRAIN.OPTIMIZER and related parameters
        model: PyTorch model whose parameters to optimize
    
    Returns:
        torch.optim.Optimizer: Configured optimizer instance
    """
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR
        )
    elif cfg.TRAIN.OPTIMIZER == 'rmsprop':
        optimizer = optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            alpha=cfg.TRAIN.RMSPROP_ALPHA,
            centered=cfg.TRAIN.RMSPROP_CENTERED
        )
    return optimizer


def _to_serializable(x):
    """
    Convert common objects into something torch.save can handle cleanly.
    - Tensor -> CPU tensor
    - numpy -> leave as is
    - dict/list/tuple -> recursively convert tensors inside
    """
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k: _to_serializable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_serializable(v) for v in x)
    return x


def _extract_weights_for_saving(state_obj):
    """
    Accepts:
      - DataParallel model (has .module)
      - nn.Module (has .state_dict)
      - OrderedDict / dict (already weights)

    Returns a plain weights dict suitable for torch.save(...).
    """
    if state_obj is None:
        return None

    # DataParallel
    if hasattr(state_obj, "module") and hasattr(state_obj.module, "state_dict"):
        return state_obj.module.state_dict()

    # Plain nn.Module
    if hasattr(state_obj, "state_dict") and not isinstance(state_obj, dict):
        return state_obj.state_dict()

    # Already weights dict / OrderedDict
    if isinstance(state_obj, dict):
        return state_obj

    raise TypeError(f"Unsupported type for state_dict saving: {type(state_obj)}")


def _safe_write_latest(output_dir, filename, latest_name="latest.pth"):
    """
    Maintain latest.pth as a symlink if possible, otherwise copy/overwrite a real file.
    This avoids platform/filesystem issues with symlinks.
    """
    latest_path = os.path.join(output_dir, latest_name)
    target_path = os.path.join(output_dir, filename)

    # Remove existing latest
    try:
        if os.path.islink(latest_path) or os.path.exists(latest_path):
            os.remove(latest_path)
    except Exception:
        pass

    # Try symlink first
    try:
        os.symlink(target_path, latest_path)
        return
    except Exception:
        pass

    # Fallback: copy file
    try:
        shutil.copyfile(target_path, latest_path)
    except Exception:
        # Absolute last resort: do nothing
        pass


def save_checkpoint(states, predictions, is_best, output_dir, filename='checkpoint.pth'):
    """
    Save model checkpoint and related files.
    
    This function saves checkpoints in a robust manner, handling both full checkpoints
    and best model weights. It also maintains a 'latest.pth' symlink/copy for easy
    checkpoint resuming.

    Args:
        states: Dictionary containing model state, epoch, best_nme, optimizer state, d_vect, etc.
        predictions: Current predictions (saved as current_pred.pth)
        is_best: Whether this is the best model so far (based on validation NME)
        output_dir: Directory to save checkpoints
        filename: Filename for this checkpoint (e.g., 'checkpoint_50.pth')
    
    Saves:
        - <filename>: Full checkpoint dict with all training state
        - current_pred.pth: Current predictions (if provided)
        - latest.pth: Symlink (preferred) or copy (fallback) to the latest checkpoint
    
    If is_best:
        - checkpoint_best.pth: Full checkpoint dict (includes d_vect, optimizer, epoch, etc.)
        - model_best.pth: Weights-only dict (portable, no optimizer state)
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) Save checkpoint dict
    ckpt_path = os.path.join(output_dir, filename)
    torch.save(states, ckpt_path)

    # 2) Save predictions (if provided)
    preds = _to_serializable(predictions)
    if preds is not None:
        torch.save(preds, os.path.join(output_dir, 'current_pred.pth'))

    # 3) Maintain latest.pth
    _safe_write_latest(output_dir, filename, latest_name="latest.pth")

    # 4) Best artifacts
    if is_best and isinstance(states, dict) and 'state_dict' in states:
        # Full checkpoint (best)
        torch.save(states, os.path.join(output_dir, "checkpoint_best.pth"))

        # Weights-only best
        weights = _extract_weights_for_saving(states.get('state_dict'))
        if weights is not None:
            torch.save(weights, os.path.join(output_dir, "model_best.pth"))