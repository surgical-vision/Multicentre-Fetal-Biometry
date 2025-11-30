#!/bin/bash

'''
Author: Chiara Di Vece (chiara.divece.20@ucl.ac.uk)
Date: 2024-11-19    

This script trains all models for all anatomical structures and datasets.
It saves the checkpoints to the output directory, with the model name as the filename.

Usage:
./run_all_training.sh
'''

# GPU selection (default: GPU 0)
# Override by setting CUDA_VISIBLE_DEVICES before running this script
# Example: CUDA_VISIBLE_DEVICES=1 ./run_all_training.sh
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0
    echo "Using default GPU: $CUDA_VISIBLE_DEVICES"
else
    echo "Using GPU(s): $CUDA_VISIBLE_DEVICES"
fi
echo ""

# Define the list of datasets and anatomical structures
datasets=("FP" "UCL" "MULTICENTRE")
structures=("brain" "femur" "abdomen")

# Add HC18 only for brain
datasets_brain=("FP" "HC18" "UCL" "MULTICENTRE")

# Define metrics for each anatomy
declare -A anatomy_metrics
anatomy_metrics[brain]="BPD OFD"
anatomy_metrics[femur]="FL"
anatomy_metrics[abdomen]="APAD TAD"

# Base directories
# This script runs from the Multicentre-Fetal-Biometry repository root
# All paths are relative to this repository
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$SCRIPT_DIR}"
EXPERIMENTS_DIR="$BASE_DIR/experiments/fetal"
OUTPUT_DIR="$BASE_DIR/output/FETAL"
LOGS_DIR="$OUTPUT_DIR/training_logs"

# Create logs directory if it doesn't exist
mkdir -p "$LOGS_DIR"

echo "================================================================================"
echo "TRAINING ALL MODELS"
echo "================================================================================"
echo "Base directory: $BASE_DIR"
echo "Experiments directory: $EXPERIMENTS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Logs directory: $LOGS_DIR"
echo ""

# Function to clean up intermediate checkpoints
cleanup_checkpoints() {
    local MODEL_DIR=$1
    local MODEL_NAME=$2
    
    echo "  Cleaning up intermediate checkpoints in $MODEL_DIR..."
    
    # Keep only these files:
    # - checkpoint_199.pth (last epoch checkpoint)
    # - current_pred.pth (predictions from last epoch)
    # - model_best.pth (best model checkpoint - lowest validation NME)
    # - final_state.pth (final model state - used by run_all_tests.sh)
    
    # Remove all checkpoint_*.pth except checkpoint_199.pth
    find "$MODEL_DIR" -name "checkpoint_*.pth" ! -name "checkpoint_199.pth" -type f -delete 2>/dev/null
    
    # Count remaining files
    local remaining=$(ls -1 "$MODEL_DIR"/*.pth 2>/dev/null | wc -l)
    echo "  ✓ Cleanup complete. Remaining .pth files: $remaining"
}

# Counter for tracking progress
total_models=0
completed_models=0

# Count total models to train (each anatomy has multiple metrics)
for STRUCTURE in "${structures[@]}"; do
    metrics=(${anatomy_metrics[$STRUCTURE]})
    num_metrics=${#metrics[@]}
    
    if [ "$STRUCTURE" == "brain" ]; then
        total_models=$((total_models + ${#datasets_brain[@]} * num_metrics))
    else
        total_models=$((total_models + ${#datasets[@]} * num_metrics))
    fi
done

echo "Total models to train: $total_models"
echo ""

# Loop over each anatomical structure
for STRUCTURE in "${structures[@]}"
do
    echo "================================================================================"
    echo "TRAINING STRUCTURE: $STRUCTURE"
    echo "================================================================================"
    echo ""
    
    # Use different dataset list for brain (includes HC18)
    if [ "$STRUCTURE" == "brain" ]; then
        CURRENT_DATASETS=("${datasets_brain[@]}")
    else
        CURRENT_DATASETS=("${datasets[@]}")
    fi
    
    # Get metrics for this anatomy
    metrics=(${anatomy_metrics[$STRUCTURE]})
    
    # Loop over each dataset
    for DATASET in "${CURRENT_DATASETS[@]}"
    do
        # Loop over each metric for this anatomy
        for METRIC in "${metrics[@]}"
        do
            completed_models=$((completed_models + 1))
            
            # Construct the paths for the configuration file
            CFG_FILE="$EXPERIMENTS_DIR/fetal_landmark_hrnet_w18_${DATASET}_${STRUCTURE}_${METRIC}.yaml"
            MODEL_NAME="fetal_landmark_hrnet_w18_${DATASET}_${STRUCTURE}_${METRIC}"
            MODEL_DIR="$OUTPUT_DIR/$MODEL_NAME"
            LOG_FILE="$LOGS_DIR/${MODEL_NAME}_train.log"
            
            echo "--------------------------------------------------------------------------------"
            echo "[$completed_models/$total_models] Training: $MODEL_NAME"
            echo "--------------------------------------------------------------------------------"
            echo "  Configuration: $CFG_FILE"
            echo "  Output directory: $MODEL_DIR"
            echo "  Log file: $LOG_FILE"
            echo ""
            
            # Check if the configuration file exists
            if [ ! -f "$CFG_FILE" ]; then
                echo "  ❌ Configuration file not found: $CFG_FILE"
                echo "  Skipping..."
                echo ""
                continue
            fi
            
            # Run the training script
            echo "  Starting training..."
            python tools/train.py --cfg "$CFG_FILE" > "$LOG_FILE" 2>&1
            
            if [ $? -eq 0 ]; then
                echo "  ✓ Training completed successfully"
                
                # Clean up intermediate checkpoints
                if [ -d "$MODEL_DIR" ]; then
                    cleanup_checkpoints "$MODEL_DIR" "$MODEL_NAME"
                fi
            else
                echo "  ❌ Training failed. Check log file: $LOG_FILE"
            fi
            
            echo ""
        done
    done
done

echo "================================================================================"
echo "TRAINING SUMMARY"
echo "================================================================================"
echo "Total models trained: $completed_models/$total_models"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "Training logs: $LOGS_DIR"
echo ""
echo "Intermediate checkpoints have been removed to save space."
echo "Kept files per model:"
echo "  - checkpoint_199.pth (last epoch checkpoint)"
echo "  - current_pred.pth (predictions from last epoch)"
echo "  - model_best.pth (best checkpoint - lowest validation NME)"
echo "  - final_state.pth (final model state - used by run_all_tests.sh)"
echo ""
echo "================================================================================"
echo "TRAINING COMPLETE"
echo "================================================================================"