#!/bin/bash
# ------------------------------------------------------------------------------
# Author: Chiara Di Vece (chiara.divece.20@ucl.ac.uk)
# Date: 2025-12-09
# ------------------------------------------------------------------------------
# This script runs all cross-validation tests for all anatomical structures and datasets.
#
# It performs comprehensive cross-validation testing by:
# 1. Testing each trained model on all available test datasets
# 2. Saving predictions with dataset-specific filenames (e.g., predictions_on_UCL.pth)
# 3. Enabling comparison of model performance across different datasets
#
# Prerequisites:
# - Models must be trained using run_all_training.sh
# - Configuration files must exist in experiments/fetal/
#
# Usage:
#   ./run_all_tests.sh
#
# Output:
#   Predictions are saved in output/FETAL/<model_name>/predictions_on_<dataset>.pth
# ------------------------------------------------------------------------------

# Define the list of datasets and anatomical structures
# HC18 only has brain data, so it's excluded from femur and abdomen
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

echo "================================================================================"
echo "RUNNING ALL CROSS-VALIDATION TESTS"
echo "================================================================================"
echo "Base directory: $BASE_DIR"
echo "Experiments directory: $EXPERIMENTS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

echo ""

# Loop over each anatomical structure
for STRUCTURE in "${structures[@]}"
do
    echo "================================================================================"
    echo "TESTING STRUCTURE: $STRUCTURE"
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
    
    # Loop over each metric
    for METRIC in "${metrics[@]}"
    do
        # Loop over each combination of model_dataset and cfg_dataset
        for MODEL_DATASET in "${CURRENT_DATASETS[@]}"
        do
            for CFG_DATASET in "${CURRENT_DATASETS[@]}"
            do
                # Construct the paths for the configuration file and the model file
                CFG_FILE="$EXPERIMENTS_DIR/fetal_landmark_hrnet_w18_${CFG_DATASET}_${STRUCTURE}_${METRIC}.yaml"
                MODEL_DIR="$OUTPUT_DIR/fetal_landmark_hrnet_w18_${MODEL_DATASET}_${STRUCTURE}_${METRIC}"
                MODEL_FILE="$MODEL_DIR/final_state.pth"

                # Skip if configuration file doesn't exist (safety check)
                if [ ! -f "$CFG_FILE" ]; then
                    continue
                fi

                # Skip if model file doesn't exist (safety check)
                if [ ! -f "$MODEL_FILE" ]; then
                    continue
                fi

                echo "Using configuration file: $CFG_FILE"
                echo "Using model file: $MODEL_FILE"

                # Run the test script
                echo "--------------------------------------------------------------------------------"
                echo "Testing: MODEL=$MODEL_DATASET, DATA=$CFG_DATASET, STRUCTURE=$STRUCTURE, METRIC=$METRIC"
                echo "--------------------------------------------------------------------------------"
                
                # Create a unique predictions filename to avoid overwriting
                # This ensures consistent naming even when MODEL_DATASET == CFG_DATASET
                PREDICTIONS_FILE="$MODEL_DIR/predictions_on_${CFG_DATASET}.pth"
                
                # The test.py script saves predictions to a directory based on the config file
                # (not the model directory), so we need to determine where it will be saved.
                # Note: When MODEL_DATASET == CFG_DATASET, CFG_OUTPUT_DIR == MODEL_DIR
                CFG_OUTPUT_DIR="$OUTPUT_DIR/fetal_landmark_hrnet_w18_${CFG_DATASET}_${STRUCTURE}_${METRIC}"
                PREDICTIONS_SOURCE="$CFG_OUTPUT_DIR/predictions.pth"
                
                # Run test and save predictions with dataset-specific name
                python tools/test.py --cfg "$CFG_FILE" --model-file "$MODEL_FILE"
                
                if [ $? -eq 0 ]; then
                    # Look for predictions in the config-based output directory
                    # (where test.py actually saves them)
                    # This works for both cross-validation (MODEL_DATASET != CFG_DATASET)
                    # and same-dataset testing (MODEL_DATASET == CFG_DATASET)
                    if [ -f "$PREDICTIONS_SOURCE" ]; then
                        # Move/rename predictions to dataset-specific name in model directory
                        mv "$PREDICTIONS_SOURCE" "$PREDICTIONS_FILE"
                        echo "✓ Test completed successfully"
                        echo "  Predictions saved to: predictions_on_${CFG_DATASET}.pth"
                    elif [ -f "$MODEL_DIR/predictions.pth" ]; then
                        # Fallback: check model directory (in case test.py behavior changed)
                        mv "$MODEL_DIR/predictions.pth" "$PREDICTIONS_FILE"
                        echo "✓ Test completed successfully"
                        echo "  Predictions saved to: predictions_on_${CFG_DATASET}.pth"
                    else
                        echo "✓ Test completed successfully (no predictions file generated)"
                    fi
                else
                    echo "❌ Test failed"
                fi
                echo ""
            done
        done
    done
done

echo "================================================================================"
echo "ALL TESTS COMPLETE"
echo "================================================================================"
echo ""
echo "Prediction files have been saved with dataset-specific names:"
echo "  Format: predictions_on_{DATASET}.pth"
echo "  Example: predictions_on_FP.pth, predictions_on_MULTICENTRE.pth"
echo ""
echo "Each model directory now contains predictions for all tested datasets."
echo "================================================================================"