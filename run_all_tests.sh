#!/bin/bash
'''
Author: Chiara Di Vece (chiara.divece.20@ucl.ac.uk)
Date: 2024-11-13

This script runs all cross-validation tests for all anatomical structures and datasets.
It assumes that the models have been trained using run_all_training.sh.
It saves the predictions to the output directory, with the dataset name as the filename.
It then uses the create_ucl_error_boxplots.py script to create boxplots of the errors.

Usage:
./run_all_tests.sh
'''

# Define the list of datasets and anatomical structures
datasets=("FP" "HC18" "UCL" "MULTICENTRE")
structures=("brain" "femur" "abdomen")

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
    
    # Get metrics for this anatomy
    metrics=(${anatomy_metrics[$STRUCTURE]})
    
    # Loop over each metric
    for METRIC in "${metrics[@]}"
    do
        # Loop over each combination of model_dataset and cfg_dataset
        for MODEL_DATASET in "${datasets[@]}"
        do
            for CFG_DATASET in "${datasets[@]}"
            do
                # Construct the paths for the configuration file and the model file
                CFG_FILE="$EXPERIMENTS_DIR/fetal_landmark_hrnet_w18_${CFG_DATASET}_${STRUCTURE}_${METRIC}.yaml"
                MODEL_DIR="$OUTPUT_DIR/fetal_landmark_hrnet_w18_${MODEL_DATASET}_${STRUCTURE}_${METRIC}"
                MODEL_FILE="$MODEL_DIR/final_state.pth"

                # Check if the configuration file exists
                if [ -f "$CFG_FILE" ]; then
                    echo "Using configuration file: $CFG_FILE"
                else
                    echo "Configuration file not found: $CFG_FILE"
                    continue
                fi

                # Check if the model file exists
                if [ -f "$MODEL_FILE" ]; then
                    echo "Using model file: $MODEL_FILE"
                else
                    echo "Model file not found: $MODEL_FILE"
                    continue
                fi

                # Run the test script
                echo "--------------------------------------------------------------------------------"
                echo "Testing: MODEL=$MODEL_DATASET, DATA=$CFG_DATASET, STRUCTURE=$STRUCTURE, METRIC=$METRIC"
                echo "--------------------------------------------------------------------------------"
                
                # Create a unique predictions filename to avoid overwriting
                PREDICTIONS_FILE="$MODEL_DIR/predictions_on_${CFG_DATASET}.pth"
                
                # Run test and save predictions with dataset-specific name
                python tools/test.py --cfg "$CFG_FILE" --model-file "$MODEL_FILE"
                
                if [ $? -eq 0 ]; then
                    # Rename the predictions.pth to dataset-specific name
                    if [ -f "$MODEL_DIR/predictions.pth" ]; then
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