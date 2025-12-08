"""
Author: Chiara Di Vece (chiara.divece.20@ucl.ac.uk)
Date: 2024-11-30

This script creates boxplots showing the absolute error between clinically measured 
and predicted biometry for the UCL test dataset in millimeters.

It assumes that the predictions have been generated using run_all_tests.sh.

Usage:
python create_ucl_error_boxplots.py

This script should be run from the Multicentre-Fetal-Biometry repository root.
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib import font_manager
from pathlib import Path

# Get script directory to ensure correct paths
SCRIPT_DIR = Path(__file__).parent.resolve()

# Try to use Lato font if available
try:
    # Register Lato font if available
    font_dir = SCRIPT_DIR / 'fonts' / 'Lato'
    if font_dir.exists():
        font_files = font_manager.findSystemFonts(fontpaths=str(font_dir))
        for font_file in font_files:
            if 'Lato' in font_file:
                font_manager.fontManager.addfont(font_file)
        plt.rcParams['font.family'] = 'Lato'
except:
    print("Lato font not found, using default font")
    pass

# Set up plotting style
plt.rcParams['font.size'] = 14
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# Define paths relative to script location (repository root)
base_dir = SCRIPT_DIR
data_dir = base_dir / "data" / "annotations"
output_dir = base_dir / "output" / "FETAL"

# Define measurements and their properties
MEASUREMENTS = {
    'BPD': {
        'anatomy': 'brain',
        'csv_prefix': 'Head',
        'landmark_prefix': 'bpd',
        'display_name': 'BPD'
    },
    'OFD': {
        'anatomy': 'brain',
        'csv_prefix': 'Head',
        'landmark_prefix': 'ofd',
        'display_name': 'OFD'
    },
    'TAD': {
        'anatomy': 'abdomen',
        'csv_prefix': 'Abdomen',
        'landmark_prefix': 'tad',
        'display_name': 'TAD'
    },
    'APAD': {
        'anatomy': 'abdomen',
        'csv_prefix': 'Abdomen',
        'landmark_prefix': 'apad',
        'display_name': 'APAD'
    },
    'FL': {
        'anatomy': 'femur',
        'csv_prefix': 'Femur',
        'landmark_prefix': 'fl',
        'display_name': 'FL'
    }
}

# Group measurements by anatomy for plotting
MEASUREMENTS_BY_ANATOMY = {
    'Head': ['BPD', 'OFD'],
    'Abdomen': ['TAD', 'APAD'],
    'Femur': ['FL']
}

# Models to compare - organized by measurement
# Map to measurement-specific directories
MODELS = {
    'UCL': {
        'BPD': 'fetal_landmark_hrnet_w18_UCL_brain_BPD',
        'OFD': 'fetal_landmark_hrnet_w18_UCL_brain_OFD',
        'TAD': 'fetal_landmark_hrnet_w18_UCL_abdomen_TAD',
        'APAD': 'fetal_landmark_hrnet_w18_UCL_abdomen_APAD',
        'FL': 'fetal_landmark_hrnet_w18_UCL_femur_FL'
    },
    'MULTICENTRE': {
        'BPD': 'fetal_landmark_hrnet_w18_MULTICENTRE_brain_BPD',
        'OFD': 'fetal_landmark_hrnet_w18_MULTICENTRE_brain_OFD',
        'TAD': 'fetal_landmark_hrnet_w18_MULTICENTRE_abdomen_TAD',
        'APAD': 'fetal_landmark_hrnet_w18_MULTICENTRE_abdomen_APAD',
        'FL': 'fetal_landmark_hrnet_w18_MULTICENTRE_femur_FL'
    },
    'FP': {
        'BPD': 'fetal_landmark_hrnet_w18_FP_brain_BPD',
        'OFD': 'fetal_landmark_hrnet_w18_FP_brain_OFD',
        'TAD': 'fetal_landmark_hrnet_w18_FP_abdomen_TAD',
        'APAD': 'fetal_landmark_hrnet_w18_FP_abdomen_APAD',
        'FL': 'fetal_landmark_hrnet_w18_FP_femur_FL'
    },
    'HC18': {
        'BPD': 'fetal_landmark_hrnet_w18_HC18_brain_BPD',
        'OFD': 'fetal_landmark_hrnet_w18_HC18_brain_OFD',
        'TAD': None,  # HC18 only has brain data
        'APAD': None,
        'FL': None
    }
}

# Define which models to use for each measurement (in display order)
MODELS_PER_MEASUREMENT = {
    'BPD': ['FP', 'HC18', 'UCL', 'MULTICENTRE'],
    'OFD': ['FP', 'HC18', 'UCL', 'MULTICENTRE'],
    'TAD': ['FP', 'UCL', 'MULTICENTRE'],
    'APAD': ['FP', 'UCL', 'MULTICENTRE'],
    'FL': ['FP', 'UCL', 'MULTICENTRE']
}

# Display names for models
MODEL_DISPLAY_NAMES = {
    'FP': 'FP',
    'HC18': 'HC18',
    'UCL': 'UCL',
    'MULTICENTRE': 'Ours'
}


def calculate_distance_pixels(x1, y1, x2, y2):
    """Calculate Euclidean distance in pixels between two points."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def load_ucl_test_data(measurement):
    """Load UCL test data including ground truth and pixel-to-mm conversion rates."""
    csv_prefix = MEASUREMENTS[measurement]['csv_prefix']
    test_file = data_dir / "UCL" / "px_to_mm" / f"{csv_prefix}_Test.csv"
    df = pd.read_csv(test_file)
    
    return df


def load_predictions(model_name, measurement):
    """Load predictions from a trained model tested on UCL.
    
    Args:
        model_name: Name of the model (UCL, FP, MULTICENTRE, HC18)
        measurement: Measurement type (BPD, OFD, TAD, APAD, FL)
    """
    model_folder = MODELS[model_name][measurement]
    
    if model_folder is None:
        return None
    
    # All models tested on UCL should have predictions_on_UCL.pth
    pred_file = output_dir / model_folder / "predictions_on_UCL.pth"
    
    if not pred_file.exists():
        print(f"Warning: Predictions file not found: {pred_file}")
        print(f"Run run_all_tests.sh to generate predictions on UCL test set")
        return None
    
    predictions = torch.load(pred_file)
    return predictions.numpy()  # Convert to numpy array


def calculate_errors_mm(measurement, model_name='UCL'):
    """Calculate absolute errors in millimeters for a given measurement and model."""
    
    # Load ground truth data
    df = load_ucl_test_data(measurement)
    
    # Load predictions
    predictions = load_predictions(model_name, measurement)
    
    if predictions is None:
        return None
    
    # Get the landmark prefix for this measurement
    landmark_prefix = MEASUREMENTS[measurement]['landmark_prefix']
    
    # Calculate distances in pixels
    errors_mm = []
    
    for idx, row in df.iterrows():
        # Ground truth distance in pixels
        gt_dist_px = calculate_distance_pixels(
            row[f'{landmark_prefix}_1_x'], row[f'{landmark_prefix}_1_y'],
            row[f'{landmark_prefix}_2_x'], row[f'{landmark_prefix}_2_y']
        )
        
        # Predicted distance in pixels
        # predictions shape: (n_samples, n_joints, 2)
        pred_pt1 = predictions[idx, 0, :]  # First landmark
        pred_pt2 = predictions[idx, 1, :]  # Second landmark
        pred_dist_px = calculate_distance_pixels(
            pred_pt1[0], pred_pt1[1],
            pred_pt2[0], pred_pt2[1]
        )
        
        # Convert to mm
        px_to_mm = row['px_to_mm_rate']
        gt_dist_mm = gt_dist_px * px_to_mm
        pred_dist_mm = pred_dist_px * px_to_mm
                
        # Calculate absolute error
        error_mm = abs(pred_dist_mm - gt_dist_mm)
        errors_mm.append(error_mm)
    
    return np.array(errors_mm)


def create_boxplots():
    """Create boxplots for all measurements showing absolute errors in mm."""
    
    # Prepare data for plotting
    plot_data = []
    
    for measurement in ['BPD', 'OFD', 'TAD', 'APAD', 'FL']:
        models_to_use = MODELS_PER_MEASUREMENT[measurement]
        for model_name in models_to_use:
            errors = calculate_errors_mm(measurement, model_name)
            
            if errors is not None:
                for error in errors:
                    plot_data.append({
                        'Measurement': measurement,
                        'Anatomy': MEASUREMENTS[measurement]['anatomy'],
                        'Model': model_name,
                        'Error (mm)': error
                    })
    
    df_plot = pd.DataFrame(plot_data)
    
    # Calculate global y-axis limit
    max_error = df_plot['Error (mm)'].max()
    y_max = max(30, max_error * 1.1)  # At least 30mm or 110% of max error
    
    # Calculate subplot widths based on number of boxplots
    # Head: 8 boxplots (4 BPD + 4 OFD), Abdomen: 6 boxplots (3 TAD + 3 APAD), Femur: 3 boxplots
    # Including gaps: Head has 8 boxes + 0.8 gap = 8.8 units, Abdomen has 6 boxes + 0.8 gap = 6.8 units, Femur has 3 boxes = 3 units
    width_ratios = [8.8, 6.8, 3]  # Proportional to number of boxplots + gaps
    
    # Create figure with subplots of different widths
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(20, 6))
    gs = GridSpec(1, 3, figure=fig, width_ratios=width_ratios, wspace=0.15)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    
    # Color palette - different colors for each model (matching the paper figure)
    color_map = {
        'FP': '#1D4963',
        'HC18': '#5D94A6',
        'UCL': '#D3DEE0',
        'MULTICENTRE': '#E6E6E6'
    }
    
    # Plot for each anatomy - each will contain boxplots for all measurements and models
    anatomy_groups = [
        ('Head', ['BPD', 'OFD'], '(a)'),
        ('Abdomen', ['TAD', 'APAD'], '(b)'),
        ('Femur', ['FL'], '(c)')
    ]
    
    for idx, (anatomy_name, measurements, subplot_label) in enumerate(anatomy_groups):
        ax = axes[idx]
        
        # Filter data for these measurements
        data_anatomy = df_plot[df_plot['Measurement'].isin(measurements)]
        
        # Create boxplot data with positions that include gaps between measurements
        box_data = []
        box_labels = []
        box_colors = []
        positions = []
        
        current_pos = 1
        gap_between_measurements = 0.8  # Gap between measurement groups
        
        # Create boxplots grouped by measurement, then by model
        for meas_idx, measurement in enumerate(measurements):
            models_for_measurement = MODELS_PER_MEASUREMENT[measurement]
            for model_idx, model in enumerate(models_for_measurement):
                model_data = data_anatomy[(data_anatomy['Measurement'] == measurement) & 
                                         (data_anatomy['Model'] == model)]['Error (mm)'].values
                if len(model_data) > 0:
                    box_data.append(model_data)
                    # Label format: "Measurement-Model"
                    box_labels.append(f"{measurement}\n{MODEL_DISPLAY_NAMES[model]}")
                    box_colors.append(color_map[model])
                    positions.append(current_pos)
                    current_pos += 1
            
            # Add gap after each measurement group (except the last one)
            if meas_idx < len(measurements) - 1:
                current_pos += gap_between_measurements
        
        ax.set_title(f"{subplot_label} {anatomy_name}", fontweight='bold', fontsize=16)
        
        bp = ax.boxplot(
            box_data,
            positions=positions,
            labels=box_labels,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(marker='+', markerfacecolor='black', markersize=8, linestyle='none', markeredgecolor='black'),
            widths=0.7,
            showmeans=True,
            meanline=True,
            meanprops=dict(color='#941100', linewidth=2, linestyle='-')
        )
        
        # Color the boxes with black edges
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)
        
        # Style whiskers and caps with black
        for whisker in bp['whiskers']:
            whisker.set_color('black')
            whisker.set_linewidth(1.5)
        
        for cap in bp['caps']:
            cap.set_color('black')
            cap.set_linewidth(1.5)
        
        # Style the plot
        ax.set_ylabel('Error (mm)', fontweight='bold', fontsize=14)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Set consistent y-axis limits across all subplots
        ax.set_ylim(0, y_max)
        
        # Style the median lines (keep them black)
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(2)
        
        # Rotate x-axis labels if needed
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
    
    # Add legend (in the same order as displayed)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=color_map['FP'], edgecolor='black', label='FP'),
        Patch(facecolor=color_map['HC18'], edgecolor='black', label='HC18'),
        Patch(facecolor=color_map['UCL'], edgecolor='black', label='UCL'),
        Patch(facecolor=color_map['MULTICENTRE'], edgecolor='black', label='Ours')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=4, fontsize=12, frameon=True, fancybox=False, shadow=False)
    
    # Adjust subplot positioning to make room for legend
    fig.subplots_adjust(top=0.92)
    
    # Save figure
    output_file = base_dir / "ucl_error_boxplots.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Boxplot saved to: {output_file}")
    
    # Also save as PDF
    output_file_pdf = base_dir / "ucl_error_boxplots.pdf"
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"Boxplot saved to: {output_file_pdf}")

    # Also save as SVG
    output_file_svg = base_dir / "ucl_error_boxplots.svg"
    plt.savefig(output_file_svg, bbox_inches='tight')
    print(f"Boxplot saved to: {output_file_svg}")
    
    plt.show()
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    for measurement in ['BPD', 'OFD', 'TAD', 'APAD', 'FL']:
        print(f"\n{measurement}:")
        models_to_use = MODELS_PER_MEASUREMENT[measurement]
        for model in models_to_use:
            data = df_plot[(df_plot['Measurement'] == measurement) & 
                          (df_plot['Model'] == model)]['Error (mm)']
            if len(data) > 0:
                print(f"  {model}: Mean={data.mean():.2f} mm, Median={data.median():.2f} mm, "
                      f"Std={data.std():.2f} mm, Min={data.min():.2f} mm, Max={data.max():.2f} mm")


if __name__ == "__main__":
    create_boxplots()

