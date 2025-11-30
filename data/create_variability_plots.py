"""
Author: chiaradivece
Date: November 2025
Description: Create variability plots for the selected dataset.
Usage: python create_variability_plots.py
"""

import pandas as pd
import numpy as np
import os
import cv2
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import font_manager

# Register the Lato font
font_dirs = ['fonts/Lato']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

# Set the default font to Lato
plt.rcParams['font.family'] = 'Lato'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'

# Make SVG fully editable (no rasterization, text as text not paths)
plt.rcParams['svg.fonttype'] = 'none'  # Keep text as editable text
plt.rcParams['path.simplify'] = False  # Don't simplify paths
plt.rcParams['path.simplify_threshold'] = 0.0  # No path simplification

# Line thickness for all elements (in points)
LINE_WIDTH = 1.125

# Colors
color_edge = '#404040'
color_red = '#941100'

# Define colors for individual metrics
metric_colors = {
    'bpd': '#28607A',  # dark teal
    'ofd': '#5D94A6',  # light teal
    'tad': '#28607A',  # dark teal
    'apad': '#5D94A6', # light teal
    'fl': '#28607A'    # dark teal
}

# ============================================================================
# CONFIGURATION: Set dataset here or via environment variable
# ============================================================================
# DATASET = 'FP'  # Options: 'FP', 'HC18', 'UCL', 'MULTICENTRE'
# DATASET = 'UCL'
DATASET = 'HC18'

# Get dataset from environment if set (for shell script use)
if 'VARIABILITY_DATASET' in os.environ:
    DATASET = os.environ['VARIABILITY_DATASET']

# ============================================================================
# DATASET-SPECIFIC PATHS
# ============================================================================
BASE_PATH = 'data'

DATASET_CONFIGS = {
    'FP': {
        'head_csv': f'{BASE_PATH}/FP/Head.csv',
        'abdomen_csv': f'{BASE_PATH}/FP/Abdomen.csv',
        'femur_csv': f'{BASE_PATH}/FP/Femur.csv',
        'head_imgs': f'{BASE_PATH}/images/FP/Head',
        'abdomen_imgs': f'{BASE_PATH}/images/FP/Abdomen',
        'femur_imgs': f'{BASE_PATH}/images/FP/FL',
        'output_dir': f'{BASE_PATH}/graphs/FP'
    },
    'HC18': {
        'head_csv': f'{BASE_PATH}/HC18/Head.csv',
        'abdomen_csv': None,
        'femur_csv': None,
        'head_imgs': f'{BASE_PATH}/images/HC18/Head',
        'abdomen_imgs': None,
        'femur_imgs': None,
        'output_dir': f'{BASE_PATH}/graphs/HC18'
    },
    'UCL': {
        'head_csv': f'{BASE_PATH}/UCL/Head.csv',
        'abdomen_csv': f'{BASE_PATH}/UCL/Abdomen.csv',
        'femur_csv': f'{BASE_PATH}/UCL/Femur.csv',
        'head_imgs': f'{BASE_PATH}/images/UCL/Head',
        'abdomen_imgs': f'{BASE_PATH}/images/UCL/Abdomen',
        'femur_imgs': f'{BASE_PATH}/images/UCL/Femur',
        'output_dir': f'{BASE_PATH}/graphs/UCL'
    },
    'MULTICENTRE': {
        'head_csv': f'{BASE_PATH}/MULTICENTRE/Head.csv',
        'abdomen_csv': f'{BASE_PATH}/MULTICENTRE/Abdomen.csv',
        'femur_csv': f'{BASE_PATH}/MULTICENTRE/Femur.csv',
        'head_imgs': None,  # No images needed for plotting
        'abdomen_imgs': None,
        'femur_imgs': None,
        'output_dir': f'{BASE_PATH}/graphs/MULTICENTRE'
    }
}

# Get config for selected dataset
config = DATASET_CONFIGS[DATASET]

# Load data
df_head = pd.read_csv(config['head_csv']) if config['head_csv'] and os.path.exists(config['head_csv']) else None
df_abdomen = pd.read_csv(config['abdomen_csv']) if config['abdomen_csv'] and os.path.exists(config['abdomen_csv']) else None
df_femur = pd.read_csv(config['femur_csv']) if config['femur_csv'] and os.path.exists(config['femur_csv']) else None

# ============================================================================
# RECALCULATE CENTER COORDINATES FROM ACTUAL IMAGES
# This ensures position measurements are relative to true image centers
# ============================================================================
if df_head is not None and config['head_imgs'] is not None:
    print(f"Recalculating center coordinates for HEAD from images...")
    for i, row in df_head.iterrows():
        img_path = os.path.join(config['head_imgs'], row['image_name'])
        if os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                h, w = img.shape
                df_head.loc[i, 'center_w'] = w / 2.0
                df_head.loc[i, 'center_h'] = h / 2.0

if df_abdomen is not None and config['abdomen_imgs'] is not None:
    print(f"Recalculating center coordinates for ABDOMEN from images...")
    for i, row in df_abdomen.iterrows():
        img_path = os.path.join(config['abdomen_imgs'], row['image_name'])
        if os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                h, w = img.shape
                df_abdomen.loc[i, 'center_w'] = w / 2.0
                df_abdomen.loc[i, 'center_h'] = h / 2.0

if df_femur is not None and config['femur_imgs'] is not None:
    print(f"Recalculating center coordinates for FEMUR from images...")
    for i, row in df_femur.iterrows():
        img_path = os.path.join(config['femur_imgs'], row['image_name'])
        if os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                h, w = img.shape
                df_femur.loc[i, 'center_w'] = w / 2.0
                df_femur.loc[i, 'center_h'] = h / 2.0

# Define which measurements to use for each anatomy
DATASETS = {}
if df_head is not None:
    DATASETS['head'] = ['ofd', 'bpd'] if 'ofd_1_x' in df_head.columns else ['bpd']
if df_abdomen is not None:
    DATASETS['abdomen'] = ['tad', 'apad']
if df_femur is not None:
    DATASETS['femur'] = ['fl']

# ============================================================================
# PLOTTING
# ============================================================================

N = 80
bottom = 8
theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
width = (2 * np.pi) / N

# Determine number of rows needed
n_anatomies = len(DATASETS)
fig = plt.figure(figsize=(12, 10) if n_anatomies == 3 else (12, 4 * n_anatomies))
plt.subplots_adjust(hspace=0.4, wspace=0.4)

for i, (db_type, meas_types) in enumerate(DATASETS.items()):
    print(f"\nProcessing {db_type.upper()} with metrics: {meas_types}")
    
    orig_db = eval(f"df_{db_type}").copy()
    
    # ========================================================================
    # 1. ANGLES (Orientation) - POLAR PLOT
    # ========================================================================
    cur_ax = plt.subplot(n_anatomies, 3, 3*i + 1, polar=True)
    
    # Set line width for polar plot spines (outermost circles)
    cur_ax.spines['polar'].set_linewidth(LINE_WIDTH)
    
    # Plot each metric's angles separately with different colors
    for idx, meas_type in enumerate(meas_types):
        angles_arr = np.arctan2(
            orig_db[f'{meas_type}_1_y'] - orig_db[f'{meas_type}_2_y'],
            orig_db[f'{meas_type}_1_x'] - orig_db[f'{meas_type}_2_x']
        )
        angles_normalized = (angles_arr + (2*np.pi)) % (2*np.pi)
        
        print(f"  Angles ({meas_type.upper()}): {len(angles_normalized)} measurements")
        
        color = metric_colors.get(meas_type, '#FF6B6B')  # Default to red if not found
        
        # Plot each metric's angles separately on the same polar subplot
        cur_ax.hist(angles_normalized, bins=theta, width=width, bottom=bottom, 
                   color=color, edgecolor=color_edge, alpha=1.0, label=meas_type.upper())
    
    cur_ax.set_rlim((1.0, 1000.0))
    cur_ax.set_rscale('symlog')
    cur_ax.set_title(f'{db_type.capitalize()} - Orientation')
    
    # Set tick parameters for polar plot
    cur_ax.tick_params(width=LINE_WIDTH)
    
    # ========================================================================
    # 2. POSITION - KDE PLOT
    # ========================================================================
    cur_ax = plt.subplot(n_anatomies, 3, 3*i + 2)
    
    # Combine position data from all metrics
    pos_x_all = []
    pos_y_all = []
    
    for meas_type in meas_types:
        pos_x = (orig_db[f'{meas_type}_1_x'] + orig_db[f'{meas_type}_2_x']) / (4.0 * orig_db['center_w'])
        pos_y = (orig_db[f'{meas_type}_1_y'] + orig_db[f'{meas_type}_2_y']) / (4.0 * orig_db['center_h'])
        pos_x_all.extend(pos_x.tolist())
        pos_y_all.extend(pos_y.tolist())
    
    print(f"  Position: Combined {len(pos_x_all)} measurements from {meas_types}")
    
    # Create temporary dataframe for combined position data
    pos_df = pd.DataFrame({'pos_x': pos_x_all, 'pos_y': pos_y_all})
    
    # Use the first metric's color for combined position plot
    color = metric_colors.get(meas_types[0], '#5D94A6')
    sns.kdeplot(data=pos_df, x="pos_x", y="pos_y", fill=True, ax=cur_ax, 
               alpha=1, color=color)
    
    cur_ax.set_ylim(0, 1)
    cur_ax.set_xlim(0, 1)
    cur_ax.set_title(f'{db_type.capitalize()} - Position')
    
    # Set box and tick line width
    for spine in cur_ax.spines.values():
        spine.set_linewidth(LINE_WIDTH)
    cur_ax.tick_params(width=LINE_WIDTH)
    
    # ========================================================================
    # 3. SIZE - KDE PLOT
    # ========================================================================
    cur_ax = plt.subplot(n_anatomies, 3, 3*i + 3)
    
    # Calculate size for each metric and take the max
    size_x_list = []
    size_y_list = []
    
    for meas_type in meas_types:
        size_x = (orig_db[f'{meas_type}_1_x'] - orig_db[f'{meas_type}_2_x']).abs()
        size_y = (orig_db[f'{meas_type}_1_y'] - orig_db[f'{meas_type}_2_y']).abs()
        size_x_list.append(size_x.reset_index(drop=True))
        size_y_list.append(size_y.reset_index(drop=True))
    
    # Combine and take max across all metrics
    size_x_combined = pd.concat(size_x_list, axis=1).max(axis=1)
    size_y_combined = pd.concat(size_y_list, axis=1).max(axis=1)
    
    totalsize = size_x_combined * size_y_combined / (4.0 * orig_db['center_w'] * orig_db['center_h'])
    
    print(f"  Size: Combined max size across {meas_types}")
    
    size_df = pd.DataFrame({'totalsize': totalsize})
    sns.kdeplot(data=size_df, x="totalsize", fill=True, color=color_red, 
               ax=cur_ax, alpha=0.25, bw_method=0.1, linewidth=LINE_WIDTH)
    
    cur_ax.set_title(f'{db_type.capitalize()} - Size')
    
    # Set box and tick line width
    for spine in cur_ax.spines.values():
        spine.set_linewidth(LINE_WIDTH)
    cur_ax.tick_params(width=LINE_WIDTH)

# ============================================================================
# SAVE FIGURE
# ============================================================================
os.makedirs(config['output_dir'], exist_ok=True)
output_png = os.path.join(config['output_dir'], f'{DATASET}_variability.png')
output_svg = os.path.join(config['output_dir'], f'{DATASET}_variability.svg')

plt.savefig(output_png, dpi=600)
plt.savefig(output_svg, dpi=600)

print(f"\n✓ Saved: {output_png}")
print(f"✓ Saved: {output_svg}")

plt.close()

