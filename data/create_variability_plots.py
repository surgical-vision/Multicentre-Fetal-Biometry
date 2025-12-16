"""
Author: Chiara Di Vece (chiara.divece.20@ucl.ac.uk)
Date: November 2025
Description: Create variability plots for the selected dataset.
Usage:
  python create_variability_plots.py --dataset HC18
  python create_variability_plots.py -d UCL
  VARIABILITY_DATASET=FP python create_variability_plots.py
"""

import os
import argparse
import cv2
import numpy as np
import pandas as pd
import seaborn as sns

# IMPORTANT: set backend BEFORE importing pyplot
try:
    import matplotlib
    matplotlib.use("Agg", force=True)
except Exception:
    pass

from matplotlib import pyplot as plt
from matplotlib import font_manager


# -----------------------------------------------------------------------------
# Resolve project directories robustly (works from any CWD)
# -----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Case A: script lives inside data/ (you ran it from data/ or referenced it there)
if os.path.isdir(os.path.join(SCRIPT_DIR, "annotations")) and os.path.isdir(os.path.join(SCRIPT_DIR, "images")):
    DATA_DIR = SCRIPT_DIR
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Case B: script lives in project root, and data/ is a subfolder
elif os.path.isdir(os.path.join(SCRIPT_DIR, "data", "annotations")) and os.path.isdir(os.path.join(SCRIPT_DIR, "data", "images")):
    PROJECT_ROOT = SCRIPT_DIR
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Case C: fallback to current working directory heuristics
else:
    CWD = os.getcwd()
    if os.path.isdir(os.path.join(CWD, "annotations")) and os.path.isdir(os.path.join(CWD, "images")):
        DATA_DIR = CWD
        PROJECT_ROOT = os.path.dirname(CWD)
    elif os.path.isdir(os.path.join(CWD, "data", "annotations")) and os.path.isdir(os.path.join(CWD, "data", "images")):
        PROJECT_ROOT = CWD
        DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    else:
        # last resort: assume data/ is next to script
        DATA_DIR = os.path.join(SCRIPT_DIR, "data")
        PROJECT_ROOT = SCRIPT_DIR

print(f"[Paths] SCRIPT_DIR   = {SCRIPT_DIR}")
print(f"[Paths] PROJECT_ROOT = {PROJECT_ROOT}")
print(f"[Paths] DATA_DIR     = {DATA_DIR}")


# -----------------------------------------------------------------------------
# Font setup
# -----------------------------------------------------------------------------
font_dirs = [os.path.join(PROJECT_ROOT, "fonts", "Lato"), os.path.join(DATA_DIR, "fonts", "Lato"), "fonts/Lato"]
font_files = []
for d in font_dirs:
    if os.path.isdir(d):
        font_files.extend(font_manager.findSystemFonts(fontpaths=[d]))

for font_file in font_files:
    try:
        font_manager.fontManager.addfont(font_file)
    except Exception:
        pass

plt.rcParams["font.family"] = "Lato"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 14
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["path.simplify"] = False
plt.rcParams["path.simplify_threshold"] = 0.0

LINE_WIDTH = 1.125

color_edge = "#404040"
color_red = "#941100"

metric_colors = {
    "bpd": "#28607A",
    "ofd": "#5D94A6",
    "tad": "#28607A",
    "apad": "#5D94A6",
    "fl":  "#28607A",
}


# -----------------------------------------------------------------------------
# CLI args (dataset as arg)
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Create variability plots for selected fetal biometry dataset.")
    parser.add_argument(
        "-d", "--dataset",
        choices=["FP", "HC18", "UCL", "MULTICENTRE"],
        help="Dataset to plot (overrides env VARIABILITY_DATASET)."
    )
    return parser.parse_args()


args = parse_args()
DATASET = args.dataset or os.environ.get("VARIABILITY_DATASET") or "HC18"
print(f"Creating variability plots for dataset: {DATASET}")


# -----------------------------------------------------------------------------
# Dataset configuration (absolute paths)
# -----------------------------------------------------------------------------
BASE_PATH = DATA_DIR  # <--- this fixes your 'cd data' problem

DATASET_CONFIGS = {
    "FP": {
        "head_csv": os.path.join(BASE_PATH, "annotations", "FP", "Head.csv"),
        "abdomen_csv": os.path.join(BASE_PATH, "annotations", "FP", "Abdomen.csv"),
        "femur_csv": os.path.join(BASE_PATH, "annotations", "FP", "Femur.csv"),
        "head_imgs": os.path.join(BASE_PATH, "images", "FP", "Head"),
        "abdomen_imgs": os.path.join(BASE_PATH, "images", "FP", "Abdomen"),
        "femur_imgs": os.path.join(BASE_PATH, "images", "FP", "Femur"),
        "output_dir": os.path.join(BASE_PATH, "graphs", "FP"),
    },
    "HC18": {
        "head_csv": os.path.join(BASE_PATH, "annotations", "HC18", "Head.csv"),
        "abdomen_csv": None,
        "femur_csv": None,
        "head_imgs": os.path.join(BASE_PATH, "images", "HC18", "Head"),
        "abdomen_imgs": None,
        "femur_imgs": None,
        "output_dir": os.path.join(BASE_PATH, "graphs", "HC18"),
    },
    "UCL": {
        "head_csv": os.path.join(BASE_PATH, "annotations", "UCL", "Head.csv"),
        "abdomen_csv": os.path.join(BASE_PATH, "annotations", "UCL", "Abdomen.csv"),
        "femur_csv": os.path.join(BASE_PATH, "annotations", "UCL", "Femur.csv"),
        "head_imgs": os.path.join(BASE_PATH, "images", "UCL", "Head"),
        "abdomen_imgs": os.path.join(BASE_PATH, "images", "UCL", "Abdomen"),
        "femur_imgs": os.path.join(BASE_PATH, "images", "UCL", "Femur"),
        "output_dir": os.path.join(BASE_PATH, "graphs", "UCL"),
    },
    "MULTICENTRE": {
        "head_csv": os.path.join(BASE_PATH, "annotations", "MULTICENTRE", "Head.csv"),
        "abdomen_csv": os.path.join(BASE_PATH, "annotations", "MULTICENTRE", "Abdomen.csv"),
        "femur_csv": os.path.join(BASE_PATH, "annotations", "MULTICENTRE", "Femur.csv"),
        "head_imgs": None,
        "abdomen_imgs": None,
        "femur_imgs": None,
        "output_dir": os.path.join(BASE_PATH, "graphs", "MULTICENTRE"),
    },
}

config = DATASET_CONFIGS[DATASET]

print("[Config] CSV/IMG paths (existence check):")
for k in ["head_csv", "abdomen_csv", "femur_csv", "head_imgs", "abdomen_imgs", "femur_imgs"]:
    v = config.get(k)
    if v is None:
        print(f"  - {k:12s}: None")
    else:
        print(f"  - {k:12s}: {v}  | exists={os.path.exists(v)}")


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
def maybe_read_csv(path, label):
    if not path:
        return None
    if not os.path.exists(path):
        print(f"[WARN] Missing {label} CSV: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Failed to read {label} CSV: {path} ({type(e).__name__}: {e})")
        return None

def save_png_via_agg_buffer(fig, out_path):
    """
    Save figure as PNG without Pillow by grabbing the Agg RGBA buffer and writing with OpenCV.
    """
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())  # H x W x 4

    if rgba.size == 0 or rgba.shape[0] == 0 or rgba.shape[1] == 0:
        raise RuntimeError(f"Rendered image is empty: shape={rgba.shape}")

    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    ok = cv2.imwrite(out_path, bgra)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for {out_path}")


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
df_head = maybe_read_csv(config["head_csv"], "HEAD")
df_abdomen = maybe_read_csv(config["abdomen_csv"], "ABDOMEN")
df_femur = maybe_read_csv(config["femur_csv"], "FEMUR")

DATASETS = {}
if df_head is not None:
    DATASETS["head"] = ["ofd", "bpd"] if "ofd_1_x" in df_head.columns else ["bpd"]
if df_abdomen is not None:
    DATASETS["abdomen"] = ["tad", "apad"]
if df_femur is not None:
    DATASETS["femur"] = ["fl"]

df_by_type = {"head": df_head, "abdomen": df_abdomen, "femur": df_femur}

if len(DATASETS) == 0:
    raise RuntimeError(
        "No datasets loaded. CSVs were not found/readable. "
        "Check the printed path existence checks above."
    )


# -----------------------------------------------------------------------------
# PLOTTING
# -----------------------------------------------------------------------------
N = 80
bottom = 8
theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
width = (2 * np.pi) / N

n_anatomies = len(DATASETS)
# Match proportions: 3 anatomies uses (12, 10), so 1 anatomy should be ~(12, 3.5) to maintain same subplot aspect ratio
if n_anatomies == 1:
    fig = plt.figure(figsize=(12, 10/3))  # Same width, proportional height for 1 row
elif n_anatomies == 3:
    fig = plt.figure(figsize=(12, 10))
else:
    fig = plt.figure(figsize=(12, 4 * n_anatomies))
plt.subplots_adjust(hspace=0.4, wspace=0.4)

for i, (db_type, meas_types) in enumerate(DATASETS.items()):
    print(f"\nProcessing {db_type.upper()} with metrics: {meas_types}")
    orig_db = df_by_type[db_type].copy()

    # 1) ORIENTATION (polar)
    cur_ax = plt.subplot(n_anatomies, 3, 3 * i + 1, polar=True)
    cur_ax.spines["polar"].set_linewidth(LINE_WIDTH)

    for meas_type in meas_types:
        angles_arr = np.arctan2(
            orig_db[f"{meas_type}_1_y"] - orig_db[f"{meas_type}_2_y"],
            orig_db[f"{meas_type}_1_x"] - orig_db[f"{meas_type}_2_x"],
        )
        angles_normalized = (angles_arr + (2 * np.pi)) % (2 * np.pi)

        print(f"  Angles ({meas_type.upper()}): {len(angles_normalized)} measurements")

        color = metric_colors.get(meas_type, "#FF6B6B")
        cur_ax.hist(
            angles_normalized,
            bins=theta,
            width=width,
            bottom=bottom,
            color=color,
            edgecolor=color_edge,
            alpha=1.0,
            label=meas_type.upper(),
        )

    cur_ax.set_rlim((1.0, 1000.0))
    cur_ax.set_rscale("symlog")
    cur_ax.set_title(f"{db_type.capitalize()} - Orientation")
    cur_ax.tick_params(width=LINE_WIDTH)

    # 2) POSITION (kde)
    cur_ax = plt.subplot(n_anatomies, 3, 3 * i + 2)

    pos_x_all, pos_y_all = [], []
    for meas_type in meas_types:
        pos_x = (orig_db[f"{meas_type}_1_x"] + orig_db[f"{meas_type}_2_x"]) / (4.0 * orig_db["center_w"])
        pos_y = (orig_db[f"{meas_type}_1_y"] + orig_db[f"{meas_type}_2_y"]) / (4.0 * orig_db["center_h"])
        pos_x_all.extend(pos_x.tolist())
        pos_y_all.extend(pos_y.tolist())

    print(f"  Position: Combined {len(pos_x_all)} measurements from {meas_types}")

    pos_df = pd.DataFrame({"pos_x": pos_x_all, "pos_y": pos_y_all})
    color = metric_colors.get(meas_types[0], "#5D94A6")
    sns.kdeplot(data=pos_df, x="pos_x", y="pos_y", fill=True, ax=cur_ax, alpha=1, color=color)

    cur_ax.set_ylim(0, 1)
    cur_ax.set_xlim(0, 1)
    cur_ax.set_title(f"{db_type.capitalize()} - Position")
    for spine in cur_ax.spines.values():
        spine.set_linewidth(LINE_WIDTH)
    cur_ax.tick_params(width=LINE_WIDTH)

    # 3) SIZE (kde)
    cur_ax = plt.subplot(n_anatomies, 3, 3 * i + 3)

    size_x_list, size_y_list = [], []
    for meas_type in meas_types:
        size_x = (orig_db[f"{meas_type}_1_x"] - orig_db[f"{meas_type}_2_x"]).abs()
        size_y = (orig_db[f"{meas_type}_1_y"] - orig_db[f"{meas_type}_2_y"]).abs()
        size_x_list.append(size_x.reset_index(drop=True))
        size_y_list.append(size_y.reset_index(drop=True))

    size_x_combined = pd.concat(size_x_list, axis=1).max(axis=1)
    size_y_combined = pd.concat(size_y_list, axis=1).max(axis=1)

    totalsize = size_x_combined * size_y_combined / (4.0 * orig_db["center_w"] * orig_db["center_h"])

    print(f"  Size: Combined max size across {meas_types}")

    size_df = pd.DataFrame({"totalsize": totalsize})
    sns.kdeplot(
        data=size_df,
        x="totalsize",
        fill=True,
        color=color_red,
        ax=cur_ax,
        alpha=0.25,
        bw_method=0.1,
        linewidth=LINE_WIDTH,
    )

    cur_ax.set_title(f"{db_type.capitalize()} - Size")
    for spine in cur_ax.spines.values():
        spine.set_linewidth(LINE_WIDTH)
    cur_ax.tick_params(width=LINE_WIDTH)


# -----------------------------------------------------------------------------
# SAVE FIGURE
# -----------------------------------------------------------------------------
os.makedirs(config["output_dir"], exist_ok=True)

output_png = os.path.join(config["output_dir"], f"{DATASET}_variability.png")
output_pdf = os.path.join(config["output_dir"], f"{DATASET}_variability.pdf")
output_svg = os.path.join(config["output_dir"], f"{DATASET}_variability.svg")

saved_formats = []

# PNG (try matplotlib first, then fallback that avoids Pillow)
try:
    fig.savefig(output_png, dpi=150, format="png")
    saved_formats.append("PNG")
    print(f"✓ Saved: {output_png}")
except Exception as e:
    print(f"✗ PNG failed via savefig: {type(e).__name__}: {e}")
    try:
        save_png_via_agg_buffer(fig, output_png)
        saved_formats.append("PNG")
        print(f"✓ Saved via Agg buffer + OpenCV: {output_png}")
    except Exception as e2:
        print(f"✗ PNG also failed via OpenCV fallback: {type(e2).__name__}: {e2}")
        try:
            fig.savefig(output_pdf, format="pdf")
            saved_formats.append("PDF")
            print(f"✓ Saved: {output_pdf} (PNG unavailable, using PDF instead)")
        except Exception as e3:
            print(f"✗ PDF also failed: {type(e3).__name__}: {e3}")

# SVG
try:
    fig.savefig(output_svg, format="svg")
    saved_formats.append("SVG")
    print(f"✓ Saved: {output_svg}")
except Exception as e:
    print(f"✗ SVG failed: {type(e).__name__}: {e}")

print(f"\n{'=' * 70}")
print(f"Saved formats: {', '.join(saved_formats) if saved_formats else '(none)'}")
print(f"{'=' * 70}")

plt.close(fig)