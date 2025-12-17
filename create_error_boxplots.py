#!/usr/bin/env python3
"""
Author: Chiara Di Vece
Date: 2025-12-17

Create boxplots showing absolute biometry error (mm) on an evaluation dataset
(default: MULTICENTRE test set).

Key design:
- We run inference directly (no cached predictions) to guarantee prediction order
  matches the dataset order.
- We FILTER the CSV first (px_to_mm present + valid landmarks), write a filtered
  CSV, and point cfg.DATASET.TESTSET to that filtered CSV so the dataset and
  your GT DataFrame are aligned by construction.

Usage:
  python create_error_boxplots.py

Run from the Multicentre-Fetal-Biometry repository root.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib import font_manager


# Add repo root to path to import lib modules
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

import lib.models as models
from lib.config import config as BASE_CONFIG
from lib.datasets import get_dataset
from lib.core import function


# -------------------- Paths --------------------

base_dir = SCRIPT_DIR
data_dir = base_dir / "data" / "annotations"
output_dir = base_dir / "output" / "FETAL"
tmp_dir = output_dir / "_tmp_filtered_csv"
tmp_dir.mkdir(parents=True, exist_ok=True)

_FILTER_CACHE: Dict[Tuple[str, str], Tuple[pd.DataFrame, Path]] = {}

# -------------------- Fonts / Style --------------------

try:
    font_dir = base_dir / "fonts" / "Lato"
    if font_dir.exists():
        font_files = font_manager.findSystemFonts(fontpaths=str(font_dir))
        for f in font_files:
            if "Lato" in f:
                font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = "Lato"
except Exception:
    print("Lato font not found, using default font")

plt.rcParams["font.size"] = 14
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"


# -------------------- Definitions --------------------

EVAL_DATASET = "MULTICENTRE"  # <---- you wanted this

MEASUREMENTS: Dict[str, Dict[str, str]] = {
    "BPD": {"anatomy": "brain",   "csv_prefix": "Head",    "landmark_prefix": "bpd",  "display_name": "BPD"},
    "OFD": {"anatomy": "brain",   "csv_prefix": "Head",    "landmark_prefix": "ofd",  "display_name": "OFD"},
    "TAD": {"anatomy": "abdomen", "csv_prefix": "Abdomen", "landmark_prefix": "tad",  "display_name": "TAD"},
    "APAD": {"anatomy": "abdomen", "csv_prefix": "Abdomen", "landmark_prefix": "apad", "display_name": "APAD"},
    "FL":  {"anatomy": "femur",   "csv_prefix": "Femur",   "landmark_prefix": "fl",   "display_name": "FL"},
}

MEASUREMENTS_BY_ANATOMY = {
    "Head": ["BPD", "OFD"],
    "Abdomen": ["TAD", "APAD"],
    "Femur": ["FL"],
}

# Checkpoint directories per model per measurement
MODELS = {
    "UCL": {
        "BPD": "fetal_landmark_hrnet_w18_UCL_brain_BPD",
        "OFD": "fetal_landmark_hrnet_w18_UCL_brain_OFD",
        "TAD": "fetal_landmark_hrnet_w18_UCL_abdomen_TAD",
        "APAD": "fetal_landmark_hrnet_w18_UCL_abdomen_APAD",
        "FL":  "fetal_landmark_hrnet_w18_UCL_femur_FL",
    },
    "MULTICENTRE": {
        "BPD": "fetal_landmark_hrnet_w18_MULTICENTRE_brain_BPD",
        "OFD": "fetal_landmark_hrnet_w18_MULTICENTRE_brain_OFD",
        "TAD": "fetal_landmark_hrnet_w18_MULTICENTRE_abdomen_TAD",
        "APAD": "fetal_landmark_hrnet_w18_MULTICENTRE_abdomen_APAD",
        "FL":  "fetal_landmark_hrnet_w18_MULTICENTRE_femur_FL",
    },
    "FP": {
        "BPD": "fetal_landmark_hrnet_w18_FP_brain_BPD",
        "OFD": "fetal_landmark_hrnet_w18_FP_brain_OFD",
        "TAD": "fetal_landmark_hrnet_w18_FP_abdomen_TAD",
        "APAD": "fetal_landmark_hrnet_w18_FP_abdomen_APAD",
        "FL":  "fetal_landmark_hrnet_w18_FP_femur_FL",
    },
    "HC18": {
        "BPD": "fetal_landmark_hrnet_w18_HC18_brain_BPD",
        "OFD": "fetal_landmark_hrnet_w18_HC18_brain_OFD",
        "TAD": None,
        "APAD": None,
        "FL":  None,
    },
}

MODELS_PER_MEASUREMENT = {
    "BPD":  ["FP", "HC18", "UCL", "MULTICENTRE"],
    "OFD":  ["FP", "HC18", "UCL", "MULTICENTRE"],
    "TAD":  ["FP", "UCL", "MULTICENTRE"],
    "APAD": ["FP", "UCL", "MULTICENTRE"],
    "FL":   ["FP", "UCL", "MULTICENTRE"],
}

MODEL_DISPLAY_NAMES = {"FP": "FP", "HC18": "HC18", "UCL": "UCL", "MULTICENTRE": "M-C"}


# -------------------- Helpers --------------------

def calculate_distance_pixels(x1, y1, x2, y2) -> float:
    return float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))


def find_eval_csv(eval_dataset: str, csv_prefix: str) -> Path:
    """
    Tries a couple of plausible locations.
    If your repo uses only one, this still works. If it uses neither, it fails loudly.
    """
    candidates = [
        data_dir / eval_dataset / "px_to_mm" / f"{csv_prefix}_Test.csv",
        data_dir / eval_dataset / f"{csv_prefix}_Test.csv",
    ]
    for p in candidates:
        if p.exists():
            return p

    tried = "\n  - " + "\n  - ".join(str(x) for x in candidates)
    raise FileNotFoundError(f"Could not find eval CSV for {eval_dataset}/{csv_prefix}. Tried:{tried}")


def filtered_csv_path(eval_dataset: str, csv_prefix: str, measurement: str) -> Path:
    return tmp_dir / f"{eval_dataset}_{csv_prefix}_Test_{measurement}_filtered.csv"


def filter_and_write_eval_csv(eval_dataset: str, measurement: str) -> Tuple[pd.DataFrame, Path]:
    """
    Filter out rows where px_to_mm_rate is missing/invalid AND where landmarks are invalid.
    Write filtered CSV so the dataset uses identical filtering as our GT DataFrame.
    """
    key_cache = (eval_dataset, measurement)
    if key_cache in _FILTER_CACHE:
        return _FILTER_CACHE[key_cache]

    csv_prefix = MEASUREMENTS[measurement]["csv_prefix"]
    landmark_prefix = MEASUREMENTS[measurement]["landmark_prefix"]

    in_csv = find_eval_csv(eval_dataset, csv_prefix)
    df = pd.read_csv(in_csv)

    junk = [c for c in df.columns if str(c).lower() == "index" or str(c).startswith("Unnamed")]
    if junk:
        df = df.drop(columns=junk)

    # px column name: accept either
    if "px_to_mm_rate" in df.columns:
        px_col = "px_to_mm_rate"
    elif "px_to_mm" in df.columns:
        px_col = "px_to_mm"
    else:
        raise KeyError(f"{in_csv} missing px_to_mm column (expected 'px_to_mm_rate' or 'px_to_mm').")

    img_col = df.columns[0]
    coords = [
        f"{landmark_prefix}_1_x", f"{landmark_prefix}_1_y",
        f"{landmark_prefix}_2_x", f"{landmark_prefix}_2_y",
    ]

    missing = [c for c in ([img_col] + coords + [px_col]) if c not in df.columns]
    if missing:
        raise KeyError(f"{in_csv} is missing required columns for {measurement}: {missing}")

    mask_img = df[img_col].isna()

    gt_xy = df[coords].to_numpy(dtype=float)
    mask_lm = ((gt_xy < 0) | np.isnan(gt_xy) | (~np.isfinite(gt_xy))).any(axis=1)

    px = pd.to_numeric(df[px_col], errors="coerce").to_numpy(dtype=float)
    mask_px = (~np.isfinite(px)) | np.isnan(px) | (px <= 0)

    keep = ~(mask_img | mask_lm | mask_px)
    df_f = df.loc[keep].reset_index(drop=True)

    out_csv = filtered_csv_path(eval_dataset, csv_prefix, measurement)
    df_f.to_csv(out_csv, index=False)

    dropped = int((~keep).sum())
    total = int(len(df))
    print(f"[{eval_dataset}-{measurement}] Filtered CSV: kept {len(df_f)}/{total} (dropped {dropped}) -> {out_csv}")

    _FILTER_CACHE[key_cache] = (df_f, out_csv)
    return df_f, out_csv


def load_checkpoint(model_folder: Path) -> Tuple[dict, Optional[np.ndarray]]:
    """
    Loads model weights (state_dict) and optional d_vect from final_state/latest.
    """
    model_file = model_folder / "final_state.pth"
    if not model_file.exists():
        model_file = model_folder / "latest.pth"
    if not model_file.exists():
        raise FileNotFoundError(f"No checkpoint found in {model_folder} (expected final_state.pth or latest.pth)")

    payload = torch.load(model_file, map_location="cpu")

    learned_d_vect = None
    if isinstance(payload, dict) and "state_dict" in payload:
        weights = payload["state_dict"]
        learned_d_vect = payload.get("d_vect", None)
    else:
        weights = payload

    if isinstance(learned_d_vect, torch.Tensor):
        learned_d_vect = learned_d_vect.detach().cpu().numpy()

    return weights, learned_d_vect


def compute_d_vect_from_training_cfg(model_name: str, anatomy: str, measurement: str) -> Optional[np.ndarray]:
    """
    Fallback if d_vect wasn't stored in checkpoint: compute from the model's training config.
    """
    cfg_file = base_dir / "experiments" / "fetal" / f"fetal_landmark_hrnet_w18_{model_name}_{anatomy}_{measurement}.yaml"
    if not cfg_file.exists():
        print(f"Warning: training cfg not found for d_vect: {cfg_file}")
        return None

    cfg = BASE_CONFIG.clone()
    cfg.defrost()
    cfg.merge_from_file(str(cfg_file))
    cfg.freeze()

    dataset_type = get_dataset(cfg)
    dummy_train = dataset_type(cfg, is_train=True)
    return getattr(dummy_train, "d_vect", None)


def load_model_and_run_inference(
    model_name: str,
    measurement: str,
    eval_dataset: str,
    filtered_csv: Path,
    expected_len: int,
) -> np.ndarray:
    """
    Load model checkpoint (from model_name) but run inference on eval_dataset test CSV (filtered).
    Returns predictions as (N, 2, 2) numpy.
    """
    anatomy = MEASUREMENTS[measurement]["anatomy"]
    model_dir_name = MODELS[model_name][measurement]
    if model_dir_name is None:
        return None

    model_folder = output_dir / model_dir_name
    weights, learned_d_vect = load_checkpoint(model_folder)

    if learned_d_vect is None:
        learned_d_vect = compute_d_vect_from_training_cfg(model_name, anatomy, measurement)
        if learned_d_vect is not None:
            print(f"[{model_name}-{measurement}] Computed d_vect from training cfg.")
        else:
            print(f"[{model_name}-{measurement}] Warning: d_vect unavailable (may affect endpoint ordering).")

    # Build eval cfg (use eval_dataset yaml as baseline)
    eval_cfg_file = base_dir / "experiments" / "fetal" / f"fetal_landmark_hrnet_w18_{eval_dataset}_{anatomy}_{measurement}.yaml"
    if not eval_cfg_file.exists():
        raise FileNotFoundError(f"Eval cfg not found: {eval_cfg_file}")

    cfg = BASE_CONFIG.clone()
    cfg.defrost()
    cfg.merge_from_file(str(eval_cfg_file))
    cfg.DATASET.TESTSET = str(filtered_csv)
    cfg.MODEL.INIT_WEIGHTS = False
    cfg.freeze()

    # CuDNN
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLED

    # Model
    model = models.get_face_alignment_net(cfg)
    gpus = list(cfg.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    has_module_prefix = any(k.startswith("module.") for k in weights.keys())
    if has_module_prefix:
        model.load_state_dict(weights, strict=True)
    else:
        model.module.load_state_dict(weights, strict=True)

    dataset_type = get_dataset(cfg)
    test_dataset = dataset_type(cfg, is_train=False, d_vect=learned_d_vect)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * max(1, len(gpus)),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    model.eval()
    _, _, _, predictions = function.inference(cfg, test_loader, model)  # predictions tensor

    preds = predictions.detach().cpu().numpy()
    if preds.shape[0] != expected_len:
        raise RuntimeError(
            f"[{model_name}-{measurement}] Prediction/GT mismatch after filtered CSV: "
            f"preds={preds.shape[0]} vs expected={expected_len}"
        )
    return preds


def calculate_errors_mm(measurement: str, model_name: str, eval_dataset: str) -> Optional[np.ndarray]:
    """
    Absolute error in mm between clinically measured and predicted distances.
    """
    df_f, filtered_csv = filter_and_write_eval_csv(eval_dataset, measurement)
    predictions = load_model_and_run_inference(
        model_name=model_name,
        measurement=measurement,
        eval_dataset=eval_dataset,
        filtered_csv=filtered_csv,
        expected_len=len(df_f),
    )
    if predictions is None:
        return None

    # choose px column name consistently with filtering
    if "px_to_mm_rate" in df_f.columns:
        px_col = "px_to_mm_rate"
    elif "px_to_mm" in df_f.columns:
        px_col = "px_to_mm"
    else:
        raise KeyError("Filtered df is missing px_to_mm column (should not happen).")

    prefix = MEASUREMENTS[measurement]["landmark_prefix"]
    errors_mm = []

    for idx, row in df_f.iterrows():
        gt_dist_px = calculate_distance_pixels(
            row[f"{prefix}_1_x"], row[f"{prefix}_1_y"],
            row[f"{prefix}_2_x"], row[f"{prefix}_2_y"],
        )

        pred_pt1 = predictions[idx, 0, :]
        pred_pt2 = predictions[idx, 1, :]
        pred_dist_px = calculate_distance_pixels(pred_pt1[0], pred_pt1[1], pred_pt2[0], pred_pt2[1])

        px_to_mm = float(row[px_col])
        gt_mm = gt_dist_px * px_to_mm
        pr_mm = pred_dist_px * px_to_mm

        errors_mm.append(abs(pr_mm - gt_mm))

    return np.asarray(errors_mm, dtype=float)


# -------------------- Plotting --------------------

def create_boxplots(eval_dataset: str = EVAL_DATASET):
    plot_data = []

    for measurement in ["BPD", "OFD", "TAD", "APAD", "FL"]:
        for model_name in MODELS_PER_MEASUREMENT[measurement]:
            errors = calculate_errors_mm(measurement, model_name, eval_dataset)
            if errors is None:
                continue
            for e in errors:
                plot_data.append({
                    "Measurement": measurement,
                    "Anatomy": MEASUREMENTS[measurement]["anatomy"],
                    "Model": model_name,
                    "Error (mm)": float(e),
                })

    df_plot = pd.DataFrame(plot_data)
    if df_plot.empty:
        raise RuntimeError("No plot data collected. Check CSV paths, px_to_mm_rate column, and checkpoints.")

    max_error = float(df_plot["Error (mm)"].max())
    y_max = max(30.0, max_error * 1.1)

    # widths roughly proportional to number of boxes per panel
    width_ratios = [8.8, 6.8, 3.0]
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(20, 6))
    gs = GridSpec(1, 3, figure=fig, width_ratios=width_ratios, wspace=0.15)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    color_map = {
        "FP": "#1D4963",
        "HC18": "#5D94A6",
        "UCL": "#D3DEE0",
        "MULTICENTRE": "#E6E6E6",
    }

    anatomy_groups = [
        ("Head", ["BPD", "OFD"], "(a)"),
        ("Abdomen", ["TAD", "APAD"], "(b)"),
        ("Femur", ["FL"], "(c)"),
    ]

    for ax_idx, (anatomy_name, measurements, subplot_label) in enumerate(anatomy_groups):
        ax = axes[ax_idx]
        data_anatomy = df_plot[df_plot["Measurement"].isin(measurements)]

        box_data = []
        box_labels = []
        box_colors = []
        positions = []

        current_pos = 1.0
        gap_between_measurements = 0.8

        for meas_idx, measurement in enumerate(measurements):
            models_for_measurement = MODELS_PER_MEASUREMENT[measurement]
            for model in models_for_measurement:
                model_data = data_anatomy[
                    (data_anatomy["Measurement"] == measurement) &
                    (data_anatomy["Model"] == model)
                ]["Error (mm)"].values

                if model_data.size > 0:
                    box_data.append(model_data)
                    box_labels.append(f"{measurement}\n{MODEL_DISPLAY_NAMES[model]}")
                    box_colors.append(color_map[model])
                    positions.append(current_pos)
                    current_pos += 1.0

            if meas_idx < len(measurements) - 1:
                current_pos += gap_between_measurements

        ax.set_title(f"{subplot_label} {anatomy_name}", fontweight="bold", fontsize=16)

        bp = ax.boxplot(
            box_data,
            positions=positions,
            labels=box_labels,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(marker="+", markerfacecolor="black", markersize=8,
                            linestyle="none", markeredgecolor="black"),
            widths=0.7,
            showmeans=True,
            meanline=True,
            meanprops=dict(color="#941100", linewidth=2, linestyle="-"),
        )

        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_edgecolor("black")
            patch.set_linewidth(1.5)

        for whisker in bp["whiskers"]:
            whisker.set_color("black")
            whisker.set_linewidth(1.5)

        for cap in bp["caps"]:
            cap.set_color("black")
            cap.set_linewidth(1.5)

        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(2)

        ax.set_ylabel("Error (mm)", fontweight="bold", fontsize=14)
        ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)
        # Full range
        # ax.set_ylim(0, y_max)
        # Limited range
        ax.set_ylim(0, 30)
        # Log scale
        # ax.set_yscale("symlog", linthresh=1.0, linscale=1.0)  # linear up to 1mm, log after
        # ax.set_ylim(0, None)
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_map["FP"], edgecolor="black", label="FP"),
        Patch(facecolor=color_map["HC18"], edgecolor="black", label="HC18"),
        Patch(facecolor=color_map["UCL"], edgecolor="black", label="UCL"),
        Patch(facecolor=color_map["MULTICENTRE"], edgecolor="black", label="M-C"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=4, fontsize=12,
               frameon=True, fancybox=False, shadow=False)
    fig.subplots_adjust(top=0.92)

    out_png = base_dir / f"{eval_dataset.lower()}_error_boxplots.png"
    out_pdf = base_dir / f"{eval_dataset.lower()}_error_boxplots.pdf"
    out_svg = base_dir / f"{eval_dataset.lower()}_error_boxplots.svg"

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_svg, bbox_inches="tight")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_svg}")

    plt.show()

    # Summary
    print("\n=== Summary Statistics ===")
    for measurement in ["BPD", "OFD", "TAD", "APAD", "FL"]:
        print(f"\n{measurement}:")
        for model in MODELS_PER_MEASUREMENT[measurement]:
            data = df_plot[(df_plot["Measurement"] == measurement) & (df_plot["Model"] == model)]["Error (mm)"]
            if len(data) > 0:
                print(f"  {model}: Mean={data.mean():.2f} mm, Median={data.median():.2f} mm, "
                      f"Std={data.std():.2f} mm, Min={data.min():.2f} mm, Max={data.max():.2f} mm")


if __name__ == "__main__":
    create_boxplots(EVAL_DATASET)