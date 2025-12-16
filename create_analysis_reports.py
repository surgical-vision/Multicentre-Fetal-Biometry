#!/usr/bin/env python3
"""
Author: Chiara Di Vece
Date: 2025-12-16

Bland-Altman plot analysis for fetal biometry measurements.

Generates Bland–Altman agreement plots between GT and predicted measurements.
Supports absolute difference or percent difference (recommended for heteroscedasticity).
Optionally removes outliers using a transparent rule (Tukey IQR).
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import font_manager
from scipy.spatial import distance


# -------------------- Setup --------------------

def setup_fonts() -> None:
    """Register and set up the Lato font."""
    font_dirs = ["fonts/Lato"]
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)

    # Prefer Lato but keep a fallback
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Lato", "DejaVu Sans"]
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"


def change_to_project_dir() -> Path:
    """Change to the Multicentre-Fetal-Biometry directory if needed."""
    current_dir = Path.cwd()
    if current_dir.name == "Multicentre-Fetal-Biometry":
        return current_dir
    if (current_dir / "Multicentre-Fetal-Biometry").exists():
        os.chdir("Multicentre-Fetal-Biometry")
        return Path.cwd()
    parent_dir = current_dir.parent
    if (parent_dir / "Multicentre-Fetal-Biometry").exists():
        os.chdir(str(parent_dir / "Multicentre-Fetal-Biometry"))
        return Path.cwd()
    print("Warning: Could not find Multicentre-Fetal-Biometry directory; staying put.")
    return current_dir


# -------------------- Measurement helpers --------------------

def safe_denom(denom: np.ndarray, eps: float) -> np.ndarray:
    """
    Protect against tiny denominators in percent differences.
    Anything with |denom| < eps becomes NaN so it gets filtered out downstream.
    """
    denom = np.asarray(denom, dtype=float)
    return np.where(np.abs(denom) < eps, np.nan, denom)


def metric_coord_cols(metric: str):
    """Return the 4 GT coordinate column names for a metric (lowercase)."""
    m = metric.lower()
    return [m + "_1_x", m + "_1_y", m + "_2_x", m + "_2_y"]


def compute_pairs_from_df_and_preds(
    df: pd.DataFrame,
    predictions,
    metric: str,
    display: bool = False,
    image_root: Optional[str] = None
) -> np.ndarray:
    """
    Returns Nx2 array with [GT_distance_px, Pred_distance_px] after aligned filtering.

    Filters out rows with invalid GT coords (<0 or NaN) and invalid predictions (NaN/Inf).
    Applies the same filtering logic as the dataset class to ensure alignment.
    """
    cols = metric_coord_cols(metric)
    img_col = df.columns[0]

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing GT columns for metric {metric}: {missing}")

    # Dataset-like filtering:
    # 1) missing image name
    image_name_mask = df.iloc[:, 0].isna()

    # 2) invalid landmarks (negative or NaN)
    gt_xy = df[cols].to_numpy(dtype=float)  # (N, 4)
    landmark_mask = ((gt_xy < 0) | np.isnan(gt_xy)).any(axis=1)

    dataset_filter_mask = image_name_mask | landmark_mask
    df_filtered = df[~dataset_filter_mask].reset_index(drop=True)

    # Predictions to numpy
    pred = torch.as_tensor(predictions).detach().cpu().numpy()
    if pred.ndim != 3 or pred.shape[1:] != (2, 2):
        raise ValueError(f"Expected predictions shape (N,2,2), got {pred.shape}")

    # Lengths must match after dataset filtering
    if len(df_filtered) != pred.shape[0]:
        raise ValueError(
            f"Length mismatch after dataset filtering: df={len(df_filtered)} vs preds={pred.shape[0]}. "
            "This indicates predictions were generated with different filtering logic/order."
        )

    # Filter invalid predictions
    invalid_pred = np.isnan(pred).any(axis=(1, 2)) | np.isinf(pred).any(axis=(1, 2))

    # Safety: invalid GT (should be none after filtering, but keep it explicit)
    gt_xy_filtered = df_filtered[cols].to_numpy(dtype=float)
    invalid_gt = np.isnan(gt_xy_filtered).any(axis=1) | (gt_xy_filtered < 0).any(axis=1)

    keep = ~(invalid_gt | invalid_pred)
    df_f = df_filtered.loc[keep].reset_index(drop=True)
    pred_f = pred[keep]

    pairs = []
    for j in range(len(df_f)):
        x1, y1, x2, y2 = df_f.loc[j, cols].astype(float).tolist()
        gt_pts = np.array([[x1, y1], [x2, y2]], dtype=float)
        pr_pts = pred_f[j].astype(float)

        gt_d = distance.euclidean(gt_pts[0], gt_pts[1])
        pr_d = distance.euclidean(pr_pts[0], pr_pts[1])

        if np.isfinite(gt_d) and np.isfinite(pr_d):
            pairs.append((gt_d, pr_d))

        if display:
            if image_root is None:
                raise ValueError("display=True requires image_root to be set")
            img_path = os.path.join(image_root, str(df_f.loc[j, img_col]))
            img = cv2.imread(img_path)
            if img is not None:
                plt.figure()
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.title(str(df_f.loc[j, img_col]))
                plt.plot([x1, x2], [y1, y2], "-", linewidth=2, label="GT")
                plt.plot(pr_pts[:, 0], pr_pts[:, 1], "-", linewidth=2, label="Pred")
                plt.legend()
                plt.show()

    return np.array(pairs, dtype=float)


# -------------------- Outlier removal --------------------

def ba_diff_values(
    gt: np.ndarray,
    pr: np.ndarray,
    mode: str,
    symmetric_percent: bool,
    eps_denom: float
) -> Tuple[np.ndarray, str]:
    """Compute the BA y-values (diff) used for plotting and outlier detection."""
    gt = np.asarray(gt, dtype=float)
    pr = np.asarray(pr, dtype=float)

    if mode == "absolute":
        diff = pr - gt
        unit = "px"
        return diff, unit

    if mode == "percent":
        mean_x = (gt + pr) / 2.0
        if symmetric_percent:
            denom = safe_denom(mean_x, eps=eps_denom)
        else:
            denom = safe_denom(gt, eps=eps_denom)
        diff = 100.0 * (pr - gt) / denom
        unit = "%"
        return diff, unit

    raise ValueError("mode must be 'percent' or 'absolute'")


def tukey_iqr_inliers(values: np.ndarray, k: float = 2.5):
    """
    Tukey IQR inlier mask.
    Returns (mask, lo, hi). Mask is True for inliers.
    """
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(values)
    v = values[ok]

    if v.size < 4:
        # Not enough data to define IQR robustly.
        mask = np.ones_like(values, dtype=bool) & ok
        return mask, np.nan, np.nan

    q1, q3 = np.percentile(v, [25, 75])
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr

    mask = ok & (values >= lo) & (values <= hi)
    return mask, lo, hi


def remove_outliers(
    gt: np.ndarray,
    pr: np.ndarray,
    mode: str,
    symmetric_percent: bool,
    eps_denom: float,
    method: str = "iqr",
    iqr_k: float = 2.5,
    tag: str = ""
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove outliers based on BA difference values.
    Prints how many points were removed and the rule.
    Also logs how many were dropped due to tiny denominators (percent mode).
    """
    gt = np.asarray(gt, dtype=float)
    pr = np.asarray(pr, dtype=float)

    diff, unit = ba_diff_values(gt, pr, mode=mode, symmetric_percent=symmetric_percent, eps_denom=eps_denom)

    if method == "none":
        # Still worth logging denom drops in percent mode, if any
        if mode == "percent" and tag:
            n_bad = int(np.sum(~np.isfinite(diff)))
            if n_bad > 0:
                print(f"[{tag}] Dropped {n_bad}/{diff.size} points due to tiny/invalid denominator (eps={eps_denom:g})")
                print(f"[BA] Dropped {n_bad} points due to tiny/invalid GT denominator (eps={1e-6})")
        ok = np.isfinite(diff)
        return gt[ok], pr[ok]

    if method == "iqr":
        # First: denom/NaN filtering info (separate from IQR outliers)
        ok = np.isfinite(diff)
        n_bad = int(np.sum(~ok))
        if mode == "percent" and tag and n_bad > 0:
            print(f"[{tag}] Dropped {n_bad}/{diff.size} points due to tiny/invalid denominator (eps={eps_denom:g})")

        inlier, lo, hi = tukey_iqr_inliers(diff, k=iqr_k)
        removed = int(np.sum(~inlier))
        total = int(diff.size)

        if tag:
            if np.isfinite(lo) and np.isfinite(hi):
                print(
                    f"[{tag}] Outliers removed: {removed}/{total} using Tukey IQR "
                    f"(k={iqr_k:.2f}, bounds {lo:.2f}{unit}..{hi:.2f}{unit})"
                )
            else:
                print(
                    f"[{tag}] Outliers removed: {removed}/{total} using Tukey IQR "
                    f"(k={iqr_k:.2f}, bounds unavailable: too few points)"
                )

        return gt[inlier], pr[inlier]

    raise ValueError("Unknown outlier method. Use 'none' or 'iqr'.")


# -------------------- Plotting --------------------

def bland_altman_plot(
    ax,
    gt,
    pr,
    mode="percent",
    symmetric_percent=True,
    eps_denom: float = 1e-6,
    color_face="#5D94A6",
    color_edge="#404040",
    color_mean="#9b0a0e",
    color_loa="#0c6783",
    decimals=2
):
    """
    Draw a Bland–Altman plot.
    - x-axis: mean of GT and Pred (pixels)
    - y-axis: absolute difference (Pred-GT) or percent difference.
    """
    gt = np.asarray(gt, dtype=float)
    pr = np.asarray(pr, dtype=float)

    mean_x = (gt + pr) / 2.0

    if mode == "absolute":
        diff_y = pr - gt
        y_label = "Difference [pixels]"
        y_unit = " px"
    elif mode == "percent":
        if symmetric_percent:
            denom = safe_denom(mean_x, eps=eps_denom)
            diff_y = 100.0 * (pr - gt) / denom
        else:
            denom = safe_denom(gt, eps=eps_denom)
            diff_y = 100.0 * (pr - gt) / denom
        y_label = "Percent difference [%]"
        y_unit = " %"
    else:
        raise ValueError("mode must be 'percent' or 'absolute'")

    ok = np.isfinite(mean_x) & np.isfinite(diff_y)
    mean_x = mean_x[ok]
    diff_y = diff_y[ok]

    bias = float(np.mean(diff_y))
    sd = float(np.std(diff_y, ddof=1))  # sample SD
    upper = bias + 1.96 * sd
    lower = bias - 1.96 * sd

    ax.scatter(
        mean_x, diff_y,
        c=color_face, edgecolors=color_edge,
        linewidths=1.5, s=50, alpha=0.7
    )

    ax.axhline(bias, color=color_mean, linestyle="-", linewidth=2)
    ax.axhline(upper, color=color_loa, linestyle="--", linewidth=1.8)
    ax.axhline(lower, color=color_loa, linestyle="--", linewidth=1.8)

    # X-axis is always pixels: mean of two pixel distances
    ax.set_xlabel("Mean of GT and Pred [pixels]", fontweight="bold")
    ax.set_ylabel(y_label, fontweight="bold")
    
    # Increase tick label font size
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Calculate y-axis span for positioning labels above lines
    span = max(abs(upper - bias), abs(bias - lower))
    ax.set_ylim(bias - 1.25 * span, bias + 1.25 * span)
    
    # Position labels within frame, right above each line
    xlim = ax.get_xlim()
    x_text = xlim[0] + 0.95 * (xlim[1] - xlim[0])  # 95% from left, within frame
    y_offset = span * 0.03  # Small offset above each line (3% of span)

    fmt = "{:." + str(decimals) + "f}"
    ax.text(
        x_text, bias + y_offset, "mean diff: " + fmt.format(bias) + y_unit,
        va="bottom", ha="right", color=color_mean, fontweight="bold", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor=color_mean)
    )
    ax.text(
        x_text, upper + y_offset, "+1.96 SD: " + fmt.format(upper) + y_unit,
        va="bottom", ha="right", color=color_loa, fontweight="bold", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor=color_loa)
    )
    ax.text(
        x_text, lower + y_offset, "-1.96 SD: " + fmt.format(lower) + y_unit,
        va="bottom", ha="right", color=color_loa, fontweight="bold", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor=color_loa)
    )

    return bias, lower, upper


# -------------------- Main --------------------

def main():
    project_dir = change_to_project_dir()
    setup_fonts()
    sys.path.insert(0, str(project_dir.parent))

    folders = ["FP", "UCL", "HC18"]

    anatomy_config = {
        "FP": {
            "brain":   {"key": "Head",    "anatomy": "brain",   "metrics": ["BPD", "OFD"]},
            "abdomen": {"key": "Abdomen", "anatomy": "abdomen", "metrics": ["TAD", "APAD"]},
            "femur":   {"key": "Femur",   "anatomy": "femur",   "metrics": ["FL"]},
        },
        "UCL": {
            "brain":   {"key": "Head",    "anatomy": "brain",   "metrics": ["BPD", "OFD"]},
            "abdomen": {"key": "Abdomen", "anatomy": "abdomen", "metrics": ["TAD", "APAD"]},
            "femur":   {"key": "Femur",   "anatomy": "femur",   "metrics": ["FL"]},
        },
        "HC18": {
            "brain":   {"key": "Head",    "anatomy": "brain",   "metrics": ["BPD", "OFD"]},
        }
    }

    MODE = "percent"                 # "percent" recommended for heteroscedasticity
    SYMMETRIC_PERCENT = False         # percent diff vs mean (recommended)

    OUTLIERS = "iqr"                 # "none" or "iqr"
    IQR_K = 2.5                      # 1.5 standard, 2.5-3.0 more conservative

    EPS_DENOM = 1e-6                 # tiny denom threshold for percent diffs

    for folder in folders:
        for anatomy_name, cfg in anatomy_config[folder].items():
            key = cfg["key"]
            anatomy = cfg["anatomy"]

            out_dir = Path(f"output/FETAL/{folder}_figs")
            out_dir.mkdir(parents=True, exist_ok=True)

            for metric in cfg["metrics"]:
                # Load GT annotations
                if folder == "HC18":
                    df = pd.read_csv(f"data/annotations/{folder}/Head_Test.csv")
                else:
                    df = pd.read_csv(f"data/annotations/{folder}/{key}_Test.csv")

                # Drop junk index cols
                junk = [c for c in df.columns if str(c).lower() == "index" or str(c).startswith("Unnamed")]
                if junk:
                    df = df.drop(columns=junk)

                # Load predictions
                pred_path = (
                    f"output/FETAL/fetal_landmark_hrnet_w18_{folder}_{anatomy}_{metric}"
                    f"/predictions_on_{folder}.pth"
                )
                predictions = torch.load(pred_path)

                pairs = compute_pairs_from_df_and_preds(df, predictions, metric, display=False)
                if pairs.shape[0] < 2:
                    print(f"Skipping {folder} {anatomy} {metric}: not enough valid pairs ({pairs.shape[0]})")
                    continue

                gt = pairs[:, 0]
                pr = pairs[:, 1]

                # Outlier removal (transparent, logged)
                tag = f"{folder}-{anatomy}-{metric}"
                gt, pr = remove_outliers(
                    gt, pr,
                    mode=MODE,
                    symmetric_percent=SYMMETRIC_PERCENT,
                    eps_denom=EPS_DENOM,
                    method=OUTLIERS,
                    iqr_k=IQR_K,
                    tag=tag
                )

                if gt.size < 2:
                    print(f"Skipping {folder} {anatomy} {metric}: too few points after filtering ({gt.size})")
                    continue

                fig, ax = plt.subplots(figsize=(7, 5))
                bias, loa_lo, loa_hi = bland_altman_plot(
                    ax, gt, pr,
                    mode=MODE,
                    symmetric_percent=SYMMETRIC_PERCENT,
                    eps_denom=EPS_DENOM,
                    decimals=2
                )

                ax.set_title(f"{folder} {metric} ({anatomy_name})", fontsize=14, fontweight="bold")
                plt.tight_layout()

                suffix = f"_{OUTLIERS}" if OUTLIERS != "none" else ""
                png_path = out_dir / f"{anatomy_name.capitalize()}_{metric}{suffix}.png"
                svg_path = out_dir / f"{anatomy_name.capitalize()}_{metric}{suffix}.svg"
                fig.savefig(str(png_path), dpi=600)
                fig.savefig(str(svg_path))
                plt.close(fig)

                print(f"{folder} {anatomy} {metric}: bias={bias:.2f}, LoA=[{loa_lo:.2f}, {loa_hi:.2f}] ({MODE})")


if __name__ == "__main__":
    main()