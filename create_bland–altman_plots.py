#!/usr/bin/env python3
"""
Author: Chiara Di Vece
Date: 2025-12-17

Bland–Altman plot analysis for fetal biometry measurements.

Key design (same idea as boxplots):
- Within-dataset evaluation: FP model on FP test set, UCL model on UCL test set, HC18 model on HC18 test set.
- Filter the evaluation CSV first (px_to_mm present + valid GT landmarks), write a filtered CSV.
- Point cfg.DATASET.TESTSET to the filtered CSV so dataset order == GT order.
- Re-run inference (no cached predictions) to avoid order/filter mismatches.
- Compute distances in *mm* (pixel distance * px_to_mm) BEFORE Bland–Altman.
- Support absolute difference (mm) or percent difference (asymmetric vs GT, or symmetric vs mean).
- Optional outlier removal via Tukey IQR on the BA y-values (transparent rule).

Usage:
  python create_bland–altman_plots.py

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
from matplotlib import pyplot as plt
from matplotlib import font_manager
from scipy.spatial import distance

# Add repo root to path to import lib modules
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

import lib.models as models
from lib.config import config as BASE_CONFIG
from lib.datasets import get_dataset
from lib.core import function


# -------------------- User knobs --------------------

# Within-dataset evaluation: each model evaluated on its own test set
# (FP model on FP test set, UCL model on UCL test set, HC18 model on HC18 test set)
FOLDERS = ["FP", "UCL", "HC18"]

MODE = "percent"               # "percent" or "absolute"
SYMMETRIC_PERCENT = False      # False => percent diff vs GT (asymmetric). True => vs mean (symmetric).
OUTLIERS = "iqr"               # "none" or "iqr"
IQR_K = 2.5
EPS_DENOM = 1e-6               # denom threshold in percent mode

# Output
OUT_ROOT = Path("output/FETAL")
TMP_DIR = OUT_ROOT / "_tmp_filtered_csv"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Cache filtered CSVs so you don’t print the same line 12 times.
_FILTER_CACHE: Dict[Tuple[str, str], Tuple[pd.DataFrame, Path, str]] = {}


# -------------------- Plot style --------------------

def setup_fonts() -> None:
    font_dirs = [str(SCRIPT_DIR / "fonts" / "Lato")]
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for ff in font_files:
        font_manager.fontManager.addfont(ff)

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Lato", "DejaVu Sans"]
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"


# -------------------- Measurement definitions --------------------

MEASUREMENTS = {
    "BPD":  {"anatomy": "brain",   "csv_prefix": "Head",    "landmark_prefix": "bpd"},
    "OFD":  {"anatomy": "brain",   "csv_prefix": "Head",    "landmark_prefix": "ofd"},
    "TAD":  {"anatomy": "abdomen", "csv_prefix": "Abdomen", "landmark_prefix": "tad"},
    "APAD": {"anatomy": "abdomen", "csv_prefix": "Abdomen", "landmark_prefix": "apad"},
    "FL":   {"anatomy": "femur",   "csv_prefix": "Femur",   "landmark_prefix": "fl"},
}

ANATOMY_CONFIG = {
    "brain":   {"key": "Head",    "metrics": ["BPD", "OFD"]},
    "abdomen": {"key": "Abdomen", "metrics": ["TAD", "APAD"]},
    "femur":   {"key": "Femur",   "metrics": ["FL"]},
}


# -------------------- Helpers --------------------

def safe_denom(denom: np.ndarray, eps: float) -> np.ndarray:
    denom = np.asarray(denom, dtype=float)
    return np.where(np.abs(denom) < eps, np.nan, denom)


def calculate_distance_pixels(x1, y1, x2, y2) -> float:
    return float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))


def find_eval_csv(eval_dataset: str, csv_prefix: str) -> Path:
    candidates = [
        Path("data/annotations") / eval_dataset / "px_to_mm" / f"{csv_prefix}_Test.csv",
        Path("data/annotations") / eval_dataset / f"{csv_prefix}_Test.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    tried = "\n  - " + "\n  - ".join(str(x) for x in candidates)
    raise FileNotFoundError(f"Could not find eval CSV for {eval_dataset}/{csv_prefix}. Tried:{tried}")


def filtered_csv_path(eval_dataset: str, csv_prefix: str, metric: str) -> Path:
    return TMP_DIR / f"{eval_dataset}_{csv_prefix}_Test_{metric}_filtered.csv"


def filter_and_write_eval_csv(eval_dataset: str, metric: str) -> Tuple[pd.DataFrame, Path, str]:
    """
    Returns (df_filtered, filtered_csv_path, px_col_name)
    Filters: image present, GT coords valid, px_to_mm valid (>0, finite).
    """
    key_cache = (eval_dataset, metric)
    if key_cache in _FILTER_CACHE:
        return _FILTER_CACHE[key_cache]

    csv_prefix = MEASUREMENTS[metric]["csv_prefix"]
    lm = MEASUREMENTS[metric]["landmark_prefix"]
    in_csv = find_eval_csv(eval_dataset, csv_prefix)

    df = pd.read_csv(in_csv)

    junk = [c for c in df.columns if str(c).lower() == "index" or str(c).startswith("Unnamed")]
    if junk:
        df = df.drop(columns=junk)

    if "px_to_mm_rate" in df.columns:
        px_col = "px_to_mm_rate"
    elif "px_to_mm" in df.columns:
        px_col = "px_to_mm"
    else:
        raise KeyError(f"{in_csv} missing px_to_mm column (expected 'px_to_mm_rate' or 'px_to_mm').")

    img_col = df.columns[0]
    coords = [f"{lm}_1_x", f"{lm}_1_y", f"{lm}_2_x", f"{lm}_2_y"]
    missing = [c for c in ([img_col] + coords + [px_col]) if c not in df.columns]
    if missing:
        raise KeyError(f"{in_csv} missing required columns for {metric}: {missing}")

    # same spirit as dataset filters + your mm requirement
    mask_img = df[img_col].isna()

    gt_xy = df[coords].to_numpy(dtype=float)
    mask_lm = ((gt_xy < 0) | np.isnan(gt_xy) | (~np.isfinite(gt_xy))).any(axis=1)

    px = pd.to_numeric(df[px_col], errors="coerce").to_numpy(dtype=float)
    mask_px = (~np.isfinite(px)) | np.isnan(px) | (px <= 0)

    keep = ~(mask_img | mask_lm | mask_px)
    df_f = df.loc[keep].reset_index(drop=True)

    out_csv = filtered_csv_path(eval_dataset, csv_prefix, metric)
    df_f.to_csv(out_csv, index=False)

    dropped = int((~keep).sum())
    total = int(len(df))
    print(f"[{eval_dataset}-{metric}] Filtered CSV: kept {len(df_f)}/{total} (dropped {dropped}) -> {out_csv}")

    _FILTER_CACHE[key_cache] = (df_f, out_csv, px_col)
    return df_f, out_csv, px_col


def load_checkpoint(model_folder: Path) -> Tuple[dict, Optional[np.ndarray]]:
    model_file = model_folder / "final_state.pth"
    if not model_file.exists():
        model_file = model_folder / "latest.pth"
    if not model_file.exists():
        raise FileNotFoundError(f"No checkpoint found in {model_folder} (final_state.pth / latest.pth)")

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


def compute_d_vect_from_training_cfg(model_name: str, anatomy: str, metric: str) -> Optional[np.ndarray]:
    cfg_file = Path("experiments/fetal") / f"fetal_landmark_hrnet_w18_{model_name}_{anatomy}_{metric}.yaml"
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


def model_dir_name(model_name: str, anatomy: str, metric: str) -> str:
    return f"fetal_landmark_hrnet_w18_{model_name}_{anatomy}_{metric}"


def run_inference_on_filtered_csv(
    trained_model: str,
    eval_dataset: str,
    anatomy: str,
    metric: str,
    filtered_csv: Path,
    expected_len: int,
) -> np.ndarray:
    """
    Loads weights from the *trained_model* checkpoint but runs on eval_dataset filtered CSV.
    """
    ckpt_dir = Path("output/FETAL") / model_dir_name(trained_model, anatomy, metric)
    weights, learned_d_vect = load_checkpoint(ckpt_dir)

    if learned_d_vect is None:
        learned_d_vect = compute_d_vect_from_training_cfg(trained_model, anatomy, metric)
        if learned_d_vect is None:
            print(f"[{trained_model}-{metric}] Warning: d_vect unavailable (may affect endpoint ordering).")

    # Use eval_dataset cfg as baseline so dataset settings match evaluation domain
    eval_cfg_file = Path("experiments/fetal") / f"fetal_landmark_hrnet_w18_{eval_dataset}_{anatomy}_{metric}.yaml"
    if not eval_cfg_file.exists():
        raise FileNotFoundError(f"Eval cfg not found: {eval_cfg_file}")

    cfg = BASE_CONFIG.clone()
    cfg.defrost()
    cfg.merge_from_file(str(eval_cfg_file))
    cfg.DATASET.TESTSET = str(filtered_csv)
    cfg.MODEL.INIT_WEIGHTS = False
    cfg.freeze()

    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLED

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
    with torch.no_grad():
        _, _, _, predictions = function.inference(cfg, test_loader, model)

    preds = predictions.detach().cpu().numpy()
    if preds.shape[0] != expected_len:
        raise RuntimeError(
            f"[{trained_model} on {eval_dataset} - {metric}] preds/GT mismatch: "
            f"preds={preds.shape[0]} vs expected={expected_len}"
        )
    return preds


def compute_gt_pred_distances_mm(
    df_f: pd.DataFrame,
    preds: np.ndarray,
    metric: str,
    px_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (gt_mm, pr_mm) as 1D arrays.
    Expects df_f and preds aligned and same length.
    """
    lm = MEASUREMENTS[metric]["landmark_prefix"]
    cols = [f"{lm}_1_x", f"{lm}_1_y", f"{lm}_2_x", f"{lm}_2_y"]

    if preds.ndim != 3 or preds.shape[1:] != (2, 2):
        raise ValueError(f"Expected preds shape (N,2,2), got {preds.shape}")
    if len(df_f) != preds.shape[0]:
        raise ValueError(f"df/preds length mismatch: df={len(df_f)} preds={preds.shape[0]}")

    gt_mm = []
    pr_mm = []

    for i, row in df_f.iterrows():
        x1, y1, x2, y2 = row[cols].astype(float).tolist()
        gt_px = calculate_distance_pixels(x1, y1, x2, y2)

        p1 = preds[i, 0, :]
        p2 = preds[i, 1, :]
        pr_px = calculate_distance_pixels(float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1]))

        px_to_mm = float(row[px_col])
        gt_mm.append(gt_px * px_to_mm)
        pr_mm.append(pr_px * px_to_mm)

    return np.asarray(gt_mm, dtype=float), np.asarray(pr_mm, dtype=float)


# -------------------- Outliers (same logic) --------------------

def ba_diff_values(
    gt: np.ndarray,
    pr: np.ndarray,
    mode: str,
    symmetric_percent: bool,
    eps_denom: float
) -> Tuple[np.ndarray, str]:
    gt = np.asarray(gt, dtype=float)
    pr = np.asarray(pr, dtype=float)

    if mode == "absolute":
        return (gt - pr), "mm"

    if mode == "percent":
        mean_x = (gt + pr) / 2.0
        denom = safe_denom(mean_x, eps_denom) if symmetric_percent else safe_denom(gt, eps_denom)
        return (100.0 * (gt - pr) / denom), "%"

    raise ValueError("mode must be 'percent' or 'absolute'")


def tukey_iqr_inliers(values: np.ndarray, k: float = 2.5):
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(values)
    v = values[ok]

    if v.size < 4:
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
    tag: str = "",
) -> Tuple[np.ndarray, np.ndarray]:
    diff, unit = ba_diff_values(gt, pr, mode, symmetric_percent, eps_denom)

    ok = np.isfinite(diff)
    n_bad = int(np.sum(~ok))
    if mode == "percent" and tag and n_bad > 0:
        print(f"[{tag}] Dropped {n_bad}/{diff.size} points due to tiny/invalid denominator (eps={eps_denom:g})")

    if method == "none":
        return gt[ok], pr[ok]

    if method == "iqr":
        inlier, lo, hi = tukey_iqr_inliers(diff, k=iqr_k)
        removed = int(np.sum(~inlier))
        total = int(diff.size)

        if tag:
            if np.isfinite(lo) and np.isfinite(hi):
                print(f"[{tag}] Outliers removed: {removed}/{total} using Tukey IQR "
                      f"(k={iqr_k:.2f}, bounds {lo:.2f}{unit}..{hi:.2f}{unit})")
            else:
                print(f"[{tag}] Outliers removed: {removed}/{total} using Tukey IQR "
                      f"(k={iqr_k:.2f}, bounds unavailable: too few points)")

        return gt[inlier], pr[inlier]

    raise ValueError("Unknown outlier method. Use 'none' or 'iqr'.")


# -------------------- Bland–Altman plotting (mm) --------------------

def bland_altman_plot(
    ax,
    gt_mm,
    pr_mm,
    mode="percent",
    symmetric_percent=SYMMETRIC_PERCENT,
    eps_denom: float = 1e-6,
    color_face="#5D94A6",
    color_edge="#404040",
    color_mean="#9b0a0e",
    color_loa="#0c6783",
    decimals=2,
):
    gt = np.asarray(gt_mm, dtype=float)
    pr = np.asarray(pr_mm, dtype=float)

    mean_x = (gt + pr) / 2.0  # mm

    if mode == "absolute":
        diff_y = gt - pr
        y_label = "Difference [mm]"
        y_unit = " mm"
    elif mode == "percent":
        denom = safe_denom(mean_x, eps_denom) if symmetric_percent else safe_denom(gt, eps_denom)
        diff_y = 100.0 * (gt - pr) / denom
        y_label = "Difference [%]"
        y_unit = " %"
    else:
        raise ValueError("mode must be 'percent' or 'absolute'")

    ok = np.isfinite(mean_x) & np.isfinite(diff_y)
    mean_x = mean_x[ok]
    diff_y = diff_y[ok]

    bias = float(np.mean(diff_y))
    sd = float(np.std(diff_y, ddof=1))
    upper = bias + 1.96 * sd
    lower = bias - 1.96 * sd

    ax.scatter(mean_x, diff_y, c=color_face, edgecolors=color_edge, linewidths=1.5, s=50)
    ax.axhline(bias, color=color_mean, linestyle="-", linewidth=1.5)
    ax.axhline(upper, color=color_loa, linestyle="--", linewidth=1.5)
    ax.axhline(lower, color=color_loa, linestyle="--", linewidth=1.5)

    ax.set_xlabel("Mean [mm]", fontweight="bold")
    ax.set_ylabel(y_label, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=14)

    span = max(abs(upper - bias), abs(bias - lower))
    ax.set_ylim(bias - 1.25 * span, bias + 1.25 * span)

    xlim = ax.get_xlim()
    x_text = xlim[0] + 0.95 * (xlim[1] - xlim[0])
    y_offset = span * 0.03

    fmt = "{:." + str(decimals) + "f}"
    ax.text(x_text, bias + y_offset, "mean diff: " + fmt.format(bias) + y_unit,
            va="bottom", ha="right", color=color_mean, fontweight="bold", fontsize=14,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor=color_mean))
    ax.text(x_text, upper + y_offset, "+1.96 SD: " + fmt.format(upper) + y_unit,
            va="bottom", ha="right", color=color_loa, fontweight="bold", fontsize=14,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor=color_loa))
    ax.text(x_text, lower + y_offset, "-1.96 SD: " + fmt.format(lower) + y_unit,
            va="bottom", ha="right", color=color_loa, fontweight="bold", fontsize=14,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor=color_loa))

    return bias, lower, upper


# -------------------- Main --------------------

def main():
    setup_fonts()

    for folder in FOLDERS:
        print(f"\n{'='*60}")
        print(f"Processing folder: {folder}")
        print(f"{'='*60}\n")

        # Output directory for this folder (matching original notebook structure)
        folder_out = OUT_ROOT / f"{folder}_figs"
        folder_out.mkdir(parents=True, exist_ok=True)

        # Define anatomy config based on folder
        if folder == "FP" or folder == "UCL":
            anatomy_config = ANATOMY_CONFIG
        else:  # HC18 only has brain
            anatomy_config = {"brain": ANATOMY_CONFIG["brain"]}

        for anatomy, acfg in anatomy_config.items():
            metrics = acfg["metrics"]

            for metric in metrics:
                # Filter CSV for this folder's test set
                df_f, filtered_csv, px_col = filter_and_write_eval_csv(folder, metric)

                # Run inference: model trained on 'folder' evaluated on 'folder' test set (within-dataset)
                preds = run_inference_on_filtered_csv(
                    trained_model=folder,
                    eval_dataset=folder,
                    anatomy=anatomy,
                    metric=metric,
                    filtered_csv=filtered_csv,
                    expected_len=len(df_f),
                )

                gt_mm, pr_mm = compute_gt_pred_distances_mm(df_f, preds, metric, px_col)

                # Outliers based on BA y-values (mm or %)
                tag = f"{folder}-{anatomy}-{metric}"
                gt_mm, pr_mm = remove_outliers(
                    gt_mm, pr_mm,
                    mode=MODE,
                    symmetric_percent=SYMMETRIC_PERCENT,
                    eps_denom=EPS_DENOM,
                    method=OUTLIERS,
                    iqr_k=IQR_K,
                    tag=tag,
                )

                if gt_mm.size < 2:
                    print(f"Skipping {tag}: too few points after filtering ({gt_mm.size})")
                    continue

                fig, ax = plt.subplots(figsize=(7, 5))
                bias, loa_lo, loa_hi = bland_altman_plot(
                    ax, gt_mm, pr_mm,
                    mode=MODE,
                    symmetric_percent=SYMMETRIC_PERCENT,
                    eps_denom=EPS_DENOM,
                    decimals=2,
                )

                pct_tag = "vsMean" if (MODE == "percent" and SYMMETRIC_PERCENT) else ("vsGT" if MODE == "percent" else "abs")
                suffix = f"_{pct_tag}_{OUTLIERS}" if OUTLIERS != "none" else f"_{pct_tag}"
                title = f"{folder} {metric} ({anatomy})"
                ax.set_title(title, fontsize=12, fontweight="bold")

                plt.tight_layout()

                png_path = folder_out / f"{anatomy.capitalize()}_{metric}{suffix}.png"
                svg_path = folder_out / f"{anatomy.capitalize()}_{metric}{suffix}.svg"
                fig.savefig(png_path, dpi=600)
                fig.savefig(svg_path)
                plt.close(fig)

                print(f"{title}: bias={bias:.2f}, LoA=[{loa_lo:.2f}, {loa_hi:.2f}] ({MODE})")


if __name__ == "__main__":
    main()