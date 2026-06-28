# Fetal Biometry Dataset  
**A multicentre benchmark dataset for comprehensive landmark-based fetal ultrasound biometry**

---

## Overview  

Accurate fetal growth assessment from ultrasound depends on the precise measurement of biometric parameters, obtained by identifying anatomical landmarks in standard fetal planes. Manual measurement is time-consuming, operator-dependent, and sensitive to variability across scanners and acquisition sites, limiting reproducibility in both clinical and research settings.  

This repository accompanies the publication *“A multicentre benchmark dataset for comprehensive landmark-based fetal ultrasound biometry”*. It provides an open-access dataset and reference code for automated fetal biometry research.  

The dataset combines ultrasound images from **three independent sources** acquired on **seven different ultrasound devices**, all of which are annotated by experts for standard fetal biometric landmarks. It supports training and evaluation of algorithms for fetal biometry estimation, growth assessment, and cross-domain generalization.  

---

## Dataset Summary  

| Dataset | Source / Institution | Subjects | Images | Anatomical Planes | Devices | Annotation Tool |
|--------|----------------------|---------:|-------:|-------------------|---------|-----------------|
| **Fetal Planes (FP)** | Vall d'Hebron & Sant Joan de Déu, Barcelona, Spain | 1,047 | 3,090 | Head (1,637), Abdomen (693), Femur (760) | GE Voluson E6/S8/S10, Aloka | VIA (manual landmarks) |
| **HC18** | Radboud University Medical Center, Netherlands | 806 | 999 | Head only | GE Voluson E8, 730 | Ellipse fitting from HC masks |
| **UCL** | University College London Hospital (UCLH), UK | 51 | 424 | Head (159), Abdomen (130), Femur (135) | GE Voluson | VIA (manual landmarks) |

Each dataset includes 2D ultrasound standard planes and corresponding landmark annotations for:  
- **Head:** biparietal diameter (BPD), occipito-frontal diameter (OFD)  
- **Abdomen:** transverse abdominal (TAD) and anterior–posterior abdominal (APAD) diameters  
- **Femur:** femur length (FL)

All images are de-identified and stored at their original variable resolutions. During training and evaluation, regions of interest are dynamically extracted and resized to 256×256 pixels via scale-aware cropping.

---

## Repository Structure  

```plaintext
Multicentre-Fetal-Biometry/
├── data/                          # Images and annotations (see detailed structure below)
├── experiments/fetal/             # Model configuration files (.yaml) for each dataset/anatomy
├── tools/                         # Training and testing scripts
│   ├── train.py
│   └── test.py
├── lib/                           # Model implementations and datasets
│   ├── models/                    # HRNet model architecture
│   ├── datasets/                  # Dataset loaders
│   ├── core/                      # Training/evaluation functions
│   └── utils/                     # Utility functions
├── hrnetv2_pretrained/            # HRNetV2 ImageNet pretrained weights
├── output/                        # Training outputs (checkpoints, logs)
├── fonts/                         # Font files for visualization
├── run_all_training.sh            # Automated script to train all models
├── run_all_tests.sh               # Automated script for cross-validation testing
├── create_error_boxplots.py       # Error analysis and visualization
├── create_bland-altman_plots.py   # Generate Bland-Altman agreement plots
├── create_train_test_matrices.py  # Generate cross-validation heatmap matrices
├── environment.yml                # Conda environment specification
├── requirements.txt               # Additional pip requirements
└── README.md
```

---

## Data Format

- **Images:** JPEG or PNG (depending on source dataset)  
- **Annotations:** CSV files with landmark coordinates and metadata (plus optional VIA JSON in some cases)  

See the dataset-specific READMEs for full column descriptions per anatomy:

- `data/README-general.md` – Overview of all datasets
- `data/README-FP.md` – Fetal Planes dataset details
- `data/README-HC18.md` – HC18 challenge dataset details
- `data/README-UCL.md` – UCL dataset details
- `data/README-MULTICENTRE.md` – Combined multicentre dataset details

Each dataset README includes:
- Number of subjects, images, and anatomical breakdowns
- Train/test split information
- CSV column descriptions
- Data acquisition details (devices, protocols)

---

## Benchmark Evaluation  

The dataset was benchmarked using **BiometryNet** (Avisdris *et al.*, MICCAI 2022), an HRNet-based landmark regression framework with Dynamic Orientation Determination (DOD). We performed comprehensive cross-validation across all datasets (FP, HC18, UCL) and the combined multicentre dataset (M-C). Results are reported as **Normalised Mean Error (NME) ± standard deviation**, where NME is unitless (measurement error normalised by inter-landmark distance).

### Comprehensive Cross-Validation Results

The table below shows cross-data evaluation results for all train–test combinations across four datasets and three anatomies. Within each training dataset block and for each biometric measurement, **bold** indicates the best (lowest) NME on each test set, and *italic* indicates the second-best.

| Train | Test | BPD | OFD | APAD | TAD | FL |
|-------|------|-----|-----|------|-----|----|
| **FP** | FP | **0.03±0.06** | **0.03±0.05** | **0.08±0.06** | **0.08±0.06** | **0.03±0.11** |
| | HC18 | 0.08±0.12 | 0.08±0.13 | — | — | — |
| | UCL | 0.38±0.26 | 0.22±0.22 | 0.31±0.23 | 0.45±0.28 | 0.90±0.54 |
| | M-C | *0.06±0.14* | *0.05±0.10* | *0.13±0.15* | *0.16±0.21* | *0.12±0.34* |
| **HC18** | FP | *0.06±0.07* | *0.06±0.07* | — | — | — |
| | HC18 | **0.05±0.09** | **0.04±0.08** | — | — | — |
| | UCL | 0.15±0.16 | 0.19±0.23 | — | — | — |
| | M-C | 0.06±0.11 | 0.07±0.11 | — | — | — |
| **UCL** | FP | *0.10±0.11* | *0.09±0.09* | 0.17±0.13 | 0.16±0.12 | 0.07±0.18 |
| | HC18 | 0.17±0.25 | 0.13±0.16 | — | — | — |
| | UCL | **0.08±0.18** | **0.05±0.11** | **0.08±0.14** | **0.08±0.14** | **0.02±0.03** |
| | M-C | 0.12±0.17 | 0.10±0.12 | *0.15±0.14* | *0.14±0.13* | *0.06±0.17* |
| **M-C** | FP | *0.03±0.05* | **0.03±0.04** | 0.08±0.06 | 0.09±0.07 | 0.03±0.10 |
| | HC18 | 0.05±0.08 | 0.04±0.07 | — | — | — |
| | UCL | **0.02±0.02** | 0.03±0.11 | **0.05±0.12** | **0.05±0.12** | **0.01±0.01** |
| | M-C | 0.04±0.07 | *0.03±0.06* | *0.07±0.08* | *0.08±0.08* | *0.03±0.09* |

**Key observations:**

- **Within-dataset performance:** All models achieve excellent performance when tested on their own dataset (diagonal entries), with NME typically < 0.10
- **Domain shift:** Significant performance degradation is observed under cross-dataset evaluation, particularly for FP→UCL and UCL→FP in femur measurements
- **Multicentre advantage:** The M-C model (trained on combined FP+HC18+UCL data) achieves the best or second-best performance across most test sets, demonstrating superior generalization
- **Head biometry:** Most robust across domains, with M-C achieving 0.02±0.02 NME on UCL for BPD
- **Abdomen biometry:** M-C models achieve 0.05±0.12 NME on UCL for both APAD and TAD
- **Femur biometry:** Most challenging for cross-domain transfer, but M-C models achieve excellent performance (0.01±0.01 NME on UCL)

**Note:** HC18 dataset contains only head measurements; therefore, no results are reported for abdomen and femur anatomies.

---

## Installation & Setup

### Environment Requirements
This code is developed using Python 3.6 and PyTorch 1.0.0 on Linux with NVIDIA GPUs. The provided `environment.yml` file specifies all dependencies. Training and testing are performed using NVIDIA GPUs with CUDA-compatible PyTorch builds. The code has been tested on Ubuntu 20.04 but should work on other Linux distributions with compatible CUDA drivers.

### Setup Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/surgical-vision/Multicentre-Fetal-Biometry
   cd Multicentre-Fetal-Biometry
   ```

2. **Create conda environment from provided file:**
   ```bash
   conda env create -f environment.yml
   conda activate fetalbiometry
   ```

   **Or install dependencies manually:**
   ```bash
   pip install torch==1.0.0 torchvision==0.2.1
   pip install -r requirements.txt
   ```

3. **Download HRNetV2 pretrained weights:**
   ```bash
   mkdir -p hrnetv2_pretrained
   # Download hrnetv2_w18_imagenet_pretrained.pth into this folder
   ```
   
   Download pretrained model: [HRNetV2-W18 ImageNet weights](https://1drv.ms/u/s!Aus8VCZ_C_33cMkPimlmClRvmpw)

### Obtain the Datasets

Download the data archives from the UCL Research Data Repository and extract them into the `data/` directory.

> [UCL Research Data Repository](https://doi.org/10.5522/04/30819911)

### Expected Data Structure

After downloading and extracting the datasets, your directory should look like:

```plaintext
Multicentre-Fetal-Biometry/
├── data/
│   ├── annotations/
│   │   ├── FP/
│   │   │   ├── Head.csv
│   │   │   ├── Head_Train.csv
│   │   │   ├── Head_Test.csv
│   │   │   ├── Abdomen.csv
│   │   │   ├── Abdomen_Train.csv
│   │   │   ├── Abdomen_Test.csv
│   │   │   ├── Femur.csv
│   │   │   ├── Femur_Train.csv
│   │   │   └── Femur_Test.csv
│   │   ├── HC18/
│   │   │   ├── Head.csv
│   │   │   ├── Head_Train.csv
│   │   │   └── Head_Test.csv
│   │   ├── UCL/
│   │   │   ├── Head.csv, Head_Train.csv, Head_Test.csv
│   │   │   ├── Abdomen.csv, Abdomen_Train.csv, Abdomen_Test.csv
│   │   │   └── Femur.csv, Femur_Train.csv, Femur_Test.csv
│   │   └── MULTICENTRE/
│   │       ├── Head.csv, Head_Train.csv, Head_Test.csv
│   │       ├── Abdomen.csv, Abdomen_Train.csv, Abdomen_Test.csv
│   │       └── Femur.csv, Femur_Train.csv, Femur_Test.csv
│   └── images/
│       ├── FP/
│       │   ├── Head/        # PNG images
│       │   ├── Abdomen/     # PNG images
│       │   └── Femur/       # PNG images
│       ├── HC18/
│       │   └── Head/        # PNG images
│       ├── UCL/
│       │   ├── Head/        # JPEG/JPG images 
│       │   ├── Abdomen/     # JPEG/PNG images
│       │   └── Femur/       # JPEG/PNG images
│       └── MULTICENTRE/
│           ├── Head/
│           ├── Abdomen/
│           └── Femur/
├── experiments/fetal/       # Configuration files for each dataset/anatomy
├── hrnetv2_pretrained/      # HRNetV2 ImageNet pretrained weights
├── tools/
├── lib/                     # Model and dataset implementations
└── output/                  # Training outputs (checkpoints, logs)
```

---

## Training Models

To train BiometryNet (HRNet-based landmark detector) on any dataset/anatomy combination:

```bash
python tools/train.py --cfg experiments/fetal/<CONFIG-FILE>.yaml
```

### Train All Models (Automated)

To train all models for all datasets and anatomies automatically:

```bash
./run_all_training.sh
```

This script will:
- Train all models for FP, HC18, UCL, and MULTICENTRE datasets
- Train separate models for each anatomy (head, abdomen, femur) and metric (BPD, OFD, TAD, APAD, FL)
- Save training logs to `output/FETAL/training_logs/`
- Clean up intermediate checkpoints to save disk space
- Use GPU 0 by default (override with `CUDA_VISIBLE_DEVICES=1 ./run_all_training.sh`)

### Examples (Individual Training)

**Train on FP dataset for Head/BPD:**
```bash
python tools/train.py --cfg experiments/fetal/fetal_landmark_hrnet_w18_FP_brain_BPD.yaml
```

**Train on UCL dataset for Abdomen/TAD:**
```bash
python tools/train.py --cfg experiments/fetal/fetal_landmark_hrnet_w18_UCL_abdomen_TAD.yaml
```

**Train on MULTICENTRE dataset for Femur/FL:**
```bash
python tools/train.py --cfg experiments/fetal/fetal_landmark_hrnet_w18_MULTICENTRE_femur_FL.yaml
```

### Available Configuration Files

All configuration files are in `experiments/fetal/`:

- **FP dataset:** `FP_brain_BPD`, `FP_brain_OFD`, `FP_abdomen_TAD`, `FP_abdomen_APAD`, `FP_femur_FL`
- **HC18 dataset:** `HC18_brain_BPD`, `HC18_brain_OFD`
- **UCL dataset:** `UCL_brain_BPD`, `UCL_brain_OFD`, `UCL_abdomen_TAD`, `UCL_abdomen_APAD`, `UCL_femur_FL`
- **MULTICENTRE:** `MULTICENTRE_brain_BPD`, `MULTICENTRE_brain_OFD`, `MULTICENTRE_abdomen_TAD`, `MULTICENTRE_abdomen_APAD`, `MULTICENTRE_femur_FL`

Training outputs (model checkpoints, logs) are saved to `output/FETAL/fetal_landmark_hrnet_w18_<DATASET>_<ANATOMY>_<MEASUREMENT>/`.

**Note:** When testing a model on different datasets, predictions are saved as `predictions_on_<DATASET>.pth` to avoid overwriting results during cross-validation experiments.

---

## Testing & Evaluation

To evaluate a trained model on a test set:

```bash
python tools/test.py --cfg <CONFIG-FILE> --model-file <MODEL-WEIGHT-PATH>
```

**Important**: Predictions are saved as `predictions_on_{DATASET}.pth` to avoid overwriting when testing the same model on multiple datasets.

### Test All Models (Automated Cross-Validation)

To run comprehensive cross-validation testing (all models on all datasets):

```bash
./run_all_tests.sh
```

This script will:
- Test each trained model on all test sets (FP, HC18, UCL, MULTICENTRE)
- Generate dataset-specific prediction files: `predictions_on_FP.pth`, `predictions_on_UCL.pth`, etc.
- Compute NME metrics for each combination
- Enable cross-domain evaluation (e.g., FP-trained model tested on UCL data)

Example output structure:
```
output/FETAL/fetal_landmark_hrnet_w18_FP_brain_BPD/
├── final_state.pth                # Model state after final epoch (used by run_all_tests.sh)
├── model_best.pth                 # Best model checkpoint (lowest validation NME)
├── predictions_on_FP.pth          # FP model tested on FP
├── predictions_on_HC18.pth        # FP model tested on HC18
├── predictions_on_UCL.pth         # FP model tested on UCL
└── predictions_on_MULTICENTRE.pth # FP model tested on MULTICENTRE
```

### Examples (Individual Testing)

**Test FP-trained model on FP test set (within-domain):**
```bash
python tools/test.py --cfg experiments/fetal/fetal_landmark_hrnet_w18_FP_brain_BPD.yaml \
                     --model-file output/FETAL/fetal_landmark_hrnet_w18_FP_brain_BPD/final_state.pth
```

**Test FP-trained model on UCL test set (cross-domain):**
```bash
python tools/test.py --cfg experiments/fetal/fetal_landmark_hrnet_w18_UCL_brain_BPD.yaml \
                     --model-file output/FETAL/fetal_landmark_hrnet_w18_FP_brain_BPD/final_state.pth
```

This reproduces the cross-domain evaluation results (e.g., FP→UCL) shown in the benchmark tables above.

> **Note**: The automated testing script `run_all_tests.sh` uses `final_state.pth` by default. You can also use `model_best.pth` (best checkpoint during training) by modifying the `--model-file` argument.

### Reproduce Paper Results

To reproduce the benchmark results in Table 2 of the paper:

1. Train models on each dataset (FP, HC18, UCL) for each anatomy/measurement
2. Evaluate each trained model on all test sets (within-domain and cross-domain)
3. The test script computes Normalised Mean Error (NME) automatically

Example workflow for Head/BPD:
```bash
# Train on FP
python tools/train.py --cfg experiments/fetal/fetal_landmark_hrnet_w18_FP_brain_BPD.yaml

# Test on FP test set (within-domain)
python tools/test.py --cfg experiments/fetal/fetal_landmark_hrnet_w18_FP_brain_BPD.yaml \
                     --model-file output/FETAL/fetal_landmark_hrnet_w18_FP_brain_BPD/final_state.pth

# Test on UCL test set (cross-domain)
python tools/test.py --cfg experiments/fetal/fetal_landmark_hrnet_w18_UCL_brain_BPD.yaml \
                     --model-file output/FETAL/fetal_landmark_hrnet_w18_FP_brain_BPD/final_state.pth
```

**Or use the automated scripts** to train and test all models:
```bash
./run_all_training.sh  # Train all models
./run_all_tests.sh     # Test all models on all datasets (cross-validation)
```

---

## Variability Analysis

Scripts for anatomical variability analysis and error visualization are provided:

### Generate Variability Plots

```bash
python data/create_variability_plots.py
```

This generates orientation, position, and size distribution plots for each anatomy (head, abdomen, femur) showing the variability in landmark placement across the MULTICENTRE dataset. Plots are saved to `data/variability_plots/MULTICENTRE/`.

The script:
- Analyzes landmark orientation (polar histograms)
- Visualizes normalized landmark positions (KDE plots)
- Shows size distributions for each measurement (histograms)
- Supports all datasets (FP, HC18, UCL, MULTICENTRE)
- Dynamically recalculates image centers from actual image dimensions

Change the dataset by editing the `DATASET` variable in the script (default: `'MULTICENTRE'`).

### Error Boxplots

```bash
python create_error_boxplots.py
```

Generates boxplots showing absolute error (in millimeters) between ground truth and predicted biometry measurements. This script:
- Supports all datasets (FP, HC18, UCL, MULTICENTRE)
- Generates per-anatomy boxplots (head, abdomen, femur)
- Requires predictions from trained models to be available
- Saves plots to `output/FETAL/error_boxplots/`

### Bland-Altman Analysis

```bash
python create_bland-altman_plots.py
```

Generates Bland-Altman agreement plots for within-dataset evaluation (FP on FP, UCL on UCL, HC18 on HC18). These plots show the agreement between ground truth and predicted measurements, displaying mean difference and 95% limits of agreement. Plots are saved to `output/FETAL/{DATASET}_figs/`.

The script:
- Computes mean difference and limits of agreement (mean ± 1.96×SD)
- Generates scatter plots with regression lines
- Applies Tukey IQR outlier filtering
- Supports all anatomies and metrics
- Requires predictions from trained models to be available

### Cross-Validation Matrix Visualization

```bash
python create_train_test_matrices.py
```

Generates heatmap matrices showing cross-validation NME results across all train-test combinations. The output is formatted for publication with:
- Square matrices for each metric (BPD, OFD, APAD, TAD, FL)
- Grouped by anatomy (Head, Abdomen, Femur)
- Shared colorbars for each anatomy group
- Custom colormap from light to dark blue
- Saved as `cross_data_metrics.png`

This script parses the results table (LaTeX format) and generates a publication-ready figure.

### Additional Tools

- **Image preprocessing:** Dynamic scale-aware cropping to 256×256 pixels with rotation augmentation (applied automatically during training)
- **Data augmentation:** Standard augmentation techniques (rotation ±30°, scaling ±25%, horizontal flipping) applied during training
- **Normalization:** Pixel intensities normalized with ImageNet mean/std during data loading

See individual scripts for detailed usage.

---

## Citation

If this dataset or code is used in your research, please cite the following:

### Paper

```bibtex
@article{divece2026multicentre,
  title={A multicentre benchmark dataset for comprehensive landmark-based fetal ultrasound biometry},
  author={Di Vece, Chiara and Mao, Zhehua and Avisdris, Netanell and Dromey, Brian and Napolitano, Raffaele and Ben Bashat, Dafna and Vasconcelos, Francisco and Stoyanov, Danail and Joskowicz, Leo and Bano, Sophia},
  journal={Scientific Reports},
  year={2026},
  publisher={Nature Publishing Group UK London}
}
```

### Dataset

```bibtex
@article{divece2026multicentredataset,
author = {Di Vece, Chiara and Mao, Zhehua and Avisdris, Netanell and Dromey, Brian and Napolitano, Raffaele and Ben Bashat, Dafna and Vasconcelos, Francisco and Stoyanov, Danail and Joskowicz, Leo and Bano, Sophia},
title = {A multicentre benchmark dataset for comprehensive landmark-based fetal ultrasound biometry},
year = {2025},
month = {12},
url = {https://rdr.ucl.ac.uk/articles/dataset/A_multi-centre_multi-device_benchmark_dataset_for_landmark-based_comprehensive_fetal_biometry/30819911},
doi = {10.5522/04/30819911.v1}
}
```
---

## Acknowledgments

This implementation builds on [HRNet for Facial Landmark Detection](https://github.com/HRNet/HRNet-Facial-Landmark-Detection), adapted for fetal biometry landmark regression. The HRNet architecture was originally developed for human pose estimation:

```bibtex
@inproceedings{SunXLW19,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Ke Sun and Bin Xiao and Dong Liu and Jingdong Wang},
  booktitle={CVPR},
  year={2019}
}

@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal={TPAMI},
  year={2019}
}
```

The BiometryNet framework with Dynamic Orientation Determination (DOD) is described in:

```bibtex
@inproceedings{avisdris2022biometrynet,
  title={BiometryNet: Landmark-based Fetal Biometry Estimation from Standard Ultrasound Planes},
  author={Avisdris, Netanell and Di Vece, Chiara and Yaqub, Mohammad and Napolitano, Raffaele and Papageorghiou, Aris T. and Noble, J. Alison and Joskowicz, Leo},
  booktitle={MICCAI},
  year={2022}
}
```

---

## License

- **Code:** Released under the [MIT LICENSE](https://github.com/surgical-vision/Fetal-Biometry-Dataset/blob/main/LICENSE).
  Permission is granted to use, copy, modify, and distribute the software for any purpose with attribution.  

- **Data:** Released under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://github.com/surgical-vision/Multicentre-Fetal-Biometry/blob/main/DATA-LICENSE) license.  
  You may share and adapt the dataset, provided that appropriate credit is given.  

---

## Contact

**Corresponding author:**  
Chiara Di Vece  
Department of Computer Science and UCL Hawkes Institute  
University College London  
📧 chiara.divece.20@ucl.ac.uk  

