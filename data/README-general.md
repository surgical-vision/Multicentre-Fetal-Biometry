# Fetal Biometry Dataset

This repository contains a comprehensive, multicentre, multi-device benchmark dataset for landmark-based fetal biometry estimation from ultrasound images.

## Overview

The dataset contains **4,513 de-identified ultrasound images** from **1,904 unique subjects** acquired at **four clinical sites** using **seven different ultrasound devices**. Expert anatomical landmark annotations are provided for clinically used fetal biometric measurements.

### Biometric Measurements

The dataset covers all major fetal biometry measurements:

- **Head**: Bi-parietal diameter (BPD) and Occipito-frontal diameter (OFD)
- **Abdomen**: Transverse abdominal diameter (TAD) and Anterior–posterior abdominal diameter (APAD)
- **Femur**: Femur length (FL)

## Dataset Structure

The repository is organised into three **primary subsets** (as described in the paper) and one **derived multicentre subset** for convenience:

```
fetalbiometrydata/
├── FP/                  # Fetal Planes dataset (1,047 subjects, 3,090 images)
│   ├── annotations/     # Landmark annotations (CSV format)
│   └── data/           # Ultrasound images (PNG format)
├── HC18/               # Head Circumference 2018 challenge (806 subjects, 999 images)
│   ├── annotations/     # Landmark annotations (CSV format)
│   └── data/           # Ultrasound images (PNG format)
├── MULTICENTRE/        # Multicentre combined dataset (1,904 subjects, 4,513 images)
│   ├── annotations/     # Landmark annotations (CSV format)
│   └── data/           # Ultrasound images (JPEG/JPG/PNG format)
└── UCL/                # UCL dataset (51 subjects, 424 images)
    ├── annotations/     # Landmark annotations (CSV format)
    └── data/           # Ultrasound images (JPEG/JPG/PNG format)
```

### Dataset Characteristics (Paper Values)

Primary subsets (as used in the Scientific Reports paper):

| Dataset   | Sites      | Devices (examples)                               | Subjects | Images | Anatomies             |
|----------|------------|---------------------------------------------------|----------|--------|------------------------|
| **FP**   | 2 (Spain)  | GE Voluson E6/S8/S10, Aloka Prosound             | 1,047    | 3,090  | Head (1,637), Abdomen (693), Femur (760)  |
| **HC18** | 1 (NL)     | GE Voluson E8/730                                | 806      | 999    | Head only (999)             |
| **UCL**  | 1 (UK)     | GE Voluson (single institutional protocol)       | 51       | 424    | Head (159), Abdomen (130), Femur (135)  |
| **MULTICENTRE (combined)** | 3 | 7 device types                         | 1,904    | 4,513  | Head (2,795), Abdomen (823), Femur (895)             |

⚠️ **Important**:  
The **MULTICENTRE (M-C)** dataset is the complete combined dataset containing all images from FP, HC18, and UCL. It represents the full **4,513 images** and **1,904 unique subjects** (4 fewer images than originally reported due to removal of images with missing metric coordinates).

**Total**: 3 clinical sites, 7 device types, **1,904** unique subjects, **4,513** images.

## Data Organization

### Annotation Files

Each dataset folder contains CSV annotation files with landmark coordinates:

- **`[Anatomy].csv`**: Complete dataset with all annotations
- **`[Anatomy]_Train.csv`**: Training split (subject-disjoint)
- **`[Anatomy]_Test.csv`**: Test split (subject-disjoint)

where `[Anatomy]` is one of: `Head`, `Abdomen`, or `Femur` (HC18 provides `Head` only).

### Image Files

Image files are organised by anatomy type inside each dataset folder:

- **FP**:  
  - `FP/data/Head/Patient[ID]_Plane[N]_[M]_of_[K].png`  
  - `FP/data/Abdomen/...`  
  - `FP/data/FL/...` (Femur)
- **HC18**:  
  - `HC18/data/Head/[ID]_HC.png` or `[ID]_[N]HC.png`
- **UCL**:  
  - `UCL/data/Head/[filename].[jpeg|jpg|png]` (15 Head images converted from PNG to JPG to avoid HC18 conflicts)
  - `UCL/data/Abdomen/[filename].[jpeg|png]`  
  - `UCL/data/Femur/[filename].[jpeg|png]`
- **Multicentre**:  
  - `Multicentre/data/[Anatomy]/[filename].[jpeg|jpg|png]` (mixed naming from FP/HC18/UCL)

### CSV Annotation Format

```csv
index,image_name,scale,center_w,center_h,fl_1_x,fl_1_y,fl_2_x,fl_2_y,px_to_mm_rate,mm_dist,Algo,SubjectID,Split
```
#### Head (BPD and OFD measurements)

- **ofd_1_x, ofd_1_y**: First landmark for occipito-frontal diameter  
- **ofd_2_x, ofd_2_y**: Second landmark for occipito-frontal diameter  
- **bpd_1_x, bpd_1_y**: First landmark for bi-parietal diameter  
- **bpd_2_x, bpd_2_y**: Second landmark for bi-parietal diameter  

#### Abdomen (TAD and APAD measurements)

- **tad_1_x, tad_1_y**: First landmark for transverse abdominal diameter  
- **tad_2_x, tad_2_y**: Second landmark for transverse abdominal diameter  
- **apad_1_x, apad_1_y**: First landmark for anterior–posterior abdominal diameter  
- **apad_2_x, apad_2_y**: Second landmark for anterior–posterior abdominal diameter  

#### Femur (FL measurement)


- **fl_1_x, fl_1_y**: First landmark for femur length (proximal end)  
- **fl_2_x, fl_2_y**: Second landmark for femur length (distal end)  

### Common Fields

All CSV files include:

- **index**: Sequential index in the dataset  
- **image_name**: Corresponding image filename  
- **scale**: Image scaling factor applied during preprocessing  
- **center_w, center_h**: Center coordinates of the region of interest  
- **px_to_mm_rate**: Pixel-to-millimetre conversion rate (when available)  
- **mm_dist**: Distance marker value (when available, typically 5 mm or 10 mm)  
- **Algo**: Algorithm or device identifier (e.g. `"AlokaFit"` for Aloka devices)  
- **SubjectID**: De-identified subject identifier  
- **Split**: Data split indicator (`Train` or `Test`)  

## Data Splits

The dataset provides standardised, **subject-disjoint** train/test splits to ensure fair and reproducible evaluation:

- **Training split**: For model development and training  
- **Test split**: For performance evaluation only  

⚠️ **Important**:  
The splits are **subject-disjoint** – images from the same subject (or pregnancy) appear only in one split (either train or test, never both).

Exact train/test image counts per anatomy are documented in the dataset-specific READMEs and can be obtained directly from the `*_Train.csv` and `*_Test.csv` files.

## Usage

Each subdirectory (`FP/`, `HC18/`, `UCL/`, `MULTICENTRE/`) contains a detailed README with:
For each dataset you can also find a detailed README_`[dataset]', where `[dataset]' is (`FP/`, `HC18/`, `UCL/`, `MULTICENTRE/`) with:

- Dataset-specific information  
- Device and acquisition details  
- Annotation conventions and measurement definitions  
- Notes on preprocessing and quality control  
- Suggested use cases  

The **MULTICENTRE** subset is intended for experiments focusing on cross-site/domain generalisation; for baseline experiments matching the paper, use FP, HC18, and UCL as described in the manuscript.


## License

Please refer to the `LICENSE` file for terms of use.

## Contact

For questions or issues regarding the dataset, please open an issue on this GitHub repository or contact the corresponding author.

## Acknowledgments

This work was supported by the Wellcome/EPSRC Centre for Interventional and Surgical Sciences (WEISS) [203145/Z/16/Z], EPSRC [EP/P027938/1, EP/R004080/1, EP/P012841/1], and Kamin Grants [63418, 72126] from the Israel Innovation Authority.
