# MULTICENTRE Dataset (Combined Dataset)

## Overview

The MULTICENTRE dataset is the **complete combined dataset** containing all images from the three primary datasets (FP, HC18, UCL). It contains **4,517 ultrasound images** from **1,904 unique subjects** acquired at **three clinical sites** using **seven different ultrasound devices**.

This combined dataset represents the full multi-centre, multi-device benchmark described in the paper and is designed for experiments on cross-site generalization, domain adaptation, and robustness evaluation.

⚠️ **Important**:  
The MULTICENTRE dataset contains **all** images from FP, HC18, and UCL combined. The individual datasets (FP, HC18, UCL) are subsets of MULTICENTRE, not separate additions. The total unique count is **1,904 subjects** and **4,517 images** as reported in the paper (Table 3).

## Dataset Characteristics

- **Number of subjects**: 1,904
- **Number of images**: 4,517  
- **Clinical sites**: 3 (Barcelona, Netherlands, London)  
- **Ultrasound devices**:
  - GE Voluson series (E6, E8, S8, S10, 730)
  - Aloka Prosound
  - Other institutional devices
- **Anatomies covered**: Head, Abdomen, Femur  
- **Image format**: JPEG and PNG  

Per-anatomy image counts (as reported in Table 3 of the paper):

| Anatomy | Total Images |
|---------|--------------|
| Head    | 2,798        |
| Abdomen | 825          |
| Femur   | 895          |
| **Total** | **4,517**  |

### Breakdown by source dataset:

| Source | Head | Abdomen | Femur | Total |
|--------|------|---------|-------|-------|
| FP     | 1,638 | 693    | 760   | 3,091 |
| HC18   | 999  | -      | -     | 999   |
| UCL    | 161  | 131    | 135   | 427   |
| **MULTICENTRE** | **2,798** | **825** | **895** | **4,517** |

## Directory Structure

```
MULTICENTRE/
├── annotations/
│   ├── Head.csv          # Complete head annotations (2,798 images)
│   ├── Head_Train.csv    # Training split
│   ├── Head_Test.csv     # Test split
│   ├── Abdomen.csv       # Complete abdomen annotations (825 images)
│   ├── Abdomen_Train.csv # Training split
│   ├── Abdomen_Test.csv  # Test split
│   ├── Femur.csv         # Complete femur annotations (895 images)
│   ├── Femur_Train.csv   # Training split
│   └── Femur_Test.csv    # Test split
└── data/
    ├── Head/             # Head ultrasound images (2,798 images)
    ├── Abdomen/          # Abdomen ultrasound images (825 images)
    └── Femur/            # Femur ultrasound images (895 images)
```


## Image Naming Convention

Images in the Multi-centre subset inherit naming conventions from their source datasets (FP, HC18, UCL), so multiple patterns appear:

**Pattern 1: Numeric identifiers (often HC18-style)**
```
[0-9]+[Anatomy].[jpeg|png]
```

Examples:

- `009Abdomen.jpeg`
- `0053Abdomen.jpeg`
- `0352_HC.png`

**Pattern 2: Patient–Operator–Experience based (UCL-style)**
```
Patient[ID]-Operator[ID]-[Experience]-[DateTime]_[Anatomy].[jpeg|png]
```

Examples:

- `Patient001-Novice-20200117112103_FL_measured.jpeg`
- `Patient014-MD-Expert-20200211113328_head1.jpeg`
- `Patient020-Operator100-Expert-20200310-121358_Head_meas.jpeg`

**Pattern 3: Test-based naming**
```
Test[N]-[DateTime]_[Anatomy].[jpeg|png]
```

Examples:

- `Test1-20191217111958_BPD.jpeg`
- `Test7-20191217150710_FL_measured.jpeg`

**Notes:**

- File extensions may be `.jpeg` or `.png`  
- Some filenames include suffixes like `_measured`, `_meas`, or `_meas_val` indicating images with on-screen clinical measurements  
- Operator experience (`Novice`, `Expert`, etc.) and acquisition timestamps are sometimes encoded in filenames  

## Annotation Format

Annotations follow the same CSV schema as the primary datasets.

### Head Measurements (BPD and OFD)

**Files**: `annotations/Head.csv`, `annotations/Head_Train.csv`, `annotations/Head_Test.csv`

**Columns:**
```csv
index,image_name,scale,center_w,center_h,ofd_1_x,ofd_1_y,ofd_2_x,ofd_2_y,bpd_1_x,bpd_1_y,bpd_2_x,bpd_2_y,px_to_mm_rate,mm_dist,Algo,SubjectID,Split
```

**Measurements:**
- **BPD (Bi-parietal Diameter)**: Transverse diameter of the fetal skull
  - `bpd_1_x, bpd_1_y`: First landmark
  - `bpd_2_x, bpd_2_y`: Second landmark

- **OFD (Occipito-frontal Diameter)**: Longitudinal diameter of the fetal skull
  - `ofd_1_x, ofd_1_y`: First landmark
  - `ofd_2_x, ofd_2_y`: Second landmark

### Abdomen Measurements (TAD and APAD)

**File**: `annotations/Abdomen.csv`, `annotations/Abdomen_Train.csv`, `annotations/Abdomen_Test.csv`

**Columns:**
```csv
index,image_name,scale,center_w,center_h,tad_1_x,tad_1_y,tad_2_x,tad_2_y,apad_1_x,apad_1_y,apad_2_x,apad_2_y,px_to_mm_rate,mm_dist,Algo,SubjectID,Split
```


**Measurements:**

- **TAD (Transverse Abdominal Diameter)**: Transverse diameter across the abdomen  
  - `tad_1_x, tad_1_y`: First landmark  
  - `tad_2_x, tad_2_y`: Second landmark  

- **APAD (Anterior–Posterior Abdominal Diameter)**: Anterior–posterior diameter  
  - `apad_1_x, apad_1_y`: First landmark  
  - `apad_2_x, apad_2_y`: Second landmark  

### Femur Measurements (FL)

**Files**: `annotations/Femur.csv`, `annotations/Femur_Train.csv`, `annotations/Femur_Test.csv`

**Columns:**
```csv
index,image_name,scale,center_w,center_h,fl_1_x,fl_1_y,fl_2_x,fl_2_y,px_to_mm_rate,mm_dist,Algo,SubjectID,Split
```


**Measurements:**

- **FL (Femur Length)**: Length of the femoral diaphysis  
  - `fl_1_x, fl_1_y`: Proximal end  
  - `fl_2_x, fl_2_y`: Distal end  

## Data Preprocessing

Images in the MULTICENTRE dataset have undergone the same preprocessing steps as in their source datasets:

1. **Format handling**: Original JPEG/PNG formats preserved  
2. **Resizing**: Images resized based on acquisition device and protocol; the `scale` field records the scaling factor  
3. **Coordinate scaling**: Landmark coordinates adjusted to the resized image space  
4. **Quality filtering**: Images verified for correct standard plane and anatomical completeness  

## Data Splits

The dataset is split into training and test sets with **subject-disjoint** partitioning inherited from the source datasets (FP, HC18, UCL):

⚠️ **Important**:

- Images from the same subject appear only in one split (train or test), preventing data leakage  
- Split information is encoded in the `Split` column and in the `*_Train.csv` / `*_Test.csv` files  
- The splits are inherited from the individual datasets to maintain consistency with published results

For exact train/test counts per anatomy, please refer to the corresponding CSV files (`Head_Train.csv`, `Head_Test.csv`, etc.).

## Multi-Site and Multi-Device Characteristics

This combined dataset is particularly valuable for studying domain shift and generalization because it brings together images from:

- **Multiple devices** (GE Voluson E6/E8/S8/S10/730, Aloka Prosound, others)  
- **Multiple sites** with:
  - Different clinical protocols (Barcelona, Netherlands, London)
  - Different acquisition settings
  - Operators with varying experience levels  

It can be used to:

- Evaluate robustness of models trained on a single site when tested across multiple sites  
- Study the impact of device and protocol variation on automated biometry performance  
- Develop domain-adaptive and domain-generalizable models

## Clinical Relevance

The MULTICENTRE dataset reflects real-world deployment conditions where:

- Different institutions and devices produce heterogeneous image appearances  
- Sonographers with different levels of expertise acquire the images  
- Imaging protocols may not be perfectly standardized across sites  

This makes it suitable for:

- **Multi-centre generalization studies**  
- **Domain adaptation and domain generalization research**  
- **Operator variability analysis**  
- **Simulation of multi-centre clinical evaluation scenarios**  
- **Benchmarking model robustness across sites and devices**

## Quality Control

Annotations in this dataset:

- Originate from expert annotation processes at each clinical site (FP, HC18, UCL)  
- Were verified for anatomical plane correctness  
- Underwent consistency checks for measurement plausibility and landmark placement  
- Follow standardized biometry measurement protocols (ISUOG guidelines)

## Notes

- **Combined dataset**: MULTICENTRE contains all images from FP, HC18, and UCL  
- **Image formats**: Both JPEG and PNG appear, depending on the source dataset  
- **Filename conventions**: Multiple naming patterns reflect different source datasets and acquisition protocols  
- **Missing fields**: Some entries may have empty `SubjectID` or `Algo` fields, inherited from the source annotations  
- **Annotation methods**: Note that HC18 landmarks were derived via ellipse fitting from segmentation masks, while FP and UCL used manual VIA annotation

## Comparison with Source Datasets (Paper Table 3)

The MULTICENTRE dataset is the complete union of the following source datasets:

| Dataset     | Subjects | Images | Anatomies             |
|-------------|----------|--------|------------------------|
| FP          | 1,047    | 3,091  | Head (1,638), Abdomen (693), Femur (760)  |
| HC18        | 806      | 999    | Head only (999)             |
| UCL         | 51       | 427    | Head (161), Abdomen (131), Femur (135)  |
| **MULTICENTRE** | **1,904** | **4,517** | **Head (2,798), Abdomen (825), Femur (895)** |

⚠️ **Important**: The individual datasets (FP, HC18, UCL) are **subsets** of MULTICENTRE, not additional datasets. The total unique count is **1,904 subjects** and **4,517 images** as reported in the paper.
