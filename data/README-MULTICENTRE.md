# Multi-centre Dataset (Derived Subset)

## Overview

The multi-centre subset is a **derived, curated view** constructed from the three primary datasets (FP, HC18, UCL). It contains **613 ultrasound images** from **148 unique subjects** acquired at multiple clinical sites using different ultrasound devices.

This subset is designed to emphasise **inter-site and inter-device variability** and is intended for experiments on domain shift, generalisation, and robustness. All images in this subset also appear in the primary datasets (FP, HC18, UCL); **Multi-centre does not introduce any new images or subjects beyond those described in the paper**.

⚠️ **Important**:  
Do **not** add the Multi-centre counts to FP/HC18/UCL when computing global totals. It is an overlapping, convenience subset.

## Dataset Characteristics

- **Number of subjects**: 148 (subset of the 1,904 subjects described in the paper)  
- **Number of images**: 613 (subset of the 4,517 images described in the paper)  
- **Clinical sites represented**: multiple (Barcelona, Netherlands, London)  
- **Ultrasound devices** (as inherited from source datasets):
  - GE Voluson series (E6, E8, S8, S10, 730, etc.)
  - Aloka Prosound
  - Other institutional devices present in UCL data
- **Anatomies covered**: Head, Abdomen, Femur  
- **Image format**: JPEG and PNG  

Per-anatomy image counts in this subset (for reference):

| Anatomy | Total Images |
|---------|--------------|
| Head    | 252          |
| Abdomen | 149          |
| Femur   | 212          |
| **Total** | **613**    |

## Directory Structure

```
Multi-centre/
├── annotations/
│   ├── Head.csv          # Complete head annotations
│   ├── Head_Train.csv    # Training split
│   ├── Head_Test.csv     # Test split
│   ├── Abdomen.csv       # Complete abdomen annotations
│   ├── Abdomen_Train.csv # Training split
│   ├── Abdomen_Test.csv  # Test split
│   ├── Femur.csv         # Complete femur annotations
│   ├── Femur_Train.csv   # Training split
│   └── Femur_Test.csv    # Test split
└── data/
    ├── Head/             # Head ultrasound images
    ├── Abdomen/          # Abdomen ultrasound images
    └── Femur/            # Femur ultrasound images
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

Images in the Multi-centre subset have undergone the same preprocessing steps as in their source datasets:

1. **Format handling**: Original JPEG/PNG formats preserved  
2. **Resizing**: Images resized based on acquisition device and protocol; the `scale` field records the scaling factor  
3. **Coordinate scaling**: Landmark coordinates adjusted to the resized image space  
4. **Quality filtering**: Images verified for correct standard plane and anatomical completeness  

## Data Splits

The subset is split into training and test sets with **subject-disjoint** partitioning:

| Anatomy | Total Images | Train Images | Test Images |
|---------|--------------|--------------|-------------|
| Head    | 252          | 180          | 72          |
| Abdomen | 149          | 126          | 23          |
| Femur   | 212          | 176          | 36          |
| **Total** | **613**    |              |             |

⚠️ **Important**:

- Images from the same subject appear only in one split (train or test), preventing data leakage  
- Split information is encoded in the `Split` column and in the `*_Train.csv` / `*_Test.csv` files  

## Multi-Site and Multi-Device Characteristics

This subset is particularly valuable for studying domain shift because it brings together images from:

- **Multiple devices** (e.g., GE Voluson family, Aloka, others)  
- **Multiple sites** with:
  - Different clinical protocols
  - Different acquisition settings
  - Operators with varying experience levels  

It can be used to:

- Evaluate robustness of models trained on a single site when tested across multiple sites  
- Study the impact of device and protocol variation on automated biometry performance  

## Clinical Relevance

The Multi-centre subset reflects real-world deployment conditions where:

- Different institutions and devices produce heterogeneous image appearances  
- Sonographers with different levels of expertise acquire the images  
- Imaging protocols may not be perfectly standardised  

This makes it suitable for:

- **Generalisation studies**  
- **Domain adaptation and domain generalisation research**  
- **Operator variability analysis**  
- **Simulation of multi-centre clinical evaluation scenarios**  

## Quality Control

Annotations in this subset:

- Originate from the same expert annotation processes used in FP, HC18, and UCL  
- Were verified for anatomical plane correctness  
- Underwent consistency checks for measurement plausibility and landmark placement  

## Notes

- **Overlap**: All images in Multi-centre are also present in FP, HC18, or UCL; this subset is overlapping by design  
- **Image formats**: Both JPEG and PNG appear, depending on the source dataset  
- **Filename conventions**: Multiple naming patterns reflect different source datasets and acquisition protocols  
- **Missing fields**: Some entries may have empty `SubjectID` or `Algo` fields, inherited from the source annotations  

## Comparison with Primary Datasets (Paper Values)

For reference, the primary datasets described in the paper have the following characteristics:

| Dataset     | Subjects | Images | Anatomies             |
|-------------|----------|--------|------------------------|
| FP          | 1,047    | 3,091  | Head, Abdomen, Femur  |
| HC18        | 806      | 999    | Head only             |
| UCL         | 51       | 427    | Head, Abdomen, Femur  |
| **Combined**| 1,904    | 4,517  | All above             |
| **Multi-centre** (this subset) | 148 | 613 | Head, Abdomen, Femur (subset of above) |

Again, **Multi-centre should not be counted as an additional dataset** when quoting the total number of images or subjects.
