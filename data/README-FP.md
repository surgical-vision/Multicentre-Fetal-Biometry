# FP (Fetal Planes) Dataset

## Overview

The Fetal Planes (FP) dataset is the largest subset in this collection, containing **3,090 ultrasound images** from **1,047 unique subjects**. The dataset was acquired at two clinical sites in Barcelona using multiple ultrasound devices, and provides coverage of all major fetal biometric measurements on standard planes.

## Dataset Characteristics

- **Number of subjects**: 1,047
- **Number of images**: 3,090  
- **Clinical sites**: 2 (Vall d'Hebron and Hospital Sant Joan de Déu, Barcelona)
- **Ultrasound devices**:
  - GE Voluson E6 (three scanners)
  - GE Voluson S8
  - GE Voluson S10
  - Aloka Prosound (α7 or equivalent)
- **Anatomies covered**: Head, Abdomen, Femur
- **Image format**: PNG

Per-anatomy image counts:

- **Head**: 1,637 images (Train: 757, Test: 880)
- **Abdomen**: 693 images (Train: 568, Test: 125)
- **Femur**: 760 images (Train: 437, Test: 323)  

## Directory Structure

```
FP/
├── annotations/
│   ├── Head.csv          # Complete head annotations (with BPD and OFD)
│   ├── Head_Train.csv    # Training split
│   ├── Head_Test.csv     # Test split
│   ├── Abdomen.csv       # Complete abdomen annotations (with TAD and APAD)
│   ├── Abdomen_Train.csv # Training split
│   ├── Abdomen_Test.csv  # Test split
│   ├── Femur.csv         # Complete femur annotations (with FL)
│   ├── Femur_Train.csv   # Training split
│   └── Femur_Test.csv    # Test split
└── data/
    ├── Head/             # Head ultrasound images
    ├── Abdomen/          # Abdomen ultrasound images
    └── Femur/               # Femur ultrasound images
```

## Image Naming Convention

Images follow the naming pattern:
```
Patient[SubjectID]_Plane[PlaneNumber]_[ImageNumber]_of_[TotalImages].png
```


**Examples:**

- `Patient00168_Plane2_1_of_2.png` – First of two images from Patient 168, Plane 2  
- `Patient00647_Plane2_1_of_1.png` – Single image from Patient 647, Plane 2  

**Notes:**

- `Plane2` typically indicates the abdominal plane  
- `Plane3` typically indicates the head plane  
- `Plane5` typically indicates the femur plane  
- Multiple images from the same patient/plane represent different acquisitions or views  

## Annotation Format

### Head Measurements (BPD and OFD)

**Files**: `annotations/Head.csv`, `annotations/Head_Train.csv`, `annotations/Head_Test.csv`

**Columns:**
```csv
index,image_name,scale,center_w,center_h,ofd_1_x,ofd_1_y,ofd_2_x,ofd_2_y,bpd_1_x,bpd_1_y,bpd_2_x,bpd_2_y,px_to_mm_rate,mm_dist,Algo,SubjectID,Split
```


**Measurements:**

- **BPD (Bi-parietal Diameter)**: Distance between the outer edges of the parietal bones, measured perpendicular to the falx cerebri  
  - `bpd_1_x, bpd_1_y`: First landmark
  - `bpd_2_x, bpd_2_y`: Second landmark

- **OFD (Occipito-frontal Diameter)**: Maximum diameter from the frontal bone to the occipital bone  
  - `ofd_1_x, ofd_1_y`: First landmark
  - `ofd_2_x, ofd_2_y`: Second landmark

### Abdomen Measurements (TAD and APAD)

**Files**: `annotations/Abdomen.csv`, `annotations/Abdomen_Train.csv`, `annotations/Abdomen_Test.csv`

**Columns:**
```csv
index,image_name,scale,center_w,center_h,tad_1_x,tad_1_y,tad_2_x,tad_2_y,apad_1_x,apad_1_y,apad_2_x,apad_2_y,px_to_mm_rate,mm_dist,Algo,SubjectID,Split
```


**Measurements:**

- **TAD (Transverse Abdominal Diameter)**: Transverse diameter across the abdomen  
  - `tad_1_x, tad_1_y`: First landmark
  - `tad_2_x, tad_2_y`: Second landmark

- **APAD (Anterior-Posterior Abdominal Diameter)**: Anterior–posterior diameter  
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
  - `fl_1_x, fl_1_y`: Proximal end of the femur  
  - `fl_2_x, fl_2_y`: Distal end of the femur  

## Data Preprocessing

All images in the FP dataset have undergone standardized preprocessing:

1. **Resizing and normalization**: Images were resized to standard dimensions based on the acquisition device; intensity values were normalized for consistent appearance.  
2. **Coordinate scaling**: Landmark coordinates are provided in the rescaled image space.  
3. **Device indicator (`Algo` field)**:
   - `"AlokaFit"`: Images acquired with Aloka Prosound devices (with device-specific preprocessing)  
   - Empty string: Images acquired with GE Voluson devices  

## Data Splits

The FP dataset is provided with **subject-disjoint** train/test splits for each anatomy. Split assignments are encoded in:

- `Head_Train.csv`, `Head_Test.csv`  
- `Abdomen_Train.csv`, `Abdomen_Test.csv`  
- `Femur_Train.csv`, `Femur_Test.csv`  

The total number of images per anatomy:

| Anatomy | Total Images | Train | Test |
|---------|--------------|-------|------|
| Head    | 1,637        | 757   | 880  |
| Abdomen | 693          | 568   | 125  |
| Femur   | 760          | 437   | 323  |

## Clinical Relevance

The FP dataset represents a comprehensive protocol for fetal biometry assessment following clinical guidelines (ISUOG Practice Guidelines). The measurements included are used for:

- **Fetal growth assessment**: Monitoring fetal development across gestational age  
- **Gestational age estimation**: Dating pregnancy based on biometric measurements  
- **Detection of growth abnormalities**: Identifying intrauterine growth restriction (IUGR) or macrosomia  
- **Abdominal circumference (AC)**: Can be approximated from TAD and APAD using  
  \[
  \mathrm{AC} \approx \pi \times \frac{\mathrm{TAD} + \mathrm{APAD}}{2}
  \]
- **Head circumference (HC)**: Can be approximated from BPD and OFD using  
  \[
  \mathrm{HC} \approx \pi \times \frac{\mathrm{BPD} + \mathrm{OFD}}{2}
  \]

## Quality Control

All annotations in this dataset were:

- Performed by expert sonographers with extensive clinical experience  
- Verified for anatomical correctness and measurement accuracy  
- Cross-checked for consistency across multiple acquisitions from the same subject  

## Notes

- The `px_to_mm_rate` field may be empty for images where calibration information was not available.  
- The `mm_dist` field indicates the distance marker value visible in the ultrasound image (typically 5 mm or 10 mm).  
- Images with `Algo="AlokaFit"` have specific preprocessing applied for the Aloka device family.  
