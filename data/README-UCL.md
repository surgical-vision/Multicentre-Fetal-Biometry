# UCL Dataset

## Overview

The UCL dataset consists of a curated subset of **427 ultrasound images** from **51 pregnancies** acquired at University College London Hospital (UCLH). This dataset provides fetal biometry measurements on standard planes with expert landmark annotations from a single clinical site.

## Dataset Characteristics

- **Number of pregnancies**: 51  
- **Number of images**: 427  
- **Clinical sites**: 1 (University College London Hospital)  
- **Ultrasound devices**:
  - GE Voluson series (single institutional protocol)
- **Anatomies covered**: Head, Abdomen, Femur  
- **Image format**: JPG or JPEG

Per-anatomy image counts (as reported in the paper):

- **Head**: 161 images  
- **Abdomen**: 131 images  
- **Femur**: 135 images  

## Directory Structure

```
UCL/
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
    ├── Head/             # Head ultrasound images (JPEG)
    ├── Abdomen/          # Abdomen ultrasound images (JPEG)
    └── Femur/            # Femur ultrasound images (JPEG)
```
## Image Naming Convention

Images in the UCL datase follow a simplified naming pattern:
```
[ID]_HC.jpg or [ID]_HC.jpeg
[ID]_AC.jpg or [ID]_AC.jpeg
[ID]_FL.jpg or [ID]_FL.jpeg
```

or for multiple images from the same subject:
```
[ID]_[N]HC.jpg or [ID]_[N]HC.jpeg
[ID]_[N]AC.jpg or [ID]_[N]AC.jpeg
[ID]_[N]FL.jpg or [ID]_[N]FL.jpeg
```


**Examples:**

- `005_3HC.jpeg` – Third head image with ID 005
- `001_AC.jpg` – First abdomen image from subject ID 001
- `002_FL.jpeg` – First femur image from subject ID 002

**Notes:**

- Image IDs are sequential but not necessarily continuous
- Some subjects have multiple acquisitions indicated by the `_[N]` prefix before `HC`, `AC`, and `FL`

## Annotation Format

### Head Measurements (BPD and OFD)

**Files**: `annotations/Head.csv`, `annotations/Head_Train.csv`, `annotations/Head_Test.csv`

**Columns:**
```csv
index,image_name,scale,center_w,center_h,ofd_1_x,ofd_1_y,ofd_2_x,ofd_2_y,bpd_1_x,bpd_1_y,bpd_2_x,bpd_2_y,px_to_mm_rate,mm_dist,Algo,SubjectID,Split
```


**Measurements:**

- **BPD (Bi-parietal Diameter)**: Transverse diameter of the fetal head  
  - `bpd_1_x, bpd_1_y`: First landmark (left parietal bone)  
  - `bpd_2_x, bpd_2_y`: Second landmark (right parietal bone)  

- **OFD (Occipito-frontal Diameter)**: Longitudinal diameter of the fetal head  
  - `ofd_1_x, ofd_1_y`: First landmark (frontal bone)  
  - `ofd_2_x, ofd_2_y`: Second landmark (occipital bone)  

### Abdomen Measurements (TAD and APAD)

**Files**: `annotations/Abdomen.csv`, `annotations/Abdomen_Train.csv`, `annotations/Abdomen_Test.csv`

**Columns:**
```csv
index,image_name,scale,center_w,center_h,tad_1_x,tad_1_y,tad_2_x,tad_2_y,apad_1_x,apad_1_y,apad_2_x,apad_2_y,px_to_mm_rate,mm_dist,Algo,SubjectID,Split
```


**Measurements:**

- **TAD (Transverse Abdominal Diameter)**: Transverse diameter across the abdomen at the level of the stomach and portal vein  
  - `tad_1_x, tad_1_y`: First landmark (left side)  
  - `tad_2_x, tad_2_y`: Second landmark (right side)  

- **APAD (Anterior–Posterior Abdominal Diameter)**: Anterior–posterior diameter perpendicular to TAD  
  - `apad_1_x, apad_1_y`: First landmark (anterior)  
  - `apad_2_x, apad_2_y`: Second landmark (posterior)  

### Femur Measurements (FL)

**Files**: `annotations/Femur.csv`, `annotations/Femur_Train.csv`, `annotations/Femur_Test.csv`

**Columns:**
```csv
index,image_name,scale,center_w,center_h,fl_1_x,fl_1_y,fl_2_x,fl_2_y,px_to_mm_rate,mm_dist,Algo,SubjectID,Split
```


**Measurements:**

- **FL (Femur Length)**: Length of the femoral diaphysis excluding epiphyses  
  - `fl_1_x, fl_1_y`: Proximal end of the femur  
  - `fl_2_x, fl_2_y`: Distal end of the femur  

## Data Preprocessing

All images in the UCL dataset have undergone standardized preprocessing:

1. **Format handling**: JPEG compression considered in quality assessment  
2. **Resizing**: Images resized to standard dimensions according to device; the `scale` field records the exact scaling factor applied  
3. **Coordinate scaling**: Landmark coordinates adjusted for rescaled images (provided in resized image coordinates)  
4. **Normalization**: Pixel intensities normalized to reduce appearance variability between acquisitions  

The `px_to_mm_rate` and `mm_dist` fields (when available) support conversion from pixels to millimetres.

## Data Splits

The UCL dataset is split into training and test sets with **subject-disjoint** partitioning for each anatomy. Split assignments are encoded in the `Split` column and in the corresponding `*_Train.csv` and `*_Test.csv` files.

The total number of images per anatomy matches the values reported in the paper:

| Anatomy | Total Images |
|---------|--------------|
| Head    | 161          |
| Abdomen | 131          |
| Femur   | 135          |

For exact train/test counts, please refer to the corresponding CSV files (`Head_Train.csv`, `Head_Test.csv`, etc.).

⚠️ **Important**:

- No pregnancy appears in both train and test splits  
- Multiple images from the same pregnancy, when present, are assigned to the same split  

## Clinical Protocol

The UCL dataset follows ISUOG (International Society of Ultrasound in Obstetrics and Gynecology) guidelines for fetal biometry.

### Head measurements

- **Plane**: Transventricular (standard head plane for HC/BPD/OFD)  
- **Key features**:
  - Cavum septum pellucidum (CSP)  
  - Thalami in midline  
  - Smooth, symmetric cranial vault  
- **Measurements**: BPD and OFD measured outer-to-outer

### Abdomen measurements

- **Plane**: Transverse abdomen at the level of the stomach and portal vein bifurcation  
- **Key features**:
  - Approximately circular abdomen
  - Visible stomach bubble
  - Portal vein branching
- **Measurements**: TAD and APAD measured at the skin line

### Femur measurements

- **Plane**: Long axis of the femur  
- **Key features**:
  - Full visualization of the femoral diaphysis
  - Horizontal or near-horizontal alignment
- **Measurement**: FL measured along the ossified diaphysis, excluding epiphyses

## Quality Control

All images and annotations underwent rigorous quality control:

1. **Plane verification**: Each image verified to show the correct standard plane for the intended measurement  
2. **Annotation review**: Landmarks placed and checked by expert sonographers  
3. **Measurement validation**: Biometric values checked against expected ranges for gestational age  
4. **Consistency checks**: Multiple acquisitions from the same pregnancy checked for internal consistency  
5. **Outlier detection**: Automated checks to flag potential annotation errors for manual review  

## Operator Experience

The dataset includes acquisitions from operators with varying levels of experience (e.g., novice vs expert). In many filenames, operator identifiers and experience level are encoded, enabling:

- Analysis of inter-operator variability  
- Studies of measurement reproducibility  
- Training/education research on sonographer performance  

## Notes

- Images are JPEG compressed; minor compression artefacts may be present  
- All images are de-identified according to institutional protocols  
- Some entries may have missing `SubjectID` or `Algo` fields  
- Filenames may include acquisition timestamps and measurement suffixes (e.g. `_measured`)  

## Comparison with Other Datasets

| Feature        | UCL              | FP                      | HC18                  |
|----------------|------------------|-------------------------|-----------------------|
| Site           | 1 (UCLH)         | 2 (Barcelona)           | 1 (Netherlands)       |
| Devices        | GE Voluson       | GE Voluson + Aloka      | GE Voluson E8/730     |
| Images         | 427              | 3,091                   | 999                   |
| Pregnancies    | 51               | 1,047                   | 806                   |
| Anatomies      | Head, Abdomen, Femur | Head, Abdomen, Femur | Head only             |
| Format         | JPEG             | PNG                     | PNG                   |
| Operator info  | Yes              | No                      | No                    |

## Ethical Considerations

- All images are fully de-identified according to UCLH and UCL policies  
- The study was approved by the appropriate Research Ethics Committee (IRAS ID 230125)  
- Written informed consent was obtained from all participants  
- No identifying information is present in images or annotations  