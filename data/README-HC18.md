# HC18 (Head Circumference 2018) Dataset

## Overview

The HC18 dataset originates from the Grand Challenge on Head Circumference (HC) estimation held at MICCAI 2018. In this repository, we include the **999 training images** from **806 unique subjects**, as used in the paper. This dataset focuses exclusively on fetal head measurements.

**Note**: The original HC18 challenge also provided 335 separate test images from 335 subjects. These are not redistributed in this repository but are available through the original challenge organizers (see [hc18.grand-challenge.org](https://hc18.grand-challenge.org/)).

## Dataset Characteristics

- **Number of subjects**: 806
- **Number of images**: 999
- **Clinical sites**: 1 (Radboud University Medical Center, Netherlands)
- **Ultrasound devices**:
  - General Electric Voluson E8
  - General Electric Voluson 730
- **Anatomies covered**: Head only (BPD and OFD)
- **Image format**: PNG

## Directory Structure

```
HC18/
├── annotations/
│   ├── Head.csv       # Complete head annotations
│   ├── Head_Train.csv # Training split (737 images)
│   └── Head_Test.csv  # Test split (262 images)
└── data/
    └── Head/          # Head ultrasound images
```

**Note**: The `Head_Train.csv` and `Head_Test.csv` files represent subject-disjoint splits *within* the 999 training images for model development and internal validation, not the original HC18 challenge test set.

## Image Naming Convention

Images follow a simplified naming pattern:
```
[ID]_HC.png
```

or for multiple images from the same subject:
```
[ID]_[N]HC.png
```


**Examples:**

- `055_HC.png` – Single head image with ID 055
- `796_2HC.png` – Second head image from subject ID 796
- `783_4HC.png` – Fourth head image from subject ID 783

**Notes:**

- Image IDs are sequential but not necessarily continuous
- Some subjects have multiple acquisitions indicated by the `_[N]` prefix before `HC`
- All images are from the transventricular plane for head circumference measurement

## Annotation Format

### Head Measurements (BPD and OFD)

**Files**: `annotations/Head.csv`, `annotations/Head_Train.csv`, `annotations/Head_Test.csv`

**Columns:**
```csv
index,image_name,scale,center_w,center_h,ofd_1_x,ofd_1_y,ofd_2_x,ofd_2_y,bpd_1_x,bpd_1_y,bpd_2_x,bpd_2_y,SubjectID,px_to_mm_rate,Algo,Split
```


**Measurements:**

- **BPD (Bi-parietal Diameter)**: Distance between the outer edges of the parietal bones
  - `bpd_1_x, bpd_1_y`: First landmark
  - `bpd_2_x, bpd_2_y`: Second landmark

- **OFD (Occipito-frontal Diameter)**: Maximum diameter from frontal to occipital bone
  - `ofd_1_x, ofd_1_y`: First landmark
  - `ofd_2_x, ofd_2_y`: Second landmark

**Additional Fields:**

- **SubjectID**: Original subject identifier from the HC18 challenge (may be empty)
- **px_to_mm_rate**: Pixel-to-millimeter conversion rate (required for metric measurements)
- **scale**: Image scaling factor applied during preprocessing
- **center_w, center_h**: Center coordinates of the region of interest
- **Algo**: Algorithm identifier (typically empty for this dataset)
- **Split**: Data split indicator (`Train` or `Test`)

**Important**: The landmarks in this dataset were derived automatically via least-squares ellipse fitting to expert-annotated head circumference segmentation masks (see van den Heuvel et al., 2018). This differs from the manual landmark annotation protocol used in the FP and UCL subsets.

## Data Preprocessing

All images in the HC18 dataset have undergone standardized preprocessing:

1. **Resizing**: Images were resized to maintain consistent aspect ratios
2. **Normalization**: Pixel intensities normalized for consistent appearance
3. **Coordinate adjustment**: Landmark coordinates adjusted for the resized images
4. **Scaling factor**: The `scale` field indicates the transformation applied

## Data Splits

The 999-image HC18 subset in this repository is provided with subject-disjoint train/test splits for internal validation:

| Split | Images | Subjects |
|-------|--------|----------|
| Train | 737 |  |
| Test  | 262 |  |
| **Total** | **999** | **806** |

⚠️ **Important**:

- The splits are subject-disjoint (no subject appears in both train and test)
- Some subjects have multiple images, allowing for within-subject variability analysis
- For exact train/test counts, refer to `Head_Train.csv` and `Head_Test.csv`

### Calculate Head Circumference

Head circumference can be approximated from BPD and OFD using the ellipse formula:

\[
\mathrm{HC} \approx \pi \times \frac{\mathrm{BPD} + \mathrm{OFD}}{2}
\]

This assumes the head approximates an ellipse with semi-major and semi-minor axes equal to OFD/2 and BPD/2.

## Clinical Relevance

The HC18 dataset is specifically designed for head circumference estimation, which is crucial for:

- **Gestational age estimation**: HC is one of the most reliable biometric parameters for dating pregnancy
- **Fetal growth monitoring**: Tracking head growth trajectory across gestational age
- **Detection of abnormalities**: Identifying microcephaly, macrocephaly, or disproportionate head growth
- **Standardized assessment**: Following ISUOG guidelines for head circumference measurement at the transventricular plane

### Anatomical Landmarks

The transventricular plane for HC measurement should include (per van den Heuvel et al., 2018):

- A central midline falx, interrupted in the anterior third by the cavum septi pellucidi (CSP)
- The anterior (frontal) horns of the lateral ventricles
- The posterior horns (atria) of the lateral ventricles
- A smooth, continuous, symmetric cranial vault

## Original Challenge

This dataset was originally released as part of the HC18 Grand Challenge at MICCAI 2018. The challenge focused on automated head circumference measurement and included:

- **Task**: Automated segmentation and measurement of fetal head circumference
- **Evaluation metrics**: Dice coefficient for segmentation, absolute difference and Hausdorff distance for HC measurement
- **Baseline methods**: Provided as reference for comparison

For more information about the original challenge, see:

- Challenge website: [https://hc18.grand-challenge.org/](https://hc18.grand-challenge.org/)
- Challenge paper: van den Heuvel et al., "Automated measurement of fetal head circumference using 2D ultrasound images", *PLOS ONE*, 2018

## Quality Control

All annotations in this dataset were:

- Derived from expert-annotated head circumference segmentation masks
- Quality-controlled for anatomical plane correctness
- Verified against clinical measurement standards
- Cross-checked for consistency with manual measurements

## Notes

- All images are from the transventricular plane (standard plane for HC measurement)
- The `px_to_mm_rate` field is essential for converting pixel measurements to millimeters
- Some images may have slightly different resolutions due to device variations
- The dataset includes natural variability in image quality, fetal positioning, and gestational age
- Landmarks were derived via ellipse fitting to segmentation masks, not manual point annotation (see paper Discussion for implications on cross-dataset performance)

## Comparison with Other Datasets

The HC18 dataset differs from the FP and UCL datasets in:

- **Single anatomy focus**: Head only (vs. Head, Abdomen, Femur in FP and UCL)
- **Image naming**: Simpler ID-based naming (vs. Patient-Plane-based naming in FP)
- **Annotation method**: Ellipse-fitting from segmentation masks (vs. manual VIA landmark annotation in FP and UCL)
- **Acquisition sites**: 1 site with GE devices only
- **Purpose**: Originally designed for a grand challenge on HC estimation

