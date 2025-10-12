# Fetal Biometry Dataset  
**A benchmark multi-centre multi-device dataset for landmark-based comprehensive fetal biometry**

---

## Overview  

Accurate fetal growth assessment from ultrasound depends on the precise measurement of biometric parameters, obtained by identifying anatomical landmarks in standard fetal planes. Manual measurement is time-consuming, operator-dependent, and sensitive to variability across scanners and acquisition sites, limiting reproducibility in both clinical and research settings.  

This repository accompanies the publication *“A benchmark multi-centre multi-device dataset for landmark-based comprehensive fetal biometry”*. It provides an open-access dataset and reference code for automated fetal biometry research.  

The dataset combines ultrasound images from **three independent sources** acquired on **seven different ultrasound devices**, all of which are annotated by experts for standard fetal biometric landmarks. It supports training and evaluation of algorithms for fetal biometry estimation, growth assessment, and cross-domain generalization.  

---

## Dataset Summary  

| Dataset | Source / Institution | Subjects | Images | Anatomical Planes | Devices | Annotation Tool |
|----------|---------------------|-----------:|---------:|-------------------|----------|----------------|
| **Fetal Planes (FP)** | Vall d’Hebron & Sant Joan de Déu, Barcelona, Spain | 1,047 | 3,091 | Head (1,638), Abdomen (694), Femur (760) | GE Voluson E6/S8/S10, Aloka | VIA |
| **HC18** | Radboud University Medical Center, Netherlands | 806 | 999 | Head | GE Voluson E8, 730 | Ellipse fitting |
| **UCLH** | University College London Hospital, UK | 42 | 346 | Head (135), Abdomen (103), Femur (108) | GE Voluson | VIA |

Each dataset includes 2D ultrasound standard planes and corresponding landmark annotations for:  
- **Head:** biparietal diameter (BPD), occipito-frontal diameter (OFD)  
- **Abdomen:** transverse abdominal (TAD) and anterior-posterior abdominal (APAD) diameters  
- **Femur:** femur length (FL)

All images are de-identified and standardized to a resolution of 1024 × 1024 pixels.  

---

## Repository Structure  

```plaintext
Fetal-Biometry-Dataset/
├── data/                      # Image datasets (FP, HC18, UCLH)
├── annotations/               # VIA JSON annotation files
├── scripts/                   # Preprocessing, evaluation, and visualization
├── splits/                    # Train / val / test CSVs
└── README.md
```

---

## Data Format

**Images:** `.png`  
**Annotations:** VIA-compatible `.json` files containing landmark coordinates and measurement types.  

---

## Benchmark Evaluation  

The dataset was benchmarked using **BiometryNet** (Avisdris *et al.*, MICCAI 2022), an HRNet-based landmark regression framework with Dynamic Orientation Determination (DOD).  

| Training Set | Test Set | Anatomy | NME ± STD |
|---------------|-----------|----------|------------|
| FP | FP | Head | 0.46 ± 0.48 |
| FP | UCLH | Head | 0.68 ± 0.51 |
| FP | HC18 | Head | 0.75 ± 0.12 |
| UCLH | UCLH | Abdomen | 0.32 ± 0.43 |
| UCLH | FP | Femur | 0.99 ± 0.09 |

Cross-dataset experiments demonstrate the effects of multi-centre and multi-device variability and provide a quantitative reference for future methods.  

---

## Usage  

### Clone the repository  

```bash
git clone https://github.com/surgical-vision/Fetal-Biometry-Dataset
cd Fetal-Biometry-Dataset
```

## Obtain the datasets

Download the data archives from the UCL Research Data Repository (link available upon publication) and extract them into the data/ directory.

TO DO: add link to UCL dataset

### Reproduce variability analysis

TBD

### Run BiometryNet baseline

TBD

---

## Citation

If this dataset or code is used in your research, please cite:

> **Di Vece, C., Mao, Z., Avisdris, N., Dromey, B., Vasconcelos, F., Stoyanov, D., Joskowicz, L., Bano, S.**  
> *A benchmark multi-centre multi-device dataset for landmark-based comprehensive fetal biometry.*  
> *JOURNAL*, 2025.

---

## License

This dataset and accompanying code are distributed under the *Creative Commons Attribution 4.0 International (CC BY 4.0)* license.
Users may share and adapt the material provided that appropriate credit is given.

---

## Contact

**Corresponding author:**  
Chiara Di Vece  
Department of Computer Science and UCL Hawkes Institute  
University College London  
📧 chiara.divece.20@ucl.ac.uk  
