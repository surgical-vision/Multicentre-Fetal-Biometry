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
|--------|----------------------|---------:|-------:|-------------------|---------|-----------------|
| **Fetal Planes (FP)** | Vall d’Hebron & Sant Joan de Déu, Barcelona, Spain | 1,047 | 3,091 | Head (1,638), Abdomen (693), Femur (760) | GE Voluson E6/S8/S10, Aloka | VIA (manual landmarks) |
| **HC18** | Radboud University Medical Center, Netherlands | 806 | 999 | Head only | GE Voluson E8, 730 | Ellipse fitting from HC masks |
| **UCL** | University College London Hospital (UCLH), UK | 51 | 427 | Head (161), Abdomen (131), Femur (135) | GE Voluson | VIA (manual landmarks) |

Each dataset includes 2D ultrasound standard planes and corresponding landmark annotations for:  
- **Head:** biparietal diameter (BPD), occipito-frontal diameter (OFD)  
- **Abdomen:** transverse abdominal (TAD) and anterior–posterior abdominal (APAD) diameters  
- **Femur:** femur length (FL)

All images are de-identified. Preprocessing scripts in this repository convert images to a standard 1024 × 1024 resolution for training and evaluation.

---

## Repository Structure  

```plaintext
Fetal-Biometry-Dataset/ 
├── fetalbiometrydata/         # Image + annotation folders (FP, HC18, UCL, MULTICENTRE) 
│   ├── FP/ 
│   ├── HC18/ 
│   ├── UCL/ 
│   └── MULTICENTRE/           # Derived multi-centre subset (overlapping view) 
├── scripts/                   # Preprocessing, training, evaluation, visualization 
├── configs/                   # Model configuration files (e.g. HRNet, BiometryNet) 
├── env/                       # Environment/setup scripts 
└── README.md
```

- Detailed per-dataset READMEs are in `fetalbiometrydata/FP/`, `HC18/`, `UCL/`, and `MULTICENTRE/`.

---

## Data Format

- **Images:** JPEG or PNG (depending on source dataset)  
- **Annotations:** CSV files with landmark coordinates and metadata (plus optional VIA JSON in some cases)  

See the dataset-specific READMEs for full column descriptions per anatomy.

---

## Benchmark Evaluation  

The dataset was benchmarked using **BiometryNet** (Avisdris *et al.*, MICCAI 2022), an HRNet-based landmark regression framework with Dynamic Orientation Determination (DOD). Results below are **Normalised Mean Error (NME) ± standard deviation**, consistent with the paper.

### Brain (Head) Biometry – BPD and OFD

| Train set | Test set | NME (BPD) ± STD | NME (OFD) ± STD |
|-----------|----------|-----------------|-----------------|
| FP        | FP       | 0.4600 ± 0.4835 | 0.4552 ± 0.4817 |
| FP        | HC18     | 0.5637 ± 0.3572 | 0.5743 ± 0.3257 |
| FP        | UCL      | 0.4583 ± 0.3910 | 0.4780 ± 0.3177 |
| HC18      | FP       | 0.5731 ± 0.4002 | 0.5021 ± 0.4737 |
| HC18      | UCL      | 0.7985 ± 0.4653 | 0.5610 ± 0.4265 |
| UCL       | UCL      | 0.2439 ± 0.3749 | 0.2230 ± 0.3728 |
| Ours (FP+HC18+UCL) | UCL | **0.2110 ± 0.3822** | **0.1927 ± 0.3700** |

### Abdomen – APAD and TAD

| Train set | Test set | NME (APAD) ± STD | NME (TAD) ± STD |
|-----------|----------|------------------|-----------------|
| FP        | FP       | 0.5987 ± 0.4527  | 0.6091 ± 0.4482 |
| FP        | UCL      | 0.4041 ± 0.3800  | 0.5164 ± 0.3663 |
| UCL       | UCL      | **0.3149 ± 0.4246** | **0.3518 ± 0.4209** |

### Femur – FL

| Train set | Test set | NME (FL) ± STD   |
|-----------|----------|------------------|
| FP        | FP       | 0.0338 ± 0.1168  |
| UCL       | UCL      | **0.0259 ± 0.0410** |
| FP        | UCL      | 0.8371 ± 0.4023  |
| UCL       | FP       | 0.9839 ± 0.1007  |
| Ours      | UCL      | 0.9994 ± 0.0060  |

These results reproduce Table 2 in the paper and highlight:

- Strong within-dataset performance (e.g., FP→FP, UCL→UCL)  
- Significant performance degradation under cross-dataset evaluation (domain shift)  
- Improved generalisation for head biometry when training on multi-centre data (“Ours”) and testing on UCL.

---

## Usage  

### Clone the repository  

```bash
git clone https://github.com/surgical-vision/Fetal-Biometry-Dataset
cd Fetal-Biometry-Dataset
```

## Obtain the datasets

Download the data archives from the UCL Research Data Repository (link to be added upon publication) and extract them into the `fetalbiometrydata/` directory.

> **TODO**: Add final DOI / URL for the UCL/open-access data repository.

### Reproduce variability analysis

Preprocessing scripts (e.g. cropping overlays, resizing to 1024×1024, normalisation, augmentation) and anatomical variability analysis are provided in `scripts/`. Usage examples will be added once the public release is finalised.

### Run BiometryNet baseline

Configuration files and scripts to run BiometryNet on FP, HC18, UCL, and multi-centre training are provided (e.g., HRNet configs in `configs/`, training scripts in `scripts/`).

> **TODO**: Add exact command-line examples for reproducing the baseline results in the paper.

---

## Citation

If this dataset or code is used in your research, please cite:

> **TODO**: Add reference

---

## License

- **Code:** Released under the [MIT LICENSE](https://github.com/surgical-vision/Fetal-Biometry-Dataset/blob/main/LICENSE).
  Permission is granted to use, copy, modify, and distribute the software for any purpose with attribution.  

- **Data:** Released under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://github.com/surgical-vision/Fetal-Biometry-Dataset/blob/main/LICENSE-DATA) license.  
  You may share and adapt the dataset, provided that appropriate credit is given.  

---

## Contact

**Corresponding author:**  
Chiara Di Vece  
Department of Computer Science and UCL Hawkes Institute  
University College London  
📧 chiara.divece.20@ucl.ac.uk  
