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
Multicentre-Fetal-Biometry/
├── data/                      # Image + annotation folders (FP, HC18, UCL, MULTICENTRE)
│   ├── FP/
│   ├── HC18/
│   ├── UCL/
│   └── MULTICENTRE/
├── experiments/fetal/         # Model configuration files (.yaml) for each dataset/anatomy
├── tools/                     # Training and testing scripts
│   ├── train.py
│   └── test.py
├── lib/                       # Model implementations and datasets
│   ├── models/                # HRNet model architecture
│   ├── datasets/              # Dataset loaders
│   ├── core/                  # Training/evaluation functions
│   └── utils/                 # Utility functions
├── hrnetv2_pretrained/        # HRNetV2 ImageNet pretrained weights
├── output/                    # Training outputs (checkpoints, logs)
├── fonts/                     # Font files for visualization
├── environment.yml            # Conda environment specification
├── requirements.txt           # Additional pip requirements
├── create_ucl_error_boxplots.py  # Error analysis scripts
├── analysis_reports.ipynb     # Jupyter notebook for analysis
└── README.md
```

- Detailed per-dataset information and annotations are in the `data/` subdirectories.

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

Download the data archives from the UCL Research Data Repository (link to be added upon publication) and extract them into the `data/` directory.

> **Note**: Dataset DOI and download instructions will be added upon publication.

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
│       │   └── FL/          # PNG images (femur)
│       ├── HC18/
│       │   └── Head/        # PNG images
│       ├── UCL/
│       │   ├── Head/        # JPEG images
│       │   ├── Abdomen/     # JPEG images
│       │   └── Femur/       # JPEG images
│       └── MULTICENTRE/
│           ├── Head/
│           ├── Abdomen/
│           └── Femur/
├── experiments/fetal/       # Configuration files for each dataset/anatomy
├── hrnetv2_pretrained/      # HRNetV2 ImageNet pretrained weights
├── tools/
│   ├── train.py             # Training script
│   └── test.py              # Testing/evaluation script
├── lib/                     # Model and dataset implementations
└── output/                  # Training outputs (checkpoints, logs)
```

---

## Training Models

To train BiometryNet (HRNet-based landmark detector) on any dataset/anatomy combination:

```bash
python tools/train.py --cfg experiments/fetal/<CONFIG-FILE>.yaml
```

### Examples

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

Training outputs (model checkpoints, logs) are saved to `output/<DATASET>_<ANATOMY>_<MEASUREMENT>/`.

---

## Testing & Evaluation

To evaluate a trained model on a test set:

```bash
python tools/test.py --cfg <CONFIG-FILE> --model-file <MODEL-WEIGHT-PATH>
```

### Examples

**Test FP-trained model on FP test set (within-domain):**
```bash
python tools/test.py --cfg experiments/fetal/fetal_landmark_hrnet_w18_FP_brain_BPD.yaml \
                     --model-file output/FP_brain_BPD/model_best.pth
```

**Test FP-trained model on UCL test set (cross-domain):**
```bash
python tools/test.py --cfg experiments/fetal/fetal_landmark_hrnet_w18_UCL_brain_BPD.yaml \
                     --model-file output/FP_brain_BPD/model_best.pth
```

This reproduces the cross-domain evaluation results (e.g., FP→UCL) shown in the benchmark tables above.

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
                     --model-file output/FP_brain_BPD/model_best.pth

# Test on UCL test set (cross-domain)
python tools/test.py --cfg experiments/fetal/fetal_landmark_hrnet_w18_UCL_brain_BPD.yaml \
                     --model-file output/FP_brain_BPD/model_best.pth
```

---

## Variability Analysis

Preprocessing scripts for image standardization and anatomical variability analysis are provided:

- **Image preprocessing:** Crop overlays, resize to 1024×1024, normalization
- **Variability analysis:** `create_ucl_error_boxplots.py` for generating error distribution plots
- **Augmentation:** Standard augmentation techniques applied during training

See individual scripts for detailed usage.

---

## Citation

If this dataset or code is used in your research, please cite:

```bibtex
@article{divece2024fetal,
  title={A benchmark multi-centre multi-device dataset for landmark-based comprehensive fetal biometry},
  author={Di Vece, Chiara and Mao, Zhehua and Avisdris, Netanell and Dromey, Brian and Napolitano, Raffaele and Vasconcelos, Francisco and Stoyanov, Danail and Joskowicz, Leo and Bano, Sophia},
  journal={Scientific Reports},
  year={2024},
  note={Dataset available at [TO BE ADDEDD]}
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

- **Data:** Released under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://github.com/surgical-vision/Fetal-Biometry-Dataset/blob/main/LICENSE-DATA) license.  
  You may share and adapt the dataset, provided that appropriate credit is given.  

---

## Contact

**Corresponding author:**  
Chiara Di Vece  
Department of Computer Science and UCL Hawkes Institute  
University College London  
📧 chiara.divece.20@ucl.ac.uk  
