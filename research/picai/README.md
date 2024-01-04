# Overview

The [PI-CAI](https://pi-cai.grand-challenge.org/) (Prostate Imaging: Cancer AI) is a collection of 10,000 prostate MRI exams to train and vaildate AI algorithms for detection of Clinically Significant Prostate Cancer Detection (csPCa).

## Dataset

The dataset is partitioned into multiple splits corresponding to the different phases of the PICAI competition. The Public Training and Devlopment Dataset is available on the cluster at the following path:
```
/ssd003/projects/aieng/public/PICAI
```

For each patient exam in this dataset, the following information is available:
- Clinicial Variables: patient age, prostate volume, PSA level, PSA density (as reported in their diagnostic reports)
- Acquisition Variables: scanner manufacturer, scanner model name, diffusion b-value, and
- bpMRI Scans: acquired using Siemens Healthineers or Philips Medical Systems-based scanners with surface coils.

Imaging consists of the following sequences:
- Axial, sagittal and coronal T2-weighted imaging (T2W).
- Axial high b-value (≥ 1000 s/mm²) diffusion-weighted imaging (DWI).
- Axial apparent diffusion coefficient maps (ADC).
