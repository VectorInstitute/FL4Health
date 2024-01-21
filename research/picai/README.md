# Overview

The [PI-CAI](https://pi-cai.grand-challenge.org/) (Prostate Imaging: Cancer AI) is a collection of 10,000 prostate MRI exams to train and vaildate AI algorithms for detection of Clinically Significant Prostate Cancer Detection (csPCa).

## Raw Dataset

The dataset is partitioned into multiple splits corresponding to the different phases of the PICAI competition. The Public Training and Devlopment Dataset, and preprocessed variants, are available on the cluster at the following path:
```
/ssd003/projects/aieng/public/PICAI
```

For each patient exam in this dataset, the following information is available:
- Clinicial Variables: patient age, prostate volume, PSA level, PSA density (as reported in their diagnostic reports)
- Acquisition Variables: scanner manufacturer, scanner model name, diffusion b-value, and
- bpMRI Scans: acquired using Siemens Healthineers or Philips Medical Systems-based scanners with surface coils.
- Annotations: Human or AI Derived Annotations of csPCa lesions (if any) in MR Sequences

### Imaging

Imaging consists of the following sequences:
- Axial, sagittal and coronal T2-weighted imaging (T2W).
- Axial high b-value (≥ 1000 s/mm²) diffusion-weighted imaging (DWI).
- Axial apparent diffusion coefficient maps (ADC).

Every patient case will have the aforementioned T2W, DWI and ADC sequences stored in a file ending with `_t2w.mha`, `_hbv.mha` and `_adc.mha`, respectively. Additionally, they can also have either, both or none of these optional imaging sequences: sagittal and coronal T2W scans (i.e. files ending in `_sag.mha` and `_cor.mha`. 

The raw imaging data for each patient exam is stored on the cluster at the following path: 
```
/ssd003/projects/aieng/public/PICAI/input/images/<patient_id>
```

where **<patient_id>** is a unique identifier for a given patient. 

**Note:** Each patient can potentially be a part of multiple studies (exams at different time points) so a **<patient_id>** does not uniquely identify a specific patients exam. To account for this imaging files in each patient folder use the following naming convention: `<patient_id>_<study_id>_<sequence_id>.mha`. **<study_id>** uniquely identifies a particular study and **sequence_id>** specifies the sequence type (ie `_t2w`, `_hbv` or `_adc`). Together the **<patient_id**> and the **<study_id>** uniquely identify an exam. For example, `patient_id=10417` has two exams with `study_id=1000424` and `study_id=1000425`. Hence, this patient has two different T2W sequences (as well as two different ADC and DWI sequences) availabe at: 

```
/ssd003/projects/aieng/public/PICAI/input/images/10417/10417_1000424_t2w.mha
```
and 
```
/ssd003/projects/aieng/public/PICAI/input/images/10417/10417_1000425_t2w.mha
```

### Annotations 
Out of the 1500 cases shared in the Public Training and Development Dataset, 1075 cases have benign tissue or indolent PCa (i.e. their labels should be empty or full of 0s) and 425 cases have csPCa (i.e. their labels should have lesion blobs of value 2, 3, 4 or 5). Out of these 425 positive cases, only 220 cases carry an annotation derived by a human expert. The remaining 205 positive cases have not been annotated. In other words, only 17% (220/1295) of the annotations provided in should have csPCa lesion annotations, while the remaining 83% (1075/1295) of annotations should be empty. AI-derived annotation have been made available for practitioners that want to work with fully labeled dataset. Alternatively, practitioners can opt to leverage the samples without csPCa annotation in alternative ways if they wish to do so.

The human expert annotations are available on the cluster at: 
```
/ssd003/projects/aieng/public/PICAI/input/picai_labels/csPCa_lesion_delineations/human_expert/original/<patient_id>_<exam_id>.nii
```

Alternatively, the AI-derived annotations are available on the cluster at:
```
/ssd003/projects/aieng/public/PICAI/input/picai_labels/csPCa_lesion_delineations/AI/Bosma22a/<patient_id>_<exam_id>.nii
```

The raw annotations differ in spatial resolution across annotators. Specifically, some annotations have been created at the spatial resolution and orientation of the T2W image, while others have been created at the resolution and orientation of the DWI/ADC images. To account for this, resampled annotations have also been made available. The resampled annotations are at the same dimension and spatial resolution astheir correspinding T2W images, following the guidelines of the PICAI competition. The resampled annotations are available at: 
```
/ssd003/projects/aieng/public/PICAI/input/picai_labels/csPCa_lesion_delineations/human_expert/resampled/<patient_id>_<exam_id>.nii
```

For more information on the raw dataset (imaging or labels) or splits other than the Public Training and Development Dataset, refer to the [PICAI competition documentation](https://pi-cai.grand-challenge.org/DATA/). 

## Preprocessed Dataset


