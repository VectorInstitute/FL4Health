#  Overview

The [PI-CAI](https://pi-cai.grand-challenge.org/) (Prostate Imaging: Cancer AI) is a collection of MRI exams to train and validate AI algorithms for detection of Clinically Significant Prostate Cancer Detection (csPCa). This folder is focused on providing examples and utilities for performing experiments on the PICAI dataset for csPCa using both centralized and a federated setup. The federated learning examples heavily leverage the [fl4health package](/README.md) to conveniently apply state-of-the-art FL techniques to real world datasets. To this end, there is currently two examples:
- [U-Net on PICAI with Centralized Setup](/research/picai/central)
- [U-Net on PICAI with Federated Setup with FedAvg](/research/picai/fedavg)

## Development Requirements

For development and testing, we use [Poetry](https://python-poetry.org/) for dependency management. The library dependencies and those for development and testing are listed in the `pyproject.toml` file. You may use whatever virtual environment management tool that you would like. These include conda, poetry itself, and virtualenv. Poetry is also used to produce our releases, which are managed and automated by GitHub.

The easiest way to create and activate a virtual environment is by using the [virtualenv](https://pypi.org/project/virtualenv/) package:
```bash
virtualenv "ENV_PATH"
source "ENV_PATH/bin/activate"
pip install --upgrade pip poetry
poetry install --with "picai"
```

This will setup an environment with the proper dependencies to run the provided scripts out of the box. For more information about environment configuration, please refer to the [main documentation](/README.md).

## Data
### Raw Dataset

The dataset is partitioned into multiple splits corresponding to the different phases of the PICAI competition. The Public Training and Development Dataset, and preprocessed variants, are available on the cluster at the following path:
```
/ssd003/projects/aieng/public/PICAI
```

For each patient exam in this dataset, the following information is available:
- **Clinical Variables:** patient age, prostate volume, PSA level, PSA density (as reported in their diagnostic reports)
- **Acquisition Variables:** scanner manufacturer, scanner model name, diffusion b-value, and
- **bpMRI Scans:** acquired using Siemens Healthineers or Philips Medical Systems-based scanners with surface coils.
- **Annotations:** Human or AI Derived Annotations of csPCa lesions (if any) in MR Sequences

#### Imaging

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

**Note:** Each patient can potentially be a part of multiple studies (exams at different time points) so a **<patient_id>** does not uniquely identify a specific patients exam. To account for this imaging files in each patient folder use the following naming convention:
```
<patient_id>_<study_id>_<sequence_id>.mha
```

**<study_id>** uniquely identifies a particular study and **<sequence_id>** specifies the sequence type (ie `_t2w`, `_hbv` or `_adc`). Together the **<patient_id**> and the **<study_id>** uniquely identify an exam. For example, `patient_id=10417` has two exams with `study_id=1000424` and `study_id=1000425`. Hence, this patient has two different T2W sequences (as well as two different ADC and DWI sequences) available at:

```
/ssd003/projects/aieng/public/PICAI/input/images/10417/10417_1000424_t2w.mha
```
and
```
/ssd003/projects/aieng/public/PICAI/input/images/10417/10417_1000425_t2w.mha
```

#### Annotations
Out of the 1500 cases shared in the Public Training and Development Dataset, 1075 cases have benign tissue or indolent PCa and 425 cases have csPCa. Out of these 425 positive cases, only 220 cases carry an annotation derived by a human expert. The remaining 205 positive cases have not been annotated. AI-derived annotation have been made available for practitioners that want to work with fully labeled dataset. Alternatively, practitioners can opt to leverage the samples without csPCa annotation in alternative ways if they wish to do so.

The human expert annotations are available on the cluster at:
```
/ssd003/projects/aieng/public/PICAI/input/picai_labels/csPCa_lesion_delineations/human_expert/original/<patient_id>_<study_id>.nii
```

Alternatively, the AI-derived annotations are available on the cluster at:
```
/ssd003/projects/aieng/public/PICAI/input/picai_labels/csPCa_lesion_delineations/AI/Bosma22a/<patient_id>_<study_id>.nii
```

The raw annotations differ in spatial resolution across annotators. Specifically, some annotations have been created at the spatial resolution and orientation of the T2W image, while others have been created at the resolution and orientation of the DWI/ADC images. To account for this, resampled annotations have also been made available. The resampled annotations are at the same dimension and spatial resolution as their corresponding T2W images, following the guidelines of the PICAI competition. The resampled annotations are available at:
```
/ssd003/projects/aieng/public/PICAI/input/picai_labels/csPCa_lesion_delineations/human_expert/resampled/<patient_id>_<study_id>.nii
```

For more information on the raw dataset (imaging or labels) or splits other than the Public Training and Development Dataset, refer to the [PICAI competition documentation](https://pi-cai.grand-challenge.org/DATA/).

### Preprocessed Dataset
In addition to Raw Dataset, a Preprocessed Dataset has been provided on the cluster that was generated using scripts described in the following section. The preprocessed dataset has been prepared in the format the [nnUNet framework](https://github.com/MIC-DKFZ/nnUNet) expects, making nnUNet easy to apply out of the box to the dataset. For more information about nnUNet as it relates to our experimentation on the PICAI dataset, refer to [nnUNet_overview.md](../picai/nnunet_scripts/nnunet_overview.md). Additional preprocessing utilities and training baselines are available in the [picai_prep package](https://github.com/DIAGNijmegen/picai_prep) and [picai_baseline package](https://github.com/DIAGNijmegen/picai_baseline). These packages were provided by the PICAI competition to give a starting point for practitioners to preprocess the data from the Public Training and Development Dataset and train a simple model for the csPCa task.

### Preprocessing
The Preprocessed Dataset was generated by running the `prepare_annotations.py` and `prepare_data.py` scripts sequentially. These scripts assume that the dataset is formatted identically to the raw dataset described above.

`prepare_annotations.py` is a simple script that copies the human and ai-derived annotations into a specified folder. An example invocation is as follows:

```
python research/picai/preprare_annoations.py --human_annotations_dir /path/to/human/annotations --ai_annotations_dir /path/to/ai/annotations --annotations_write_dir /path/to/write/dir
```

`prepare_data.py` is the main preprocessing script that takes in a number of arguments related to the location of the raw dataset and details about the preprocessing and produces a preprocessed dataset with an associated dataset overview file. Here is an example invocation:
```
python research/picai/prepare_data.py --scans_read_dir /path --annotations_read_dir /path --scans_write_dir /path --overview_write_dir /path --size 20 256 256  --spacing 3 0.5 0.5 --num_threads 4 --splits_path /path
```
The arguments are defined accordingly:
- **scans_read_dir**: Base directory containing a number of subdirectories - each with one or more scans.
- **annotations_read_dir**: Base directory containing a number of annotations files.
- **scans_write_dir**: Base directory to write all preprocessed scans.
- **annotations_write_dir**: Base directory to write all preprocessed annotations.
- **overview_write_dir**: Base directory to write all dataset overviews (more details below).
- **size**: Desired dimensions of preprocessed scans in voxels/pixels. Triplet of the form: Depth x Height x Weight.
- **spacing**: Desired voxel spacing of preprocessed scans in mm/voxel. Triplet of the form: Depth x Height x Weight.
- **num_threads**: The number of threads to use during data preprocessing. Defaults to 4.
- **splits_path**: Optional path to the desired splits. Defaults to official picai splits.

The specific artifacts of the `prepare_data.py` script are the preprocessed scans and annotations saved to the specified write directories. Dataset overview(s) are also generated. These are based on the splits json file that is passed to the script. The dataset overview files include a list of sets of scan paths along with corresponding annotations for a specific fold.

The default settings of the `prepare_data.py` script perform the following operations:
- **Resampling Spatial Resolution:** The spatial resolution of its images vary across different patient exams. For the axial T2W scans, the most common voxel spacing (in mm/voxel) observed is 3.0×0.5×0.5 (43%), followed by 3.6×0.3×0.3 (25%), 3.0×0.342×0.342 (15%) and others (17%). As a naive approach, we simply resample all scans to 3.0×0.5×0.5 mm/voxel.

- **Cropping to Region-of-Interest:** We naively assume that the prostate gland is typically located within the centre of every prostate MRI scan. Hence, we take a centre crop of each scan, measuring 20×256×256 voxels in dimensions.

Each preprocessed image sample is stored on the cluster at the following path:
```
/ssd003/projects/aieng/public/PICAI/nnUNet/nnUNet_raw/Dataset003_PICAI/imagesTr/<patient_id>_<study_id>_<modality_id>.nii.gz
```

Thus, all preprocessed images are written to the same directory. Each scan is uniquely identified with a **<patient_id>** and **<study_id>** described above, along with a **<modality_id>**. The **<modality_id>** specifies what type the scan is. In our case, the **<modality_id>** has the following mapping:
- 0000: T2W Scan
- 0001: ADC Scan
- 0002: HBV Scan

Each preprocessed scan has a corresponding label available at:
```
/ssd003/projects/aieng/public/PICAI/nnUNet/nnUNet_raw/Dataset003_PICAI/labelsTr/<patient_id>_<study_id>.nii.gz
```

### Cross Validation Splits
The PICAI competition has prepared pre-determined 5-fold cross-validation splits of all 1500 cases. The splits do not contain patient overlap between train and validation splits. A json file, corresponding to the **splits_path** argument above, was released by the PICAI challenge which specifies the samples involved in each of the splits. This file has been made available on the cluster at

```
/ssd003/projects/aieng/public/PICAI/input/splits.json
```

If **splits_path** is passed to the `prepare_data.py` script, it will generate two files for each split. One is a json file containing pairs of scan paths and their annotation path for the train portion of the fold. On the cluster these files are available at:
```
/ssd003/projects/aieng/public/PICAI/nnUNet/nnUNet_raw/Dataset003_PICAI/overviews/train-fold-<fold_id>.json
```

Similarly, the validation dataset file for a specific fold is available at:
```
/ssd003/projects/aieng/public/PICAI/nnUNet/nnUNet_raw/Dataset003_PICAI/overviews/val-fold-<fold_id>.json
```

To learn about the details of the preprocessing, please refer to the [General Setup Documentation](https://github.com/DIAGNijmegen/picai_baseline?tab=readme-ov-file#general-setup) and subsequently the [U-Net Baseline Data Preparation](https://github.com/DIAGNijmegen/picai_baseline/blob/main/unet_baseline.md#u-net---data-preparation).

### Prostate Whole Gland Segmentation
Another related segmentation task using the aforementioned dataset is Prostate Whole Gland (WG) Segmentation. As the name implies, WG Segmentation involves segmenting the entire prostate, consisting of the peripheral zone (PZ) and the transition zone (TZ) ([more on this](https://radiopaedia.org/articles/prostate)). WG Segmentation is an important task in itself but is often used as a means to localize the prostate to register between sequences or narrow the region of interest for downstream segmentation tasks. The PICAI grand challenge has provided [AI-derived segmentation maps of the WG](https://github.com/DIAGNijmegen/picai_labels) for all of the samples in the dataset. To this end, a dataset with the identical preprocessing as above (except for the fact it only includes T2 sequence) is made available on the cluster.

Each preprocessed image sample is stored on the cluster at the following path:
```
/ssd003/projects/aieng/public/PICAI/nnUNet/nnUNet_raw/Dataset001_PICAI_WG/imagesTr/
```

and the corresponding label
```
/ssd003/projects/aieng/public/PICAI/nnUNet/nnUNet_raw/Dataset001_PICAI_WG/labelsTr/
```

The images and files follow the same format and naming conventions as described above.
