# Skin Cancer Dataset Download and Preprocessing

This repository provides a set of scripts to download and preprocess skin cancer datasets for use in federated learning experiments. The datasets included are HAM10000, PAD-UFES-20, ISIC_2019, and Derm7pt.

The code is adapted from the paper by Seongjun Yang*<sup>1</sup>, Hyeonji Hwang*<sup>2</sup>, Daeyoung Kim<sup>2</sup>, Radhika Dua<sup>3</sup>, Jong-Yeup Kim<sup>4</sup>, Eunho Yang<sup>2</sup>, Edward Choi<sup>2</sup> | [Paper](https://arxiv.org/abs/2207.03075)

<sup>1</sup>KRAFTON, <sup>2</sup>KAIST AI, <sup>3</sup>Google Research, India, <sup>4</sup>College of Medicine, Konyang University

and their code available at the [medical_federated GitHub repository](https://github.com/wns823/medical_federated.git).

## Getting Started

To start using these datasets, follow the steps below.

### Prerequisites
In order to run the scripts, first ensure you have [installed the dependencies in your virtual environment according to the main README](/README.md#development-requirements) and it has been activated.

### Downloading the Datasets
To use the datasets for this project, follow the instructions below to download and unzip the required files.

## 1. Derm7pt

- **Download Link**: [SFU Derm7pt](https://derm.cs.sfu.ca/Welcome.html)
- **Instructions**:
  1. Download the `release_v0.zip` file from the link above.
  2. Place the downloaded files under `fl4health/datasets/skin_cancer`.

## 2. HAM10000

- **Download Link**: [Harvard Dataverse HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
- **Instructions**:
  1. Download the `HAM10000_images_part_1.zip`, `HAM10000_images_part_2.zip`, and `HAM10000_metadata.tab` files from the link above.
  2. Place the downloaded files under `fl4health/datasets/skin_cancer`.

### Running the Download Script

After placing the HAM10000 and Derm7pt dataset files in the `fl4health/datasets/skin_cancer` directory, run the provided shell script to organize the datasets and automatically download the ISIC_2019 and PAD-UFES-20 datasets and unzip the HAM10000 and Derm7pt datasets.

```sh
sh fl4health/datasets/skin_cancer/download.sh
```

### Directory Structure

After running the script, the following directory structure will be created under `fl4health/datasets/skin_cancer/`:

```
/datasets/skin_cancer/HAM10000/
/datasets/skin_cancer/PAD-UFES-20/
/datasets/skin_cancer/ISIC_2019/
/datasets/skin_cancer/Derm7pt/
```

### Preprocessing the Datasets

The download script will automatically call the preprocessing script to prepare the datasets for use.

```sh
python -m fl4health.datasets.skin_cancer.preprocess_skin
```

### Using the Datasets

After preprocessing, the datasets are ready to be used in the federated learning examples provided in this repository. For an example, refer to the FedBN example:

[FedBN Example README](../../examples/fedbn_example/README.md)
