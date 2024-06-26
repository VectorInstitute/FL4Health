# Skin Cancer Dataset Download and Preprocessing

This repository provides a set of scripts to download and preprocess skin cancer datasets for use in federated learning experiments. The datasets included are HAM10000, PAD-UFES-20, ISIC_2019, and Derm7pt.

## Getting Started

To start using these datasets, follow the steps below.

### Prerequisites

Ensure you have the following prerequisites installed:
- Python 3.x
- Necessary Python packages (install using `requirements.txt` if provided)

### Downloading the Datasets

First, manually download the HAM10000 and Derm7pt datasets and place them in the `fl4health/datasets/skin_cancer` directory:

1. **HAM10000**: Download from [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
2. **Derm7pt**: Download from [SFU](https://derm.cs.sfu.ca/Welcome.html)

### Running the Download Script

After placing the HAM10000 and Derm7pt datasets in the `fl4health/datasets/skin_cancer` directory, run the provided shell script to organize the datasets and automatically download the ISIC_2019 and PAD-UFES-20 datasets.

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
python fl4health/datasets/skin_cancer/preprocess_skin.py
```

### Using the Datasets

After preprocessing, the datasets are ready to be used in the federated learning examples provided in this repository. For an example, refer to the FedBN example:

[examples/fedbn_example/README.md](examples/fedbn_example/README.md)