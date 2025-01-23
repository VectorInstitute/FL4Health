# Fluorescent Microscopy Images Dataset Download and Preprocessing

This repository provides a set of scripts to download and preprocess RxRx1 datasets for use in federated learning experiments. This dataset include 6-channel fluorescent microscopy images of cells treated with different compounds. The dataset is provided by Recursion Pharmaceuticals and is available on the [RxRx1 Kaggle page](https://www.rxrx.ai/rxrx1).

## Getting Started

To start using these datasets, follow the steps below.


### Downloading the Datasets
To use the datasets for this project, run the provided shell script to download and unzip the required files.

```sh
sh fl4health/datasets/rxrx1/download.sh
```


### Preprocessing the Datasets

Once the datasets are downloaded, preprocess them to generate the required metadata file and prepare the training and testing tensors for each client participating in the federated learning experiments. The following command preprocesses the RxRx1 datasets:

```sh
python fl4health/datasets/rxrx1/preprocess.py --data_dir <path_to_rxrx1_data>
```

### Using the Datasets

After preprocessing, the datasets are ready to be used in the federated learning settings. For examples please refer to the [Rxrx1 experiments](research/rxrx1) directory.
