# Tabular Data Feature Alignment Example
This example demonstrates the feature alignment capability of the tabular data preprocessing pipeline. In a federated setting, there can be "misalignment" of features between different clients, making it impossible to train a global model. Here, "misalignment" refers to the fact that different tabular datasets can have different columns, and for columns that correspond to categorical features, the categories could be different, resulting in one-hot encodings of different lengths.

## Running the Example
In order to run the example, first ensure you have [installed the dependencies in your virtual environment according to the main README](/README.md#development-requirements) and it has been activated.

## Data Information
The dataset used here is the MIMIC3d aggregated data collected here:

https://www.kaggle.com/datasets/drscarlat/mimic3d/data

Run `misalign_data.py` to manually create two subsets `mimic3d_hospital1.csv` and `mimic3d_hospital2.csv` that are misaligned.

More specifically, the original `mimic3d.csv` is randomly split into two datasets which represent two hospitals, and the columns "ExpiredHospital", "admit_type", "NumRx", "ethnicity" are removed from the data of hospital2. In addition, a new category "Unknown" is added to the "insurance" column for hospital2, which is a categorical column.

In this example, we assume that the target column is the categorical column `LOSgroupNum`, which groups the length of stay (days) into four groups: 0-2 days, 2-4 days etc.


## Starting Server

Start the server by running something like
```bash
python -m examples.feature_alignment_example.server --config_path /path/to/config.yaml
```
from the FL4Health directory. The following arguments must be present in the specified config file:
* `n_clients`: number of clients the server waits for in order to run the FL training.
* `local_epochs`: number of epochs each client will train for locally.
* `batch_size`: size of the batches each client will train on.
* `source_specified`: whether the server is provided with the information necessary to perform alignment or not. If this field is `True`, then it means the server already has the information needed for alignment. So the server will send this information to all the clients and wait for them to perform feature alignment. If this field is `False`, then the server will randomly select one of the clients and asks it to provide information about its features, then send this information to all clients so they can use it to perform feature alignment.

## Starting Clients

Once the server has started and logged "FL starting," the next step, in separate terminals, is to start the `n_clients`
clients expected by the server. This is done by simply running (remembering to activate your environment)
```bash
python -m examples.feature_alignment_example.client --dataset_path <path_to_dataset>
```
* `path_to_dataset` is the path towards the directory where the dataset is stored. To see the feature alignment capability in this example, you should launch clients with misaligned datasets.


## Running the example

For this example, first start the server.

Then run

```bash
python -m examples.feature_alignment_example.client --dataset_path /path/mimic3d_hospital1.csv
```

and then run

```bash
python -m examples.feature_alignment_example.client --dataset_path /path/mimic3d_hospital2.csv
```
