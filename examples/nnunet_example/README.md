# NnUNetClient Example

This example demonstrates how to use the NnUNetClient to train nnunet segmentation models in a federated setting.

By default this example trains an nnunet model on the Task04_Hippocampus dataset from the Medical Segmentation Decathlon (MSD). However, any of the MSD datasets can be used by specifying them with the msd_dataset_name flag for the client. To run this example first create a config file for the server. An example config has been provided in this directory. The required keys for the config are:

```yaml
# Parameters that describe the server
n_server_rounds: 1

# Parameters that describe the clients
n_clients: 1
local_epochs: 1 # Or local_steps, one or the other must be chosen

nnunet_config: 2d
```

The only additional parameter required by nnunet is nnunet_config which is one of the official nnunet configurations (2d, 3d_fullres, 3d_lowres, 3d_cascade_fullres)

One may also add the following optional keys to the config yaml file

```yaml
# Optional config parameters
nnunet_plans: /Path/to/nnunet/plans.json
starting_checkpoint: /Path/to/starting/checkpoint.pth
```

To run a federated learning experiment with nnunet models, first ensure you are in the FL4Health directory and then start the nnunet server using the following command. To view a list of optional flags use the --help flag

```bash
python -m examples.nnunet_example.server --config_path examples/nnunet_example/config.yaml
```

Once the server has started, start the necessary number of clients specified by the n_clients key in the config file. Each client can be started by running the following command in a seperate session. To view a list of optional flags use the --help flag.

```bash
python -m examples.nnunet_example.client --dataset_path examples/datasets/nnunet
```

The MSD dataset will be downloaded and prepared automatically by the nnunet example script if it does not already exist. The dataset_path flag is used as more of a data working directory by the client. The client will create nnunet_raw, nnunet_preprocessed and nnunet_results sub directories if they do not already exist in the dataset_path folder. The dataset itself will be stored in a folder within nnunet_raw. Therefore when checking if the data already exists, the client will look for the following folder '{dataset_path}/nnunet_raw/{dataset_name}'
