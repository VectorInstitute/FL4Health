# Running FedAvg Example

To train and validate a simple U-Net model on the Preprocessed PICAI Dataset described in the [PICAI Documentation](/research/picai/README.md) in a federated manner using FedAvg, simply submit the `launch.slrm` job to the cluster using:
```
submit launch.slrm
```

This script will request compute resources, launch a FL sever and subsequently 3 FL clients, each with its own copy of the dataset. To streamline the experimentation process, the launched server and clients will reside on the same machine. Under the hood, the `launch.slrm` script will execute the `server.py` to start the server and subsequently the `client.py` for each participating client. 

`server.py` takes the following arguments: 
-- **config_path** (str): Path to configuration file for FL Experiments. Default `./config.yaml`.
-- **sever_adress** (str): Server IP Address used to communicate with clients. Default `0.0.0.0:8080`.

On the other hand, `client.py` takes the following arguments:
- **--base_dir** (str): Base path to the PICAI dataset. Defaults to the current location on the cluster. 
- **--overviews_dir** (str): Path to the directory containing overview files for the train and validation dataset of each split. Defaults to current location on the cluster. 
-- **sever_adress** (str): Server IP Address. Default `0.0.0.0:8080`.

The FL experiment can be modified by changing the arguments passed to `server.py` and `client.py` in the `launch.slrm` script or changing the values of the configuration (ie number of FL rounds, number of local epochs, batch size, etc).
