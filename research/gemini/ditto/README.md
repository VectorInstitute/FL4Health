# Mortality (7 clients):
**Reminder**: First set the number of clients in the `config.yaml` file.


Running the hyperparameter sweep:
```
chmod +x ditto/run_hp_sweep.sh
./ditto/run_hp_sweep.sh ditto/config.yaml ditto/mortality_runs/ "mortality" 7

```

config.yaml used for experiments:

```
# Parameters that describe server
n_server_rounds: 50 # The number of rounds to run FL

# Parameters that describe clients
n_clients: 7 # The number of clients in the FL experiment
local_epochs: 2 # The number of epochs to complete for client
batch_size: 64 # The batch size for client training

```



# Delirium (6 clients):
**Reminder**: First set the number of clients in the `config.yaml` file.

To perform extreme heterogeneity experiment use 300 as the model size instead of size 8093, and don't forget to apply the change in both `client.py` and `server.py`. Also, the path to the data source should be adjusted to `data_path = Path("heterogeneous_data")`

Running the hyperparameter sweep:
```
chmod +x ditto/run_hp_sweep.sh
./ditto/run_hp_sweep.sh ditto/config.yaml ditto/delirium_runs/ "delirium" 6

```
config.yaml used for experiments:

```
# Parameters that describe server
n_server_rounds: 50 # The number of rounds to run FL

# Parameters that describe clients
n_clients: 6 # The number of clients in the FL experiment
local_epochs: 4 # The number of epochs to complete for client
batch_size: 64 # The batch size for client training

```
