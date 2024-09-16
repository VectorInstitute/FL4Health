# Mortality (7 clients):
**Reminder**: First set the number of clients in the `config.yaml` file.
Running the hyperparameter sweep:
```
chmod +x perfcl/run_hp_sweep.sh
./perfcl/run_hp_sweep.sh perfcl/config.yaml perfcl/mortality_runs/ "mortality" 7

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
Running the hyperparameter sweep:
```
chmod +x perfcl/run_hp_sweep.sh
./perfcl/run_hp_sweep.sh perfcl/config.yaml perfcl/delirium_het_runs/ "delirium" 6

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
