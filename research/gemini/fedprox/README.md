## Delirium prediction
Run the hyperparameter sweep for 6 clients:
```
./FedProx/run_hp_sweep.sh FedProx/config.yaml FedProx/delirium_runs/ "delirium" 6

```


## Mortality prediction


Run the hyperparameter sweep for 7 clients:

**Reminder**: First set the number of clients in the `config.yaml` file.

```
chmod +x FedProx/run_hp_sweep.sh

./FedProx/run_hp_sweep.sh FedProx/config.yaml FedProx/7_client_results/ "mortality" 7

```

To restart the experiment first remove previous results (done.out):
