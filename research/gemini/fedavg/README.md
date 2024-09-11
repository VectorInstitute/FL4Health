
## Delirium prediction

Run the hyperparameter sweep for 6 clients:

```
./fedavg/run_hp_sweep.sh "fedavg/config.yaml" "fedavg/delirium_runs/" "delirium" 6
```


## Mortality prediction:

Run the hyperparameter sweep for 7 clients:


```
./fedavg/run_hp_sweep.sh "fedavg/config.yaml" "fedavg/path_to_results_folder/" "mortality" 7
```


**When changing the number of clients in experiments make sure to change config.yaml as well :)**
