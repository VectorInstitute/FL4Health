# APFL

## Delirium prediction

Run the hyperparameter sweep for 6 clients:


```
./apfl/run_hp_sweep.sh "apfl/config.yaml" "apfl/delirium_runs/" "delirium" 6
```


## Mortality

## Mortality prediction:

Run the hyperparameter sweep for 7 clients:

```
chmod +x apfl/run_hp_sweep.sh

./apfl/run_hp_sweep.sh "apfl/config.yaml" "apfl/7_client_results/" "mortality" 7
```

To restart the experiment for a specific hyper-parameter you have to remove the already existing results folder first.

**When changing the number of clients in experiments make sure to change config.yaml as well :)**
