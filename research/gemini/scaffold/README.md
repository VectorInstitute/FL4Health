# Scaffold gemini experiments

## Delirium prediction

Run the hyperparameter sweep for 6 clients:



```
./Scaffold/run_hp_sweep.sh "Scaffold/config.yaml" "Scaffold/delirium_runs/" "delirium" 6
```

## Mortality


Run the hyperparameter sweep for 7 clients:

```
chmod +x scaffold/run_hp_sweep.sh

./scaffold/run_hp_sweep.sh "scaffold/config.yaml" "scaffold/7_client_results/" "mortality" 7
```


**When changing the number of clients in experiments make sure to change config.yaml as well :)**
