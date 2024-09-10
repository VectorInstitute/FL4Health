## delirium

```

./FedOpt/run_hp_sweep.sh "FedOpt/config.yaml" "FedOpt/delirium_runs/" "delirium" 6
```



## mortality:

* Important point: if your set of hyper-parameters is large: the number of the port will increase a lot. As a results, the virtual environment will not launch correctly. Make sure to keep the set of hyper-parameters limited in the run_hp_sweep.

Running the hyperparameter sweep:

```
chmod +x FedOpt/run_hp_sweep.sh
chmod +x FedOpt/run_fold_experiment.sh

./FedOpt/run_hp_sweep.sh "FedOpt/config.yaml" "FedOpt/2_client_results/" "mortality" 2
```

to restart the experiment of a specific hyper-parameter you have to remove the already existing results first:

```
rm -r FedOpt/2_client_results/hp_sweep_results/ser_0.01_lr_0.0001
```



### 7 clients

```
chmod +x FedOpt/run_hp_sweep.sh
chmod +x FedOpt/run_fold_experiment.sh

./FedOpt/run_hp_sweep.sh "FedOpt/config.yaml" "FedOpt/7_client_results/" "mortality" 7
```
