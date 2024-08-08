
# Global experiment scripts
Learning rates used in hyper-parameter sweep: lr = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]

## Delirium prediction


Arguments:
1) Address to the results folder.
2) Task to be performed on GEMINI.
3) Learning rate.
4) Batch size.
5) Number of epochs of training.


### Training
A file with the name  `delirium_eval` in the current directory should be created to hold the evaluation results.

Run:

```
sbatch central/run_central.sh "central/delirium_runs/" "delirium" learning_rate batch_size num_epochs
```

### Testing
After finding the best hyper=parameter:
In `run_test.sh` replace,

```
python -m central.test --artifact_dir "run_results_dir" --task "delirium" --eval_write_path "central/delirium_eval" --n_clients 6

sbatch central/run_test.sh
```

### Results

Final results in : `delirium_eval`

Results for the extreme heterogeneity setting: ` delirium_eval_extreme `

## Mortality prediction

Arguments:
1) Address to the results folder.
2) Task to be performed on GEMINI.
3) Learning rate.
4) Batch size.
5) Number of epochs of training.

### Train


```
sbatch central/run_central.sh "central/results/" "mortality" learning_rate batch_size epochs
```


To run a single experiment without scripts:

```
python -m central.train --artifact_dir "central/results/single_run/" --run_name "run0_0.1_40" --learning_rate 0.1 --batch_size 40 --task "mortality"
```



### 7-client testing
Number of the clients matters for testing.


```
python -m central.test --artifact_dir "central/results/hp_sweep_results/lr_X_epochs_X" --task "mortality" --eval_write_path "central/evaluation/7-client" --n_clients 7
```
Run the script for testing:

```
chmod +x central/run_test.sh
sbatch central/run_test.sh
```
