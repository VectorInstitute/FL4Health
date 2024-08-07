# Mortality

## Ditto
### Hyper-parameters
```
python -m evaluation.find_best_hp --hp_sweep_dir "ditto/mortality_runs/hp_sweep_results"


Best Loss: 0.26501005024900615
Best Folder: ditto/mortality_runs/hp_sweep_results/lam_0.01_lr_0.0001
```

### Evaluation
```
python -m evaluation.evaluate_on_holdout --artifact_dir "ditto/mortality_runs/hp_sweep_results/lam_0.01_lr_0.0001" --task "mortality" --eval_write_path "ditto/mortality_eval" --n_clients 7
```
### run
```
sbatch evaluation/run_eval.sh
```

## Moon
### Hyper-parameters

```
python -m evaluation.find_best_hp --hp_sweep_dir "moon/mortality_runs/hp_sweep_results"

new hp results
Best Loss: 0.26972746666502967
Best Folder: moon/mortality_runs/hp_sweep_results/mu_0.001_lr_0.0001
```

### Evaluation
```
python -m evaluation.evaluate_on_holdout --artifact_dir "moon/mortality_runs/hp_sweep_results/mu_0.001_lr_0.0001" --task "mortality" --eval_write_path "moon/mortality_eval_new" --n_clients 7  --eval_global_model
```
### run
sbatch evaluation/run_eval.sh



## FedPer
### Hyper-parameters
```
python -m evaluation.find_best_hp --hp_sweep_dir "fedper/mortality_runs/hp_sweep_results"

results
Best Loss: 0.26800094877638986
Best Folder: fedper/mortality_runs/hp_sweep_results/lr_0.0001
```
### Evaluation
```
python -m evaluation.evaluate_on_holdout --artifact_dir "fedper/mortality_runs/hp_sweep_results/lr_0.0001" --task "mortality" --eval_write_path "fedper/mortality_eval" --n_clients 7
```
### run
```
sbatch evaluation/run_eval.sh
```



## PerFcl
### Hyper-parameters
```
python -m evaluation.find_best_hp --hp_sweep_dir "perfcl/mortality_runs/hp_sweep_results"

results
Best Loss: 0.2675459296210489
Best Folder: perfcl/mortality_runs/hp_sweep_results/gamma_10_mu_0.1_lr_0.001
```
### Evaluation
```
python -m evaluation.evaluate_on_holdout --artifact_dir "perfcl/mortality_runs/hp_sweep_results/gamma_10_mu_0.1_lr_0.001" --task "mortality" --eval_write_path "perfcl/mortality_eval" --n_clients 7
```
### run
```
sbatch evaluation/run_eval.sh
```



## FedProx

7 clients.
### Hyper-parameters

```
python -m evaluation.find_best_hp --hp_sweep_dir "FedProx/7_client_results/hp_sweep_results"


Best Loss: 0.2679186776572838
Best Folder: FedProx/7_client_results/hp_sweep_results/mu_0.1_lr_0.01
```

### Evaluation
```
python -m evaluation.evaluate_on_holdout --artifact_dir "FedProx/7_client_results/hp_sweep_results/mu_0.1_lr_0.01" --task "mortality" --eval_write_path "FedProx/evaluation_test/7-client" --n_clients 7 --eval_global_model
```



## FedAvg
7 client.
### Hyper-parameters
```
python -m evaluation.find_best_hp --hp_sweep_dir "FedAvg/7_client_results/hp_sweep_results"
```

0.27039574861006604
Best Folder: FedAvg/7_client_results/hp_sweep_results/lr_0.001

### Evaluation
```
python -m evaluation.evaluate_on_holdout --artifact_dir "FedAvg/7_client_results/hp_sweep_results/lr_0.001" --task "mortality" --eval_write_path "FedAvg/evaluation_test/7-client" --n_clients 7 --eval_global_model

```


## Fenda
7 clients.

### Hyper-parameters
```
python -m evaluation.find_best_hp --hp_sweep_dir "Fenda/7_client_results/hp_sweep_results"

Best Loss: 0.26332475776813774
Best Folder: Fenda/7_client_results_v3/hp_sweep_results/lr_0.0001
```
### Evaluation
```
python -m evaluation.evaluate_on_holdout --artifact_dir "Fenda/7_client_results/hp_sweep_results/lr_0.0001" --task "mortality" --eval_write_path "Fenda/evaluation_test/7-client" --n_clients 7
```


## APFL
7 clients.
### Hyper-parameters

```
python -m evaluation.find_best_hp --hp_sweep_dir "Apfl/7_client_results/hp_sweep_results"

Results:
Best Loss: 0.2647731873152613
Best Folder: Apfl/7_client_results/hp_sweep_results/alpha_0.01_lr_0.01
```
```
python -m evaluation.evaluate_on_holdout --artifact_dir "Apfl/7_client_results/hp_sweep_results/alpha_0.01_lr_0.01" --task "mortality" --eval_write_path "Apfl/evaluation_test/7-client" --n_clients 7 --is_apfl
```
