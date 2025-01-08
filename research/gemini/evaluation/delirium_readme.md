# Delirium

## Ditto
### hyper-parameters
```
python -m evaluation.find_best_hp --hp_sweep_dir "ditto/delirium_runs/hp_sweep_results"

Best Loss: 0.2566139146033769
Best Folder: ditto/delirium_runs/hp_sweep_results/lam_0.01_lr_0.00001
```

### Evaluation
```
python -m evaluation.evaluate_on_holdout --artifact_dir "ditto/delirium_runs/hp_sweep_results/lam_0.01_lr_0.00001" --task "delirium" --eval_write_path "ditto/delirium_eval" --n_clients 6

```

### run
```
sbatch evaluation/run_eval.sh
```

## Moon
### hyper-parameters
```
python -m evaluation.find_best_hp --hp_sweep_dir "moon/delirium_runs/hp_sweep_results"

NEW results:
Best Loss: 0.2669057900413856
Best Folder: moon/delirium_runs/hp_sweep_results/mu_1_lr_0.0001
```
### Evaluation

```
python -m evaluation.evaluate_on_holdout --artifact_dir "moon/delirium_runs/hp_sweep_results/mu_10_lr_0.001" --task "delirium" --eval_write_path "moon/delirium_eval_new" --n_clients 6  --eval_global_model
```

### run
```
sbatch evaluation/run_eval.sh
```

## Fedper
### hyper-parameters
```
python -m evaluation.find_best_hp --hp_sweep_dir "fedper/delirium_runs/hp_sweep_results"

results:
Best Loss: 0.2985169058747112
Best Folder: fedper/delirium_runs/hp_sweep_results/lr_0.00001
```

### Evaluation
```
python -m evaluation.evaluate_on_holdout --artifact_dir "fedper/delirium_runs/hp_sweep_results/lr_0.00001" --task "delirium" --eval_write_path "fedper/delirium_eval" --n_clients 6

```

### run

```
sbatch evaluation/run_eval.sh
```

## Perfcl
### hyper-parameters
```
python -m evaluation.find_best_hp --hp_sweep_dir "perfcl/delirium_runs/hp_sweep_results"

results:
Best Loss: 0.34537425909195163
Best Folder: perfcl/delirium_runs/hp_sweep_results/gamma_1_mu_1_lr_0.001
```

### Evaluation

```
python -m evaluation.evaluate_on_holdout --artifact_dir "perfcl/delirium_runs/hp_sweep_results/gamma_1_mu_1_lr_0.001" --task "delirium" --eval_write_path "perfcl/delirium_eval" --n_clients 6

```

### run
```
sbatch evaluation/run_eval.sh
```


## APFL
### hyper-parameters
```
python -m evaluation.find_best_hp --hp_sweep_dir "Apfl/delirium_runs/hp_sweep_results"

Best Loss: 0.2397983669598008
Best Folder: Apfl/delirium_runs/hp_sweep_results/alpha_0.01_lr_0.0001
```
### Evaluation
```
python -m evaluation.evaluate_on_holdout --artifact_dir "Apfl/delirium_runs/hp_sweep_results/alpha_0.01_lr_0.0001" --task "delirium" --eval_write_path "Apfl/delirium_eval" --n_clients 6 --is_apfl
```

## FENDA
### hyper-parameters
```
python -m evaluation.find_best_hp --hp_sweep_dir "Fenda/delirium_runs/hp_sweep_results"
```
### Evaluation
```
python -m evaluation.evaluate_on_holdout --artifact_dir "Fenda/delirium_runs/hp_sweep_results/lr_0.0001" --task "delirium" --eval_write_path "Fenda/delirium_eval" --n_clients 6

```


## FedOpt (FedAdam)
### hyper-parameters
```
python -m evaluation.find_best_hp --hp_sweep_dir "FedOpt/delirium_runs/hp_sweep_results"

 Best Loss: 0.36362168865310573
Best Folder: FedOpt/delirium_runs/hp_sweep_results/ser_0.001_lr_0.0001
```
### Evaluation
```
python -m evaluation.evaluate_on_holdout --artifact_dir "FedOpt/delirium_runs/hp_sweep_results/ser_0.001_lr_0.0001" --task "delirium" --eval_write_path "FedOpt/delirium_eval" --n_clients 6 --eval_global_model

```


## SCAFFOLD
### hyper-parameters
```
python -m evaluation.find_best_hp --hp_sweep_dir "Scaffold/delirium_runs/hp_sweep_results"

Best Loss: 0.6898040584120843
Best Folder: Scaffold/delirium_runs/hp_sweep_results/ser_0.001_lr_0.001

```
### Evaluation
```
python -m evaluation.evaluate_on_holdout --artifact_dir "Scaffold/delirium_runs/hp_sweep_results/ser_0.001_lr_0.001" --task "delirium" --eval_write_path "Scaffold/delirium_eval" --n_clients 6 --eval_global_model
```
