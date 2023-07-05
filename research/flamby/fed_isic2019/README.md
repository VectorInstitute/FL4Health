### Evaluating outputs of the hyperparameter search.

The hyperparameter sweep performs five training runs for each pair of parameters in the sweep. The artifacts of the training runs are stored under the same umbrella folder and are processed together to get an average performance on the test set. The `evaluate_on_holdout.py` script evaluates both the best local client models (saved during training by achieving the best loss of the validation set) on local client test sets and the best global model (as measured by model weights with the best average performance on the respective validation sets) on the pooled central test set. The script reports the average and standard deviation for both of these measures.

To run this on a particular collection of five runs (i.e. for fixed set of hyperparameters) the command
``` bash
python -m research.flamby.fed_isic2019.evaluate_on_holdout \
    --artifact_dir path/to/runs/to/analyze/ \
    --dataset_dir path/to/fedisic/datasets/ \
    --eval_write_path path/to/write/eval/results/to.txt \
    --eval_local_models \
    --eval_global_model
```

An example command for the fedprox approach is:
``` bash
python -m research.flamby.fed_isic2019.evaluate_on_holdout \
    --artifact_dir research/flamby/fed_isic2019/fedprox/hp_sweep_results/mu_0.01_lr_0.0001/ \
    --dataset_dir /Users/david/Desktop/FLambyDatasets/fedisic2019/ \
    --eval_write_path research/flamby/fed_isic2019/fedprox/test_eval_results.txt \
    --eval_local_models \
    --eval_global_model

```

__NOTE__: You must have the correct venv activated for this to run. See the [FLamby readme](/research/flamby/README.md) for guidance.


#### Two evaluation modes

There are two different evaluation modes with slightly different aims:
* `--eval_local_models` tells the evaluation script to search for local models for each client. It looks for models named `client_{client_number}_best_model.pkl` for each client number and evaluates them on their client's specific data.
* `--eval_global_model` tells the evaluation script to search for a server-side global model. It looks for a model named `server_best_model.pkl`. The script evaluates the model across all clients data, both individually and pooled.

FL approaches with both global and local models (local due to checkpointing) are: FedProx, FedAvg, FedAdam, Scaffold (include both `--eval_local_models`, `--eval_global_model`). FL approaches with only local models are FENDA and APFL (include only `--eval_local_models`).

There are two special situations for evaluation, centralized and local model training (non-FL approaches). Both result in a single server-side model (include only `--eval_global_model`)
