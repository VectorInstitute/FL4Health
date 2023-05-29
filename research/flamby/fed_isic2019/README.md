### Evaluating outputs of the hyperparameter search.

The hyperparameter sweep performs five training runs for each pair of parameters in the sweep. The artifacts of the training runs are stored under the same umbrella folder and are processed together to get an average performance on the test set. The `evaluate_on_holdout.py` script evaluates both the best local client models (saved during training by achieving the best loss of the validation set) on local client test sets and the best global model (as measured by model weights with the best average performance on the respective validation sets) on the pooled central test set. The script reports the average and standard deviation for both of these measures.

To run this on a particular collection of five runs (i.e. for fixed set of hyperparameters) the command
``` bash
python -m research.flamby.fed_isic2019.fedprox.evaluate_on_holdout_global.py \
    --artifact_dir path/to/runs/to/analyze/ \
    --dataset_dir path/to/fedisic/datasets/
```

An example command is
``` bash
python -m research.flamby.fed_isic2019.fedprox.evaluate_on_holdout_global.py \
    --artifact_dir research/flamby/fed_isic2019/fedprox/hp_sweep_results/mu_0.01_lr_0.0001/ \
    --dataset_dir /Users/david/Desktop/FLambyDatasets/fedisic2019/
```

__NOTE__: You must have the correct venv activated for this to run. See the [FLamby readme](/research/flamby/README.md) for guidance.


#### Two evaluation scripts

There are two different evaluation scripts with slightly different aims.

* `evaluate_on_holdout_global.py` is meant to be run for FL methods that have a "global" model that works for all clients and "local" checkpoints based on best performance on local data. Such methods are FedProx, FedAvg, and Scaffold.
* `evaluate_on_holdout_personal.py` is meant to be run for FL methods that only have "local" models. That is, there is no "functioning" global model. Methods like APFL and FENDA only have a notion of "local" models because a portion of the model is specific to each client.
