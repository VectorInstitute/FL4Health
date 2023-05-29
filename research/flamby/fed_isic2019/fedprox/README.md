### Running hyperparameter sweep

To run the hyperparameter sweep you simply run the command

```bash
./research/flamby/fed_isic2019/fedprox/run_hp_sweep.sh \
   path_to_config.yaml \
   path_to_folder_for_artifacts/ \
   path_to_folder_for_dataset/ \
   path_to_desired_venv/
```

from the top level directory of the repository

An example is something like
``` bash
./research/flamby/fed_isic2019/fedprox/run_hp_sweep.sh \
   research/flamby/fed_isic2019/fedprox/config.yaml \
   research/flamby/fed_isic2019/fedprox/ \
   /Users/david/Desktop/FLambyDatasets/fedisic2019/ \
   /h/demerson/vector_repositories/fl4health_env/
```

In order to manipulate the grid search being conducted, you need to chanage the parameters for `mu` and `lr`, the FedProx penalty weight and the learning rate, respectively, in the `run_hp_sweep.sh` script directly.

### Evaluating outputs of the hyperparameter search.

The hyperparameter sweep performs five training runs for each pair of parameters in the sweep. The artifacts of the training runs are stored under the same umbrella folder and are processed together to get an average performance on the test set. The `evaluate_on_holdout.py` script evaluates both the best local client models (saved during training by achieving the best loss of the validation set) on local client test sets and the best global model (as measured by model weights with the best average performance on the respective validation sets) on the pooled central test set. The script reports the average and standard deviation for both of these measures.

To run this on a particular collection of five runs (i.e. for fixed set of hyperparameters) the command
``` bash
python -m research.flamby.fed_isic2019.fedprox.evaluate_on_holdout.py \
    --artifact_dir path/to/runs/to/analyze/ \
    --dataset_dir path/to/fedisic/datasets/
```

An example command is
``` bash
python -m research.flamby.fed_isic2019.fedprox.evaluate_on_holdout.py \
    --artifact_dir research/flamby/fed_isic2019/fedprox/hp_sweep_results/mu_0.01_lr_0.0001/ \
    --dataset_dir /Users/david/Desktop/FLambyDatasets/fedisic2019/
```

__NOTE__: You must have the correct venv activated for this to run. See the [FLamby readme](/research/flamby/README.md) for guidance.
