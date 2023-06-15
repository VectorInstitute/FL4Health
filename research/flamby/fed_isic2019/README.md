### Evaluating outputs of the hyperparameter search.

The hyperparameter sweep performs five training runs for each pair of parameters in the sweep. The artifacts of the training runs are stored under the same umbrella folder and are processed together to get an average performance on the test set. The `evaluate_on_holdout.py` script evaluates both the best local client models (saved during training by achieving the best loss of the validation set) on local client test sets and the best global model (as measured by model weights with the best average performance on the respective validation sets) on the pooled central test set. The script reports the average and standard deviation for both of these measures.

To run this on a particular collection of five runs (i.e. for fixed set of hyperparameters) the command
``` bash
python -m research.flamby.fed_isic2019.evaluate_on_holdout \
    --artifact_dir path/to/runs/to/analyze/ \
    --dataset_dir path/to/fedisic/datasets/ \
    --eval_write_path path/to/write/eval/results/to.txt \
    --eval_global_model
```

An example command is
``` bash
python -m research.flamby.fed_isic2019.evaluate_on_holdout \
    --artifact_dir research/flamby/fed_isic2019/fedprox/hp_sweep_results/mu_0.01_lr_0.0001/ \
    --dataset_dir /Users/david/Desktop/FLambyDatasets/fedisic2019/ \
    --eval_write_path research/flamby/fed_isic2019/fedprox/test_eval_results.txt \
    --eval_global_model

```

__NOTE__: You must have the correct venv activated for this to run. See the [FLamby readme](/research/flamby/README.md) for guidance.


#### Two evaluation modes

There are two different evaluation modes with slightly different aims that are toggled by the flag `--eval_global_model`. Inclusion of the flag indicates that there are both "local" model checkpoints and "global" model checkpoints. With the flag activated, the script will evaluate both types of models and write out performance. Examples of methods that have global models are FedProx, FedAvg, and Scaffold.

Running the script without the `--eval_global_model` flag will only search for and evaluate "local" models. That is, models specific to each client. Methods like APFL and FENDA only have a notion of "local" models because a portion of the model is specific to each client.
