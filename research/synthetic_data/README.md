### Generate the synthetic dataset

To do so we should run the following script:

```bash
python -m research.synthetic_data.preprocess --save_dataset_dir path_to_save_partitioned_dataset --seed seed --alpha alpha --beta beta --num_clients num_clients
```

Where:
- `path_to_save_partitioned_dataset` is the path to save the partitioned dataset.
- `seed` is the seed to use for the random number generator to have reproducible splits.
- `alpha` is the imbalance level of the dataset. The higher the value, the more imbalance in the data distribution
- `beta` is also the imbalance level of the dataset. The higher the value, the more imbalance in the data distribution.
- `num_clients` is the number of clients to partition the dataset into.


### Training federated models on synthetic data

The training scripts for each federated learning approach are located in the run_fold_experiment.slrm file. To perform a hyperparameter sweep for a specific approach, execute the corresponding bash script, such as run_hp_sweep.sh. For instance, to run a hyperparameter sweep for the FedAvg approach, use the following command:

```bash
./research/synthetic_data/fedavg/run_hp_sweep.sh \
   path_to_config.yaml \
   path_to_folder_for_artifacts/ \
   path_to_folder_for_dataset/ \
   path_to_desired_venv/
```

Where:
- `path_to_config.yaml` is the path to the configuration file for the approach.
- `path_to_folder_for_artifacts/` is the path to the folder where the artifacts of the hyperparameter sweep will be saved.
- `path_to_folder_for_dataset/` is the path to the folder where the partitioned synthetic dataset is saved.
- `path_to_desired_venv/` is the path to the virtual environment.

#### Checkpointing the models during training

For each of these algorithms, we maintain local models and, in some cases, a global model. We configure four distinct checkpointing strategies for client-side models:
- `Pre-aggregation Best Client Model:` The best client model is saved for each client before the aggregation step, based on the validation loss over their individual validation set.
- `Pre-aggregation Last Client Model:` The last client model is saved for each client before the aggregation step.
- `Post-aggregation Best Client Model:` The best client model is saved for each client after the aggregation step, based on the validation loss over their individual validation set.
- `Post-aggregation Last Client Model:` The last client model is saved for each client after the aggregation step.

For the global model (applied only to general federated approaches like FedAvg), we have two checkpointing strategies:
- `Best Server Model:` The best global model is saved based on the average validation loss across all clients on their respective validation sets.
- `Last Server Model:` The last global model is saved.

### Evaluating outputs of the hyperparameter search.

The hyperparameter sweep conducts three training runs for each parameter pair in the sweep. The artifacts from these runs are stored in the same directory and processed together to compute the average performance on the test set. The evaluate_on_test.py script evaluates both the saved local client models and the global server models on the individual local clients' test sets, as well as on the pooled central test dataset. The script reports the average and standard deviation for both evaluations.

To run this evaluation on a specific collection of three runs (i.e., for a fixed set of hyperparameters), use the following command:

``` bash
python -m research.synthetic_data.evaluate_on_test \
    --artifact_dir path/to/runs/to/analyze/ \
    --dataset_dir path_to_folder_for_dataset \
    --eval_write_path path_to_write_eval_results_to.txt \
    --alpha alpha \
    --beta beta \
    --eval_best_pre_aggregation_local_models \
    --eval_last_pre_aggregation_local_models \
    --eval_best_post_aggregation_local_models \
    --eval_last_post_aggregation_local_models \
    --eval_best_global_model \
    --eval_last_global_model \
    --eval_over_aggregated_test_data \
    --use_partitioned_data
```


Where:
- `--eval_last_pre_aggregation_local_models` tells the evaluation script to search for saved last per-aggregation local models for each client. It looks for models named `pre_aggregation_client_{client_number}_last_model.pkl` for each client number and evaluates them on their client's specific data.
- `--eval_best_post_aggregation_local_models` tells the evaluation script to search for saved best post-aggregation local models for each client. It looks for models named `post_aggregation_client_{client_number}_best_model.pkl` for each client number and evaluates them on their client's specific data.
- `--eval_last_post_aggregation_local_models` tells the evaluation script to search for saved last post-aggregation local models for each client. It looks for models named `post_aggregation_client_{client_number}_last_model.pkl` for each client number and evaluates them on their client's specific data.
- `--eval_best_global_model` tells the evaluation script to search for the saved best global model on the server side. It looks for a model named `server_best_model.pkl` and evaluates it across all clients' individual datasets.
- `--eval_last_global_model` tells the evaluation script to search for the saved last global model on the server side. It looks for a model named `server_last_model.pkl` and evaluates it across all clients' individual datasets.
- `--eval_over_aggregated_test_data` tells the evaluation script to evaluate any model from the previous steps on the pooled test data.
- `--use_partitioned_data` tells the evaluation script to use preprocessed partitioned data for evaluation. If this flag is not set, the script will use the original synthetic dataset and partition a subset of data for each client based on a fixed seed.
