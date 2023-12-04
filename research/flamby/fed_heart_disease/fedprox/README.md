### Running hyperparameter sweep

To run the hyperparameter sweep you simply run the command

```bash
./research/flamby/fed_heart_disease/fedprox/run_hp_sweep.sh \
   path_to_config.yaml \
   path_to_folder_for_artifacts/ \
   path_to_folder_for_dataset/ \
   path_to_desired_venv/
```

from the top level directory of the repository

An example is something like
``` bash
./research/flamby/fed_heart_disease/fedprox/run_hp_sweep.sh \
   research/flamby/fed_heart_disease/fedprox/config.yaml \
   research/flamby/fed_heart_disease/fedprox/ \
   /Users/david/Desktop/FLambyDatasets/fed_heart_disease/ \
   /h/demerson/vector_repositories/fl4health_env/
```

In order to manipulate the grid search being conducted, you need to change the parameters for `mu` and `lr`, the FedProx penalty weight and the learning rate, respectively, in the `run_hp_sweep.sh` script directly.

### Large Model Experiments

The default setup for these experiments is "small" models using the Baseline() model implemented by FLamby. This "small" model is simply a logistic regression model with a very small number of trainable parameters. To run experiments with the "large" model, which incorporates an equivalent number of trainable parameters to the FENDA model implementation. To use the large model, one need only replace instances of Baseline() with FedHeartDiseaseLargeBaseline(), along with including the proper imports in the experimental code. The large model is implemented here:

```
research/flamby/fed_heart_disease/large_baseline.py
```
