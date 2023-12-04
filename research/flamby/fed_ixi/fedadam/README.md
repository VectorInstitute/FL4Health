### Running hyperparameter sweep

To run the hyperparameter sweep you simply run the command

```bash
./research/flamby/fed_ixi/fedadam/run_hp_sweep.sh \
   path_to_config.yaml \
   path_to_folder_for_artifacts/ \
   path_to_folder_for_dataset/ \
   path_to_desired_venv/
```

from the top level directory of the repository

An example is something like
``` bash
./research/flamby/fed_ixi/fedadam/run_hp_sweep.sh \
   research/flamby/fed_ixi/fedadam/config.yaml \
   research/flamby/fed_ixi/fedadam/ \
   /Users/david/Desktop/FLambyDatasets/fed_ixi/ \
   /h/demerson/vector_repositories/fl4health_env/
```

In order to manipulate the grid search being conducted, you need to change the parameters for `server_lr` and `lr`, the server and client learning rates, respectively, in the `run_hp_sweep.sh` script directly.
