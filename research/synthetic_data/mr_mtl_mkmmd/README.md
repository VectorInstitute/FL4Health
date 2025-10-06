### Running hyperparameter sweep

To run the hyperparameter sweep you simply run the command

```bash
./research/synthetic_data/mr_mtl_mkmmd/run_hp_sweep.sh \
   path_to_config.yaml \
   path_to_folder_for_artifacts/ \
   path_to_folder_for_dataset/ \
   path_to_desired_venv/
```

from the top level directory of the repository

An example is something like
``` bash
./research/synthetic_data/mr_mtl_mkmmd/run_hp_sweep.sh \
   research/synthetic_data/mr_mtl_mkmmd/config.yaml \
   research/synthetic_data/mr_mtl_mkmmd/ \
   /path/to/synthetic_dataset/ \
   /h/demerson/vector_repositories/fl4health_env/
```

In order to manipulate the grid search being conducted, you need to change the parameters for `lr`, the client-side learning rate, and `lam`, the mr_mtl loss weight for training,`mu`, the mkmmd loss for mr_mtl model structure and `l2`, the l2 regularization parameter for features, in the `run_hp_sweep.sh` script directly.
