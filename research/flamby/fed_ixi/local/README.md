### Running local client training

In local client training, each client trains a "global" model on its local data, which will be tested on all clients data together. This setting simulates the situation where, for example, a single hospital tries to use only its own data to train a model that will be deployed in other hospital settings. I.e. a test of generalization.

As with the centralized training setting, the training process is a bit simpler, as the model and training hyper-parameters have already been investigated by the FLamby paper. There is a script to orchestrate training each of the client based models (6 clients in total).

To run local training for each of the clients you simply run the command

```bash
./research/flamby/fed_ixi/local/run_all_clients.sh \
   path_to_folder_for_artifacts/ \
   path_to_folder_for_dataset/ \
   path_to_desired_venv/
```

from the top level directory of the repository

An example is something like
```bash
./research/flamby/fed_ixi/local/run_all_clients.sh \
   research/flamby/fed_ixi/local/ \
   /Users/david/Desktop/FLambyDatasets/fed_ixi/ \
   /h/demerson/vector_repositories/fl4health_env/
```
