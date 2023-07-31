### Running local client training

In local client training, each client trains a "global" model on its local data, which will be tested on all clients data together. This setting simulates the situation where, for example, a single hospital tries to use only its own data to train a model that will be deployed in other hospital settings. I.e. a test of generalization.

As with the centralized training setting, the training process is a bit simpler, as the model and training hyper-parameters have already been investigated by the FLamby paper. There is a script to orchestrate training each of the client based models (4 clients in total).

To run local training for each of the clients you simply run the command

```bash
./research/flamby/fed_heart_disease/local/run_all_clients.sh \
   path_to_folder_for_artifacts/ \
   path_to_folder_for_dataset/ \
   path_to_desired_venv/
```

from the top level directory of the repository

An example is something like
```bash
./research/flamby/fed_heart_disease/local/run_all_clients.sh \
   research/flamby/fed_heart_disease/local/ \
   /Users/david/Desktop/FLambyDatasets/fed_heart_disease/ \
   /h/demerson/vector_repositories/fl4health_env/
```

### Large Model Experiments

The default setup for these experiments is "small" models using the Baseline() model implemented by FLamby. This "small" model is simply a logistic regression model with a very small number of trainable parameters. To run experiments with the "large" model, which incorporates an equivalent number of trainable parameters to the FENDA model implementation. To use the large model, one need only replace instances of Baseline() with FedHeartDiseaseLargeBaseline(), along with including the proper imports in the experimental code. The large model is implemented here:

```
research/flamby/fed_heart_disease/large_baseline.py
```
