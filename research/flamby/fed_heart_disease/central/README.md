### Running centralized training

Centralized training refers to the standard means of training ML models. That is, all of the FedHeartDisease data is "centralized" into a single pool of data from all clients. Then a single model is trained using the pooled data.

For centralized training, the process is much simpler than the FL counterparts. This is because the FLamby paper already ran hyper-parameter sweeps on central model training. So no sweep is necessary. You need only follow the documentation associated with the `run_fold_experiment.slrm` script in order to produce a set of centralized models to be evaluated.

### Large Model Experiments

The default setup for these experiments is "small" models using the Baseline() model implemented by FLamby. This "small" model is simply a logistic regression model with a very small number of trainable parameters. To run experiments with the "large" model, which incorporates an equivalent number of trainable parameters to the FENDA model implementation. To use the large model, one need only replace instances of Baseline() with FedHeartDiseaseLargeBaseline(), along with including the proper imports in the experimental code. The large model is implemented here:

```
research/flamby/fed_heart_disease/large_baseline.py
```
