### Running centralized training

Centralized training refers to the standard means of training ML models. That is, all of the FedIsic data is "centralized" into a single pool of data from all clients. Then a single model is trained using the pooled data.

For centralized training, the process is much simpler than the FL counterparts. This is because the FLamby paper already ran hyper-parameter sweeps on central model training. So no sweep is necessary. You need only follow the documentation associated with the `run_fold_experiment.slrm` script in order to produce a set of centralized models to be evaluated.
