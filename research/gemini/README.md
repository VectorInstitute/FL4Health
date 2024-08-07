## Codebase for GEMINI experiments.
You can find the scripts and commands to run the experiments in each algorithm folder.

## Delirium extreme heterogeneity experiments
The experiments, by default, use naturally heterogeneous data splits. To perform extreme heterogeneity experiments, use `300` as the model's first layer size instead of `8093` when defining the model. This change can be easily applied by setting this parameter in model constructor. Don't forget to apply the change in both `client.py` and `server.py`. Also, the path to the data source should be adjusted to `data_path = Path("heterogeneous_data")`
