
## Run on your local machine
python -m examples.autoencoder_example.server 

python -m examples.autoencoder_example.client --dataset_path examples/datasets/MNIST --artifact_dir "examples/autoencoder_example/distributions/client_1/"

python -m examples.autoencoder_example.client --dataset_path examples/datasets/MNIST --artifact_dir "examples/autoencoder_example/distributions/client_2/"

## Reconstruction
python -m examples.autoencoder_example.reconstruct_data

## Generate new samples
```
python -m examples.autoencoder_example.generate_new_samples --experiment_name "beta=1" --laten_dim 64
```
Or:
```
python -m examples.autoencoder_example.generate_new_samples --experiment_name "beta=100" --laten_dim 64
```

## Plot the data distribution of the last run
This cript will plot the distribution of clients' data (MNIST data).

```
python -m examples.autoencoder_example.plot_data_distibution --experiment_name "beta=1" --n_clients 2
```

## Run on cluster
TODO: get the beta as input to the program.
```
sbatch examples/autoencoder_example/run.sh path/to/server/config_file.yaml path/to/log/experiment/output_files  "experiment_name" path/to/virtual_environment
```

For example:
```
sbatch examples/autoencoder_example/run.sh examples/autoencoder_example/config.yaml examples/autoencoder_example/ "beta=1" ~/venv/FL4Health/
```

For the less heterogenious setting:

```
sbatch examples/autoencoder_example/run.sh examples/autoencoder_example/config.yaml examples/autoencoder_example/ "beta=100" ~/venv/FL4Health/
```

Train beta=100 for 70 total rounds:
```
sbatch examples/autoencoder_example/run.sh examples/autoencoder_example/config.yaml examples/autoencoder_example/ "beta=100_r70" ~/venv/FL4Health/
```