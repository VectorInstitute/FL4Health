
## Run on your local machine 
python -m examples.autoencoder_example.server 

python -m examples.autoencoder_example.client --dataset_path examples/datasets/MNIST --artifact_dir "examples/autoencoder_example/distributions/client_1/"

python -m examples.autoencoder_example.client --dataset_path examples/datasets/MNIST --artifact_dir "examples/autoencoder_example/distributions/client_2/"

## Reconstruction
python -m examples.autoencoder_example.reconstruct_samples --experiment_name "beta=1" --laten_dim=16 --label=2 

## Generate new samples
```
python -m examples.autoencoder_example.generate_new_samples --experiment_name "beta=1" --laten_dim 64 --n_images 64
```
Or:
```
python -m examples.autoencoder_example.generate_new_samples --experiment_name "beta=100_dim16" --laten_dim 16 --n_images 64
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

### Experiments
Experiments on the stryucture of VAE showed having MLP of linear layers is very effective for generating MNIST samples. But using Conv layers we could get to an also good model with some search.

### Final
After some search, the final ConvVae() seems to be able to generate good samples. There are some structural design changes that made the model better:
- Removing the pooling layers helps a lot.
- Increased the size of channels in conv and conT but kept the laten size small (2).
- Trained for more epochs is actually effective even if we don't see a huge change in loss. I trained for max 200 server rounds with 2 local epochs. 
- It is good to have a linear layer before and 2 linear layers after the laten layer. The layer before and after the laten space have a size of 64.    

### Beta=100:
```
sbatch examples/autoencoder_example/run.sh examples/autoencoder_example/config.yaml examples/autoencoder_example/ "final" ~/venv/FL4Health/
```

#### Beta=1:   
```
sbatch examples/autoencoder_example/run.sh examples/autoencoder_example/config.yaml examples/autoencoder_example/ "fina_beta1" ~/venv/FL4Health/
```