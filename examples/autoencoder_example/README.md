
## Run on your local machine
python -m examples.autoencoder_example.server 

python -m examples.autoencoder_example.client --dataset_path examples/datasets/MNIST --artifact_dir "examples/autoencoder_example/distributions/client_1/"

python -m examples.autoencoder_example.client --dataset_path examples/datasets/MNIST --artifact_dir "examples/autoencoder_example/distributions/client_2/"

## Reconstruction
python -m examples.autoencoder_example.reconstruct_data

## Generate new samples
python -m examples.autoencoder_example.generate_new_samples

## Plot the data distribution of the last run
python -m examples.autoencoder_example.distributions.plot_data_distibution

## Run on cluster
for the 2 client setting, create two folders in this directory named after clients: client_0 and client_1.

sbatch examples/autoencoder_example/run.sh examples/autoencoder_example/config.yaml examples/autoencoder_example/ examples/autoencoder_example/ ~/venv/FL4Health/

