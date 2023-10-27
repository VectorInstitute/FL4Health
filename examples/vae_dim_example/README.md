## Run on your local machine
python -m examples.vae_dim_example.server 

python -m examples.vae_dim_example.client --dataset_path examples/datasets/MNIST

python -m examples.vae_dim_example.client --dataset_path examples/datasets/MNIST

## Cluster
sbatch examples/vae_dim_example/run.sh examples/vae_dim_example/config.yaml examples/vae_dim_example/ "ex1" ~/venv/FL4Health/