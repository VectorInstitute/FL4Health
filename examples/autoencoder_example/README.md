
python -m examples.autoencoder_example.server 

python -m examples.autoencoder_example.client --dataset_path examples/datasets/MNIST

python -m examples.autoencoder_example.reconstruct_data


python -m examples.autoencoder_example.generate_new_samples

## run on cluster
sbatch examples/autoencoder_example/run.sh examples/autoencoder_example/config.yaml examples/autoencoder_example/output/ examples/autoencoder_example/output/ ~/venv/FL4Health/

