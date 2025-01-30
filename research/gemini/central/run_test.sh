#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5120MB
#SBATCH --partition=gpu
#SBATCH --qos=hipri
#SBATCH --job-name=central-testing
#SBATCH --output=%j_%x.out
#SBATCH --error=%j_%x.err
#SBATCH --mail-user=your_email@vectorinstitute.ai


# Source the environment
. ~/py39/bin/activate
echo "Active Environment:"
which python

python -m central.test --artifact_dir "central/runs_results/hp_sweep_results/lr_0.00001_epochs_200" --task "delirium" --eval_write_path "centralized/eval_results" --n_clients 6
