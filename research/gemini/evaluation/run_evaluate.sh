#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=5000MB
#SBATCH --partition=gpu
#SBATCH --qos=hipri
#SBATCH --job-name=eval_fedper_het
#SBATCH --output=%j_%x.out
#SBATCH --error=%j_%x.err
#SBATCH --mail-user=your_email@vectorinstitute.ai


# Source the environment
. ~/flenv/bin/activate
echo "Active Environment:"
which python


python -m evaluation.evaluate_on_holdout --artifact_dir "fedper/delirium_het_runs/hp_sweep_results/lr_0.00001" --task "delirium" --eval_write_path "fedper/delirium_het_eval" --n_clients 6
