#!/bin/bash
#SBATCH --job-name=isic_eval

#SBATCH --partition=t4v2

#SBATCH --gres=gpu:1

#SBATCH --qos=m2

#SBATCH --cpus-per-task=4

#SBATCH --mem-per-cpu=8G

#SBATCH --output=/h/sayromlou/FL/FL4Health/isic_eval_fenda.out

#SBATCH --error=/h/sayromlou/FL/FL4Health/isic_eval_fenda.err

# prepare your environment here
source /ssd003/projects/aieng/public/FL_env/env3/bin/activate

# put your command here
# python3 -m research.flamby.fed_isic2019.evaluate_on_holdout --artifact_dir  /ssd003/projects/aieng/public/FL_env/models/fed_isic2019/fedavg/hp_sweep_results/lr_0.001/ --dataset_dir /ssd003/projects/aieng/public/flamby_datasets/fed_isic2019/ --eval_write_path ~/FL/FL4Health/research/flamby/fed_isic2019/fedavg/test_eval_performance.txt --eval_local_models

# python3 -m research.flamby.fed_isic2019.evaluate_on_holdout --artifact_dir  /ssd003/projects/aieng/public/FL_env/models/fed_isic2019/fedavg/hp_sweep_results_pre_trained/lr_0.001/ --dataset_dir /ssd003/projects/aieng/public/flamby_datasets/fed_isic2019/ --eval_write_path ~/FL/FL4Health/research/flamby/fed_isic2019/fedavg/test_eval_performance_pre_trained.txt --eval_local_models

# python3 -m research.flamby.fed_isic2019.evaluate_on_holdout --artifact_dir /ssd003/projects/aieng/public/FL_env/models/fed_isic2019/fenda/hp_sweep_results_contrastive_pre_trained/lr_0.001/ --dataset_dir /ssd003/projects/aieng/public/flamby_datasets/fed_isic2019/ --eval_write_path ~/FL/FL4Health/research/flamby/fed_isic2019/fenda/test_eval_performance_contrastive_pre_trained.txt --eval_local_models


# python3 -m research.flamby.fed_isic2019.evaluate_on_holdout --artifact_dir /ssd003/projects/aieng/public/FL_env/models/fed_isic2019/fenda/hp_sweep_results_cos_sim_pre_trained/lr_0.001/ --dataset_dir /ssd003/projects/aieng/public/flamby_datasets/fed_isic2019/ --eval_write_path ~/FL/FL4Health/research/flamby/fed_isic2019/fenda/test_eval_performance_cos_sim_pre_trained.txt --eval_local_models

# python3 -m research.flamby.fed_isic2019.evaluate_on_holdout --artifact_dir /ssd003/projects/aieng/public/FL_env/models/fed_isic2019/fenda/hp_sweep_results_pre_trained/lr_0.001/ --dataset_dir /ssd003/projects/aieng/public/flamby_datasets/fed_isic2019/ --eval_write_path ~/FL/FL4Health/research/flamby/fed_isic2019/fenda/test_eval_performance_pre_trained.txt --eval_local_models

# python3 -m research.flamby.fed_isic2019.evaluate_on_holdout --artifact_dir /ssd003/projects/aieng/public/FL_env/models/fed_isic2019/fenda/hp_sweep_results_perFCL_pre_trained/lr_0.001/ --dataset_dir /ssd003/projects/aieng/public/flamby_datasets/fed_isic2019/ --eval_write_path ~/FL/FL4Health/research/flamby/fed_isic2019/fenda/test_eval_performance_perFCL_pre_trained.txt --eval_local_models

python3 -m research.flamby.fed_isic2019.evaluate_on_holdout --artifact_dir /ssd003/projects/aieng/public/FL_env/models/fed_isic2019/fenda/hp_sweep_results_new/lr_0.001/ --dataset_dir /ssd003/projects/aieng/public/flamby_datasets/fed_isic2019/ --eval_write_path ~/FL/FL4Health/research/flamby/fed_isic2019/fenda/test_eval_performance_new.txt --eval_local_models

# python3 -m research.flamby.fed_isic2019.evaluate_on_holdout --artifact_dir /ssd003/projects/aieng/public/FL_env/models/fed_isic2019/fenda/hp_sweep_results_perFCL/lr_0.001/ --dataset_dir /ssd003/projects/aieng/public/flamby_datasets/fed_isic2019/ --eval_write_path ~/FL/FL4Health/research/flamby/fed_isic2019/fenda/test_eval_performance_perFCL.txt --eval_local_models

# python3 -m research.flamby.fed_isic2019.evaluate_on_holdout --artifact_dir /ssd003/projects/aieng/public/FL_env/models/fed_isic2019/fenda/hp_sweep_results_cos_sim/lr_0.001/ --dataset_dir /ssd003/projects/aieng/public/flamby_datasets/fed_isic2019/ --eval_write_path ~/FL/FL4Health/research/flamby/fed_isic2019/fenda/test_eval_performance_cos_sim.txt --eval_local_models

# python3 -m research.flamby.fed_isic2019.evaluate_on_holdout --artifact_dir /ssd003/projects/aieng/public/FL_env/models/fed_isic2019/fenda/hp_sweep_results_contrastive/lr_0.001/ --dataset_dir /ssd003/projects/aieng/public/flamby_datasets/fed_isic2019/ --eval_write_path ~/FL/FL4Health/research/flamby/fed_isic2019/fenda/test_eval_performance_contrastive.txt --eval_local_models

# python3 -m research.flamby.fed_isic2019.evaluate_on_holdout --artifact_dir /ssd003/projects/aieng/public/FL_env/models/fed_isic2019/fenda/hp_sweep_results_new/lr_0.001/ --dataset_dir /ssd003/projects/aieng/public/flamby_datasets/fed_isic2019/ --eval_write_path ~/FL/FL4Health/research/flamby/fed_isic2019/fenda/test_eval_performance_new.txt --eval_local_models

# python3 -m research.flamby.fed_isic2019.evaluate_on_holdout --artifact_dir /ssd003/projects/aieng/public/FL_env/models/fed_isic2019/moon/hp_sweep_results_pre_trained/lr_0.001/ --dataset_dir /ssd003/projects/aieng/public/flamby_datasets/fed_isic2019/ --eval_write_path ~/FL/FL4Health/research/flamby/fed_isic2019/moon/test_eval_performance_pre_trained.txt --eval_local_models

# python3 -m research.flamby.fed_isic2019.evaluate_on_holdout --artifact_dir research/flamby/fed_isic2019/fenda/hp_sweep_results_contrastive/lr_0.001/ --dataset_dir /ssd003/projects/aieng/public/flamby_datasets/fed_isic2019/ --eval_write_path ~/FL/FL4Health/research/flamby/fed_isic2019/fenda/test_eval_performance.txt --eval_local_models

# python3 -m research.flamby.fed_isic2019.evaluate_on_holdout --artifact_dir research/flamby/fed_isic2019/fenda/hp_sweep_results_cos_sim/lr_0.001/ --dataset_dir /ssd003/projects/aieng/public/flamby_datasets/fed_isic2019/ --eval_write_path ~/FL/FL4Health/research/flamby/fed_isic2019/fenda/test_eval_performance.txt --eval_local_models

# python3 -m research.flamby.fed_isic2019.evaluate_on_holdout --artifact_dir /ssd003/projects/aieng/public/FL_env/models/fed_isic2019/fenda/hp_sweep_results_contrastive_pre_train/lr_0.001/ --dataset_dir /ssd003/projects/aieng/public/flamby_datasets/fed_isic2019/ --eval_write_path ~/FL/FL4Health/research/flamby/fed_isic2019/fenda/test_eval_performance.txt --eval_local_models