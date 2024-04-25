#!/bin/bash

SPARSITY_LEVELS=( 0.6 0.7 0.8 )
BEST_LR_VALUES=( 0.001 )

for ((i=0; i<${#SPARSITY_LEVELS[@]}; i++))
do
    SPARSITY_LEVEL="${SPARSITY_LEVELS[i]}"
    LR=0.001
    python -m research.flamby.fed_ixi.evaluate_on_holdout \
        --artifact_dir "research/flamby/fed_ixi/sparse_tensor_exchange/large_final_magnitude/hp_sweep_results_sparsity_level=${SPARSITY_LEVEL}/lr_${LR}" \
        --dataset_dir /h/yuchongz/flamby_datasets/fed_ixi/ \
        --eval_write_path research/flamby/fed_ixi/sparse_tensor_exchange/large_final_magnitude/hp_sweep_results_sparsity_level=${SPARSITY_LEVEL}/eval_holdout_result.txt \
        --eval_local_models
done

# EXCHANGE_PERCENTAGES=( 0.1 0.2 0.3 0.4 0.6 0.7 0.8 0.9 )
# EXCHANGE_PERCENTAGES=( 0.5 )
# BEST_LR_VALUE=0.001
# for ((i=0; i<${#EXCHANGE_PERCENTAGES[@]}; i++))
# do
#     EXCHANGE_PERC="${EXCHANGE_PERCENTAGES[i]}"
#     python -m research.flamby.fed_ixi.evaluate_on_holdout \
#         --artifact_dir "research/flamby/fed_ixi/dynamic_layer_exchange/drift_more_normalize/hp_sweep_results_ex_percent=${EXCHANGE_PERC}/lr_${BEST_LR_VALUE}" \
#         --dataset_dir /h/yuchongz/flamby_datasets/fed_ixi/ \
#         --eval_write_path research/flamby/fed_ixi/dynamic_layer_exchange/drift_more_normalize/best_hp_eval_results/eval_holdout_result_${EXCHANGE_PERC}.txt \
#         --eval_local_models
# done
