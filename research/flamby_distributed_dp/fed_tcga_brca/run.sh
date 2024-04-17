HYPERPARAMETER_NAME=$1
shift 1
HYPERPARAMETER_VALUES=("$@")

clear 

# scancel --me 
rm -rf /scratch/ssd004/scratch/xuejzhao/log_error/fed_tcga_brca_distributed/ /scratch/ssd004/scratch/xuejzhao/log/fed_tcga_brca_distributed/ 

mkdir -p /scratch/ssd004/scratch/xuejzhao/log/fed_tcga_brca_distributed/
mkdir -p /scratch/ssd004/scratch/xuejzhao/log_error/fed_tcga_brca_distributed/

./research/flamby_distributed_dp/fed_tcga_brca/run_hp_sweep.sh \
    research/flamby_distributed_dp/fed_tcga_brca/config.yaml \
    /scratch/ssd004/scratch/xuejzhao/log/fed_tcga_brca_distributed/\
    flamby_datasets/fed_tcga_brca/ \
    .venv/ \
    $HYPERPARAMETER_NAME \
    "${HYPERPARAMETER_VALUES[@]}"

sleep 3
squeue --me