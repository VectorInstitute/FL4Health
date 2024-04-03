HYPERPARAMETER_NAME=$1
shift 1
HYPERPARAMETER_VALUES=("$@")

clear 
# scancel --me 

rm -rf /scratch/ssd004/scratch/xuejzhao/log_error/fed_heart_disease_local/ /scratch/ssd004/scratch/xuejzhao/log/fed_heart_disease_local/ 

mkdir -p /scratch/ssd004/scratch/xuejzhao/log/fed_heart_disease_local/
mkdir -p /scratch/ssd004/scratch/xuejzhao/log_error/fed_heart_disease_local/

./research/flamby_local_dp/fed_heart_disease/run_hp_sweep.sh \
    research/flamby_local_dp/fed_heart_disease/config.yaml \
    /scratch/ssd004/scratch/xuejzhao/log/fed_heart_disease_local/\
    flamby_datasets/fed_heart_disease/ \
    .venv/ \
    $HYPERPARAMETER_NAME \
    "${HYPERPARAMETER_VALUES[@]}"

sleep 3
squeue --me