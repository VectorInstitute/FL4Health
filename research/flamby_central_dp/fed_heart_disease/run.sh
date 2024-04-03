HYPERPARAMETER_NAME=$1
shift 1
HYPERPARAMETER_VALUES=("$@")

clear 
# scancel --me 
# BASE="/scratch/ssd004/scratch/xuejzhao"
rm -rf /scratch/ssd004/scratch/xuejzhao/log_error/fed_heart_disease_central/ /scratch/ssd004/scratch/xuejzhao/log/fed_heart_disease_central/ 

mkdir -p /scratch/ssd004/scratch/xuejzhao/log/fed_heart_disease_central/
mkdir -p /scratch/ssd004/scratch/xuejzhao/log_error/fed_heart_disease_central/

clear
bash ./research/flamby_central_dp/fed_heart_disease/run_hp_sweep.sh \
    research/flamby_central_dp/fed_heart_disease/config.yaml \
    /scratch/ssd004/scratch/xuejzhao/log/fed_heart_disease_central/\
    flamby_datasets/fed_heart_disease/ \
    .venv/ \
    $HYPERPARAMETER_NAME \
    "${HYPERPARAMETER_VALUES[@]}"
exit
sleep 3
squeue --me