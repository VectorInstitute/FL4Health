HYPERPARAMETER_NAME=$1
shift 1
HYPERPARAMETER_VALUES=("$@")

clear 

# scancel --me 
rm -rf /scratch/ssd004/scratch/your_usrname_here/log_error/fed_heart_disease_distributed/ /scratch/ssd004/scratch/your_usrname_here/log/fed_heart_disease_distributed/ 

mkdir -p /scratch/ssd004/scratch/your_usrname_here/log/fed_heart_disease_distributed/
mkdir -p /scratch/ssd004/scratch/your_usrname_here/log_error/fed_heart_disease_distributed/

./research/flamby_distributed_dp/fed_heart_disease/run_hp_sweep.sh \
    research/flamby_distributed_dp/fed_heart_disease/config.yaml \
    /scratch/ssd004/scratch/your_usrname_here/log/fed_heart_disease_distributed/\
    flamby_datasets/fed_heart_disease/ \
    .venv/ \
    $HYPERPARAMETER_NAME \
    "${HYPERPARAMETER_VALUES[@]}"

sleep 3
squeue --me