HYPERPARAMETER_NAME=$1
shift 1
HYPERPARAMETER_VALUES=("$@")

clear
# scancel --me 
rm -rf /scratch/ssd004/scratch/xuejzhao/log_error/fed_ixi_central/ log/fed_ixi_central/ 

mkdir -p /scratch/ssd004/scratch/xuejzhao/log_error/fed_ixi_central/
mkdir -p /scratch/ssd004/scratch/xuejzhao/log/fed_ixi_central/



research/flamby_central_dp/fed_ixi/run_hp_sweep.sh \
    research/flamby_central_dp/fed_ixi/config.yaml \
    /scratch/ssd004/scratch/xuejzhao/log/fed_ixi_central/ \
    flamby_datasets/fed_ixi/ \
    .venv/ $HYPERPARAMETER_NAME "${HYPERPARAMETER_VALUES[@]}"

sleep 5

squeue --me