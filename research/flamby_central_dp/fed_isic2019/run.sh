HYPERPARAMETER_NAME=$1
shift 1
HYPERPARAMETER_VALUES=("$@")

clear
# scancel --me 
rm -rf /scratch/ssd004/scratch/your_usrname_here/log_error/fed_isic2019_central/ log/fed_isic2019_central/ 

mkdir -p /scratch/ssd004/scratch/your_usrname_here/log_error/fed_isic2019_central/
mkdir -p /scratch/ssd004/scratch/your_usrname_here/log/fed_isic2019_central/



research/flamby_central_dp/fed_isic2019/run_hp_sweep.sh \
    research/flamby_central_dp/fed_isic2019/config.yaml \
    /scratch/ssd004/scratch/your_usrname_here/log/fed_isic2019_central/ \
    flamby_datasets/fed_isic2019/ \
    .venv/ $HYPERPARAMETER_NAME "${HYPERPARAMETER_VALUES[@]}"

sleep 5

squeue --me