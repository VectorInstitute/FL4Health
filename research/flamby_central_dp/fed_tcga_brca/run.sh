HYPERPARAMETER_NAME=$1
shift 1
HYPERPARAMETER_VALUES=("$@")

clear 
# scancel --me 
# BASE="/scratch/ssd004/scratch/xuejzhao"
rm -rf /scratch/ssd004/scratch/xuejzhao/log_error/fed_tcga_brca/ /scratch/ssd004/scratch/xuejzhao/log/fed_tcga_brca_central/ 

mkdir -p /scratch/ssd004/scratch/xuejzhao/log/fed_tcga_brca_central/
mkdir -p /scratch/ssd004/scratch/xuejzhao/log_error/fed_tcga_brca_central/

clear
bash ./research/flamby_central_dp/fed_tcga_brca/run_hp_sweep.sh \
    research/flamby_central_dp/fed_tcga_brca/config.yaml \
    /scratch/ssd004/scratch/xuejzhao/log/fed_tcga_brca_central/\
    flamby_datasets/fed_tcga_brca/ \
    .venv/ \
    $HYPERPARAMETER_NAME \
    "${HYPERPARAMETER_VALUES[@]}"
exit
sleep 3
squeue --me