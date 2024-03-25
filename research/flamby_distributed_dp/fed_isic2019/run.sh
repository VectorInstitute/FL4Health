clear 
# scancel --me 
rm -rf /scratch/ssd004/scratch/xuejzhao/log_error/fed_isic2019_distributed/ /scratch/ssd004/scratch/xuejzhao/log/fed_isic2019_distributed/ 

mkdir -p /scratch/ssd004/scratch/xuejzhao/log_error/fed_isic2019_distributed/
mkdir -p /scratch/ssd004/scratch/xuejzhao/log/fed_isic2019_distributed/

research/flamby_distributed_dp/fed_isic2019/run_hp_sweep.sh \
    research/flamby_distributed_dp/fed_isic2019/config.yaml \
    /scratch/ssd004/scratch/xuejzhao/log/fed_isic2019_distributed/ \
    flamby_datasets/fed_isic2019/ \
    .venv/

sleep 5

squeue --me