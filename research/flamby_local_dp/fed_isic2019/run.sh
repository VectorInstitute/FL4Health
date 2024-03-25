clear 
# scancel --me 
rm -rf /scratch/ssd004/scratch/xuejzhao/log_error/fed_isic2019_local/ /scratch/ssd004/scratch/xuejzhao/log/fed_isic2019_local/ 

mkdir -p /scratch/ssd004/scratch/xuejzhao/log_error/fed_isic2019_local/
mkdir -p /scratch/ssd004/scratch/xuejzhao/log/fed_isic2019_local/

research/flamby_local_dp/fed_isic2019/run_hp_sweep.sh \
    research/flamby_local_dp/fed_isic2019/config.yaml \
    /scratch/ssd004/scratch/xuejzhao/log/fed_isic2019_local/ \
    flamby_datasets/fed_isic2019/ \
    .venv/

sleep 5

squeue --me