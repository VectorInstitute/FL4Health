clear 
# scancel --me 
rm -rf /scratch/ssd004/scratch/xuejzhao/log_error/fed_heart_disease_distributed/ /scratch/ssd004/scratch/xuejzhao/log/fed_heart_disease_distributed/ 

mkdir -p /scratch/ssd004/scratch/xuejzhao/log/fed_heart_disease_distributed/
mkdir -p /scratch/ssd004/scratch/xuejzhao/log_error/fed_heart_disease_distributed/

./research/flamby_distributed_dp/fed_heart_disease/run_hp_sweep.sh \
    research/flamby_distributed_dp/fed_heart_disease/config.yaml \
    /scratch/ssd004/scratch/xuejzhao/log/fed_heart_disease_distributed/\
    flamby_datasets/fed_heart_disease/ \
    .venv/

sleep 3
squeue --me