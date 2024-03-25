clear
# scancel --me 
rm -rf /scratch/ssd004/scratch/xuejzhao/log_error/fed_ixi_distributed/ /scratch/ssd004/scratch/xuejzhao/log/fed_ixi_distributed/ 

mkdir -p /scratch/ssd004/scratch/xuejzhao/log_error/fed_ixi_distributed/
mkdir -p /scratch/ssd004/scratch/xuejzhao/log/fed_ixi_distributed/



research/flamby_distributed_dp/fed_ixi/run_hp_sweep.sh \
    research/flamby_distributed_dp/fed_ixi/config.yaml \
    /scratch/ssd004/scratch/xuejzhao/log/fed_ixi_distributed/ \
    flamby_datasets/fed_ixi/ \
    .venv/

sleep 5

squeue --me