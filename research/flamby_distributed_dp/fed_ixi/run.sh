clear
# scancel --me 
rm -rf log_error/fed_ixi/ log/fed_ixi/ 

mkdir -p log_error/fed_ixi/
mkdir -p log/fed_ixi/



research/flamby_distributed_dp/fed_ixi/run_hp_sweep.sh \
    research/flamby_distributed_dp/fed_ixi/config.yaml \
    log/fed_ixi/ \
    flamby_datasets/fed_ixi/ \
    .venv/

sleep 5

squeue --me