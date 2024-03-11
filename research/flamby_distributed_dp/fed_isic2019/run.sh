clear 
# scancel --me 
rm -rf log_error/fed_isic2019/ log/fed_isic2019/ 

mkdir -p log_error/fed_isic2019/
mkdir -p log/fed_isic2019/

research/flamby_distributed_dp/fed_isic2019/run_hp_sweep.sh \
    research/flamby_distributed_dp/fed_isic2019/config.yaml \
    log/fed_isic2019/ \
    flamby_datasets/fed_isic2019/ \
    .venv/

sleep 5

squeue --me