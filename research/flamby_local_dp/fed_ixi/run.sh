clear
scancel --me 
rm -rf log_error/fed_ixi_central/ log/fed_ixi_central/ 

mkdir -p log_error/fed_ixi_central/
mkdir -p log/fed_ixi_central/



research/flamby_central_dp/fed_ixi/run_hp_sweep.sh \
    research/flamby_central_dp/fed_ixi/config.yaml \
    log/fed_ixi_central/ \
    flamby_datasets/fed_ixi/ \
    .venv/

sleep 5

squeue --me