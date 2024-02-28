clear && scancel --me && rm -rf log_error/fed_heart_disease/ log/fed_heart_disease/ 

mkdir -p log/fed_heart_disease/
mkdir -p log_error/fed_heart_disease/

./research/flamby_distributed_dp/fed_heart_disease/run_hp_sweep.sh \
    research/flamby_distributed_dp/fed_heart_disease/config.yaml \
    log/fed_heart_disease/\
    flamby_datasets/fed_heart_disease/ \
    .venv/

sleep 3
squeue --me