clear 
scancel --me 

rm -rf log_error/fed_heart_disease_local/ log/fed_heart_disease_local/ 

mkdir -p log/fed_heart_disease_local/
mkdir -p log_error/fed_heart_disease_local/

./research/flamby_local_dp/fed_heart_disease/run_hp_sweep.sh \
    research/flamby_local_dp/fed_heart_disease/config.yaml \
    log/fed_heart_disease_local/\
    flamby_datasets/fed_heart_disease/ \
    .venv/

sleep 3
squeue --me