clear 
# scancel --me 
rm -rf log_error/fed_heart_disease_central/ log/fed_heart_disease_central/ 

mkdir -p log/fed_heart_disease_central/
mkdir -p log_error/fed_heart_disease_central/

./research/flamby_central_dp/fed_heart_disease/run_hp_sweep.sh \
    research/flamby_central_dp/fed_heart_disease/config.yaml \
    log/fed_heart_disease_central/\
    flamby_datasets/fed_heart_disease/ \
    .venv/

sleep 3
squeue --me