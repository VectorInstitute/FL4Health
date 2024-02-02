mkdir -p log/heart/
mkdir -p log_error/

./research/flamby/fed_heart_disease/fedavg/run_hp_sweep.sh \
   research/flamby/fed_heart_disease/fedavg/config.yaml \
   log/heart/ \
   flamby_datasets/fed_heart_disease/ \
   .venv/