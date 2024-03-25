mkdir -p log/server_logs/
mkdir -p log/client_logs/

rm -r log_distributed/

bash ./fed_ixi/run_fl_cluster.sh \
    8111 \
    research/flamby_local_dp/fed_ixi/config.yaml \
    log/server_logs/ \
    log/client_logs/ \
    .venv/