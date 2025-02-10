# About

This example demonstrates distributed differential privacy, with secure aggregation (no drop out setting) and distributed discrete Gaussian (DDGauss). This example also uses the mini-client approach which is experimental.

# How to Run 

Adjust the FL and privacy settings is `config.yaml`, and then run with `run.sh` (some changes might be necessary depending on your project setup).

# Logs

Flower logs to `examples/secure_aggregation_example/log`. Metric of clients and the server log to `examples/secure_aggregation_example/metrics`. Temporary models are stored in `examples/secure_aggregation_example/temp` which can be removed after FL completes.

# Warning 

1. Beware of conflicting ports and orphaned ports. You can kill unwanted processes given their PID stored at `examples/secure_aggregation_example/log/running_pid.txt`. 
2. This example needs to be updated if the client and server classes gets updated.
3. Some path configuration can be edited in `client.py` and `server.py`
