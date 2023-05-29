#!/bin/bash

n_clients_to_start=4
config_path="examples/fedprox_example/config.yaml"
dataset_path="examples/datasets/mnist_data/"
server_output_file="examples/fedprox_example/server.out"
client_output_folder="examples/fedprox_example/"


# Start the server, divert the outputs to a server file

echo "Server logging at: ${server_output_file}"

nohup python -m examples.fedprox_example.server --config_path ${config_path} > ${server_output_file} 2>&1 &

# Sleep for 20 seconds to allow the server to come up.
sleep 20

# Start n number of clients and divert the outputs to their own files
for (( i=1; i<=${n_clients_to_start}; i++ ))
do
    client_log_path="${client_output_folder}client_${i}.out"
    echo "Client ${i} logging at: ${client_log_path}"
    nohup python -m examples.fedprox_example.client --dataset_path ${dataset_path} > ${client_log_path} 2>&1 &
done
