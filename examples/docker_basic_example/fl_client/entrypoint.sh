#!/bin/bash

# give enough time for fl-server to finish initializing
sleep 60
# start client
python3 -m src.examples.docker_basic_example.fl_client.client --dataset_path src/datasets/cifar_data
