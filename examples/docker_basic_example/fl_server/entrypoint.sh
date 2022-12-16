#!/bin/bash

sleep 10
# start server
python3 -m examples.docker_basic_example.fl_server.server --config_path examples/docker_basic_example/config.yaml
