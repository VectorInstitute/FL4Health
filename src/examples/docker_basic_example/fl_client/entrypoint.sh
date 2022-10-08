#!/bin/bash

# give enough time for fl-server to finish initializing
sleep 60
# start client
python3 src/client.py
