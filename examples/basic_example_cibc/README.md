# README

The biggest issue was that the features were not scaled. Once scaled, learning
can happen an I'm reporting 0.9673 acc on the test set after just 1 round of federation:

```sh
INFO :      Received: evaluate message cec5b2b9-d26e-46f6-9c9d-18873240ec46
INFO :      Client Validation Losses:
INFO :           checkpoint: 0.1501234769821167
INFO :      Client Validation Metrics:
INFO :           val - prediction - accuracy: 0.967875
INFO :      Client Testing Losses:
INFO :           checkpoint: 0.14720943570137024
INFO :      Client Testing Metrics:
INFO :           test - prediction - accuracy: 0.9673
INFO :      No Post-aggregation checkpoint specified. Skipping.
INFO :      Sent reply
INFO :
INFO :      Received: reconnect message 565ec567-1588-4c79-97a3-88d777a56357
INFO :      Disconnect and shut down
```

## Creating Train/Test Datasets

First create the train/test df using `create_test_set.py`

```sh
python examples/basic_example_cibc/create_test_set.py
```

Note: this script handles the splitting as well as the scaling.

## Running the FL task

### Starting the server

```sh
python examples/basic_example_cibc/client.py --dataset_path examples/basic_example_cibc/
```

### Starting the client

```sh
python examples/basic_example_cibc/client.py --dataset_path examples/basic_example_cibc/
```
