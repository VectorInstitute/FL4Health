# Federated LoRA Instruction-tuning of Large Language Model Example
This example provides LoRA instruction-tuning of LLaMA-3B in a federated learning setting over the Alpaca GPT4 dataset.
The FL server expects two clients to be spun up (i.e. it will wait until two clients report in before starting training).
Each client has a partition of the Alpaca dataset and can utilize one or multiple available GPUs, with the number of GPUs
set in the configuration file. Depending on the desired level of parallelism, the client can select an appropriate DeepSpeed
ZeRO optimization level. These configurations are stored in the `./training_script/zero_config` directory and are adapted
from LLaVA-NeXT repository (https://github.com/LLaVA-VL/LLaVA-NeXT/tree/main/scripts). Available configurations are:

- `zero2.json`: Gradients and optimizer states are sharded across GPUs
- `zero2_offload.json`: Gradients and optimizer states are sharded across GPUs and offloaded to CPU to reduce GPU memory usage
- `zero3.json`: Gradients, optimizer states, and model parameters are sharded across GPUs
- `zero3_offload.json`: Gradients, optimizer states, and model parameters are sharded across GPUs and offloaded to CPU to reduce
GPU memory usage
- `zero3pp.json`: Has an improvement over ZeRO-3 by reducing the memory overhead through hierarchical partitioning and quantization

**Note**: For more information on DeepSpeed ZeRO configurations, please refer to the DeepSpeed documentation
(https://www.deepspeed.ai/docs/config-json/).

The server utilizes Federated Averaging as its optimization strategy. Before initiating the training process, it waits for additional
GPUs to come online based on the number of participating clients.

## Running the Example
In order to run the example, first ensure you have [installed the dependencies in your virtual environment according to the CONTRIBUTING](/CONTRIBUTING.MD#development-requirements) with addition of `llm` group dependencies.

Then run the following command to kick off the training process:

```bash
training_script/run_fl_cluster.sh \
    server_socket_address \
    path_to_config.yaml \
    path_to_folder_for_artifacts\
    path_to_folder_for_distributed_server_logs \
    path_to_folder_for_distributed_client_logs \
    path_to_desired_venv
```

Where:
- `path_to_config.yaml` is the path to the configuration file for the approach.
- `path_to_folder_for_artifacts/` is the path to the folder where the artifacts of the hyperparameter sweep will be saved.
- `path_to_folder_for_distributed_server_logs/` is the path to the folder where the server logs are saved.
- `path_to_folder_for_distributed_client_logs/` is the path to the folder where the client logs are saved.
- `path_to_desired_venv/` is the path to the virtual environment.
