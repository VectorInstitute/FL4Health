# File name mapped to tuples of name appearing on the graph, keys for the mean, and keys for the std dev
# NOTE: that only some methods with both server and local models have multiple mean and std dev keys
fed_isic_file_names_to_info: list[tuple[str, str, tuple[list[str], list[str]]]] = [
    (
        "central_eval_performance.txt",
        "Central",
        (
            ["avg_server_model_avg_across_clients"],
            ["std_server_model_avg_across_clients"],
        ),
    ),
    (
        "client_0_eval_performance.txt",
        "Local 0",
        (
            ["avg_server_model_avg_across_clients"],
            ["std_server_model_avg_across_clients"],
        ),
    ),
    (
        "client_1_eval_performance.txt",
        "Local 1",
        (
            ["avg_server_model_avg_across_clients"],
            ["std_server_model_avg_across_clients"],
        ),
    ),
    (
        "client_2_eval_performance.txt",
        "Local 2",
        (
            ["avg_server_model_avg_across_clients"],
            ["std_server_model_avg_across_clients"],
        ),
    ),
    (
        "client_3_eval_performance.txt",
        "Local 3",
        (
            ["avg_server_model_avg_across_clients"],
            ["std_server_model_avg_across_clients"],
        ),
    ),
    (
        "client_4_eval_performance.txt",
        "Local 4",
        (
            ["avg_server_model_avg_across_clients"],
            ["std_server_model_avg_across_clients"],
        ),
    ),
    (
        "client_5_eval_performance.txt",
        "Local 5",
        (
            ["avg_server_model_avg_across_clients"],
            ["std_server_model_avg_across_clients"],
        ),
    ),
    (
        "fedavg_eval_performance.txt",
        "FedAvg",
        (
            ["avg_server_model_avg_across_clients", "avg_local_model_avg_across_clients"],
            ["std_server_model_avg_across_clients", "std_local_model_avg_across_clients"],
        ),
    ),
    (
        "fedadam_eval_performance.txt",
        "FedAdam",
        (
            ["avg_server_model_avg_across_clients", "avg_local_model_avg_across_clients"],
            ["std_server_model_avg_across_clients", "std_local_model_avg_across_clients"],
        ),
    ),
    (
        "fedprox_eval_performance.txt",
        "FedProx",
        (
            ["avg_server_model_avg_across_clients", "avg_local_model_avg_across_clients"],
            ["std_server_model_avg_across_clients", "std_local_model_avg_across_clients"],
        ),
    ),
    (
        "scaffold_eval_performance.txt",
        "SCAFFOLD",
        (
            ["avg_server_model_avg_across_clients", "avg_local_model_avg_across_clients"],
            ["std_server_model_avg_across_clients", "std_local_model_avg_across_clients"],
        ),
    ),
    (
        "fenda_eval_performance_001.txt",
        "FENDA",
        (
            ["avg_local_model_avg_across_clients"],
            ["std_local_model_avg_across_clients"],
        ),
    ),
    (
        "apfl_eval_performance.txt",
        "APFL",
        (
            ["avg_local_model_avg_across_clients"],
            ["std_local_model_avg_across_clients"],
        ),
    ),
]

# File name mapped to tuples of name appearing on the graph, keys for the mean, and keys for the std dev
# NOTE: that only some methods with both server and local models have multiple mean and std dev keys
fed_heart_disease_file_names_to_info: list[tuple[str, str, tuple[list[str], list[str]]]] = [
    (
        "central_eval_performance_small_model.txt",
        "Central_S",
        (
            ["avg_server_model_avg_across_clients"],
            ["std_server_model_avg_across_clients"],
        ),
    ),
    (
        "central_eval_performance_big_model.txt",
        "Central_L",
        (
            ["avg_server_model_avg_across_clients"],
            ["std_server_model_avg_across_clients"],
        ),
    ),
    (
        "client_0_eval_performance_small_model.txt",
        "Local 0_S",
        (
            ["avg_server_model_avg_across_clients"],
            ["std_server_model_avg_across_clients"],
        ),
    ),
    (
        "client_0_eval_performance_big_model.txt",
        "Local 0_L",
        (
            ["avg_server_model_avg_across_clients"],
            ["std_server_model_avg_across_clients"],
        ),
    ),
    (
        "client_1_eval_performance_small_model.txt",
        "Local 1_S",
        (
            ["avg_server_model_avg_across_clients"],
            ["std_server_model_avg_across_clients"],
        ),
    ),
    (
        "client_1_eval_performance_big_model.txt",
        "Local 1_L",
        (
            ["avg_server_model_avg_across_clients"],
            ["std_server_model_avg_across_clients"],
        ),
    ),
    (
        "client_2_eval_performance_small_model.txt",
        "Local 2_S",
        (
            ["avg_server_model_avg_across_clients"],
            ["std_server_model_avg_across_clients"],
        ),
    ),
    (
        "client_2_eval_performance_big_model.txt",
        "Local 2_L",
        (
            ["avg_server_model_avg_across_clients"],
            ["std_server_model_avg_across_clients"],
        ),
    ),
    (
        "client_3_eval_performance_small_model.txt",
        "Local 3_S",
        (
            ["avg_server_model_avg_across_clients"],
            ["std_server_model_avg_across_clients"],
        ),
    ),
    (
        "client_3_eval_performance_big_model.txt",
        "Local 3_L",
        (
            ["avg_server_model_avg_across_clients"],
            ["std_server_model_avg_across_clients"],
        ),
    ),
    (
        "fedavg_eval_performance_small_model.txt",
        "FedAvg_S",
        (
            ["avg_server_model_avg_across_clients", "avg_local_model_avg_across_clients"],
            ["std_server_model_avg_across_clients", "std_local_model_avg_across_clients"],
        ),
    ),
    (
        "fedavg_eval_performance_big_model.txt",
        "FedAvg_L",
        (
            ["avg_server_model_avg_across_clients", "avg_local_model_avg_across_clients"],
            ["std_server_model_avg_across_clients", "std_local_model_avg_across_clients"],
        ),
    ),
    (
        "fedadam_eval_performance_small_model.txt",
        "FedAdam_S",
        (
            ["avg_server_model_avg_across_clients", "avg_local_model_avg_across_clients"],
            ["std_server_model_avg_across_clients", "std_local_model_avg_across_clients"],
        ),
    ),
    (
        "fedadam_eval_performance_big_model.txt",
        "FedAdam_L",
        (
            ["avg_server_model_avg_across_clients", "avg_local_model_avg_across_clients"],
            ["std_server_model_avg_across_clients", "std_local_model_avg_across_clients"],
        ),
    ),
    (
        "fedprox_eval_performance_small_model.txt",
        "FedProx_S",
        (
            ["avg_server_model_avg_across_clients", "avg_local_model_avg_across_clients"],
            ["std_server_model_avg_across_clients", "std_local_model_avg_across_clients"],
        ),
    ),
    (
        "fedprox_eval_performance_big_model.txt",
        "FedProx_L",
        (
            ["avg_server_model_avg_across_clients", "avg_local_model_avg_across_clients"],
            ["std_server_model_avg_across_clients", "std_local_model_avg_across_clients"],
        ),
    ),
    (
        "scaffold_eval_performance_small_model.txt",
        "SCAFFOLD_S",
        (
            ["avg_server_model_avg_across_clients", "avg_local_model_avg_across_clients"],
            ["std_server_model_avg_across_clients", "std_local_model_avg_across_clients"],
        ),
    ),
    (
        "scaffold_eval_performance_big_model.txt",
        "SCAFFOLD_L",
        (
            ["avg_server_model_avg_across_clients", "avg_local_model_avg_across_clients"],
            ["std_server_model_avg_across_clients", "std_local_model_avg_across_clients"],
        ),
    ),
    (
        "fenda_eval_performance_big_model.txt",
        "FENDA",
        (
            ["avg_local_model_avg_across_clients"],
            ["std_local_model_avg_across_clients"],
        ),
    ),
    (
        "apfl_eval_performance_big_model.txt",
        "APFL",
        (
            ["avg_local_model_avg_across_clients"],
            ["std_local_model_avg_across_clients"],
        ),
    ),
]

# File name mapped to tuples of name appearing on the graph, keys for the mean, and keys for the std dev
# NOTE: that only some methods with both server and local models have multiple mean and std dev keys
fed_ixi_file_names_to_info: list[tuple[str, str, tuple[list[str], list[str]]]] = [
    (
        "central_eval_performance.txt",
        "Central",
        (
            ["avg_server_model_avg_across_clients"],
            ["std_server_model_avg_across_clients"],
        ),
    ),
    (
        "client_0_eval_performance.txt",
        "Local 0",
        (
            ["avg_server_model_avg_across_clients"],
            ["std_server_model_avg_across_clients"],
        ),
    ),
    (
        "client_1_eval_performance.txt",
        "Local 1",
        (
            ["avg_server_model_avg_across_clients"],
            ["std_server_model_avg_across_clients"],
        ),
    ),
    (
        "client_2_eval_performance.txt",
        "Local 2",
        (
            ["avg_server_model_avg_across_clients"],
            ["std_server_model_avg_across_clients"],
        ),
    ),
    (
        "fedavg_eval_performance.txt",
        "FedAvg",
        (
            ["avg_server_model_avg_across_clients", "avg_local_model_avg_across_clients"],
            ["std_server_model_avg_across_clients", "std_local_model_avg_across_clients"],
        ),
    ),
    (
        "fedadam_eval_performance.txt",
        "FedAdam",
        (
            ["avg_server_model_avg_across_clients", "avg_local_model_avg_across_clients"],
            ["std_server_model_avg_across_clients", "std_local_model_avg_across_clients"],
        ),
    ),
    (
        "fedprox_eval_performance.txt",
        "FedProx",
        (
            ["avg_server_model_avg_across_clients", "avg_local_model_avg_across_clients"],
            ["std_server_model_avg_across_clients", "std_local_model_avg_across_clients"],
        ),
    ),
    (
        "scaffold_eval_performance.txt",
        "SCAFFOLD",
        (
            ["avg_server_model_avg_across_clients", "avg_local_model_avg_across_clients"],
            ["std_server_model_avg_across_clients", "std_local_model_avg_across_clients"],
        ),
    ),
    (
        "fenda_eval_performance.txt",
        "FENDA",
        (
            ["avg_local_model_avg_across_clients"],
            ["std_local_model_avg_across_clients"],
        ),
    ),
    (
        "apfl_eval_performance.txt",
        "APFL",
        (
            ["avg_local_model_avg_across_clients"],
            ["std_local_model_avg_across_clients"],
        ),
    ),
]
