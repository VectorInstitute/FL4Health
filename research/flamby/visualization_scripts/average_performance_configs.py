from typing import List, Tuple

# File name mapped to tuples of name appearing on the graph, keys for the mean, and keys for the std dev
# NOTE: that only some methods with both server and local models have multiple mean and std dev keys
fed_isic_file_names_to_info: List[Tuple[str, str, Tuple[List[str], List[str]]]] = [
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
        "fedprox_eval_performance.txt",
        "FedProx",
        (
            ["avg_server_model_avg_across_clients", "avg_local_model_avg_across_clients"],
            ["std_server_model_avg_across_clients", "std_local_model_avg_across_clients"],
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
        "fenda_eval_performance_001.txt",
        "FENDA",
        (
            ["avg_local_model_avg_across_clients"],
            ["std_local_model_avg_across_clients"],
        ),
    ),
]
