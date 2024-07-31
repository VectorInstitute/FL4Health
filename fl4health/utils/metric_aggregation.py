from collections import defaultdict
from typing import DefaultDict, List, Tuple

from flwr.common.typing import Metrics


def uniform_metric_aggregation(
    all_client_metrics: List[Tuple[int, Metrics]],
) -> Tuple[DefaultDict[str, int], Metrics]:
    """
    Function that aggregates client metrics and divides by the number of clients that contributed to metric.

    Args:
        all_client_metrics (List[Tuple[int, Metrics]]): A list of tuples with the
            sample counts and metrics for each client.

    Returns:
        Tuple[DefaultDict[str, int], Metrics]: Client counts per metric and the uniformly aggregated metrics.
    """
    aggregated_metrics: Metrics = {}
    total_client_count_by_metric: DefaultDict[str, int] = defaultdict(int)
    # Run through all of the metrics
    for _, client_metrics in all_client_metrics:
        for metric_name, metric_value in client_metrics.items():
            if isinstance(metric_value, float):
                current_metric_value = aggregated_metrics.get(metric_name, 0.0)
                assert isinstance(current_metric_value, float)
                aggregated_metrics[metric_name] = current_metric_value + metric_value
                total_client_count_by_metric[metric_name] += 1
            elif isinstance(metric_value, int):
                current_metric_value = aggregated_metrics.get(metric_name, 0)
                assert isinstance(current_metric_value, int)
                aggregated_metrics[metric_name] = current_metric_value + metric_value
                total_client_count_by_metric[metric_name] += 1
            else:
                raise ValueError("Metric type is not supported")
    # Compute average of each metric by dividing by number of clients contributing
    uniform_normalize_metrics(total_client_count_by_metric, aggregated_metrics)
    return total_client_count_by_metric, aggregated_metrics


def metric_aggregation(
    all_client_metrics: List[Tuple[int, Metrics]],
) -> Tuple[int, Metrics]:
    """
    Function that computes a weighted aggregation of metrics normalized by the total number of samples.

    Args:
        all_client_metrics (List[Tuple[int, Metrics]]): A list of tuples with the
            sample counts and metrics for each client.

    Returns:
        Tuple[int, Metrics]: The total number of examples along with aggregated metrics.
    """
    aggregated_metrics: Metrics = {}
    total_examples = 0
    # Run through all of the metrics
    for num_examples_on_client, client_metrics in all_client_metrics:
        total_examples += num_examples_on_client
        for metric_name, metric_value in client_metrics.items():
            # Here we assume each metric is normalized by the number of examples on the client. So we scale up to
            # get the "raw" value
            if isinstance(metric_value, float):
                current_metric_value = aggregated_metrics.get(metric_name, 0.0)
                assert isinstance(current_metric_value, float)
                aggregated_metrics[metric_name] = current_metric_value + num_examples_on_client * metric_value
            elif isinstance(metric_value, int):
                current_metric_value = aggregated_metrics.get(metric_name, 0)
                assert isinstance(current_metric_value, int)
                aggregated_metrics[metric_name] = current_metric_value + num_examples_on_client * metric_value
            else:
                raise ValueError("Metric type is not supported")
    return total_examples, aggregated_metrics


def normalize_metrics(total_examples: int, aggregated_metrics: Metrics) -> Metrics:
    """
    Function that normalizes metrics by provided sample count.

    Args:
        total_examples (int): The total number of samples across all client datasets.
        aggregated_metrics (Metrics): Metrics that have been aggregated across clients.

    Returns:
        Metrics: The metrics normalized by total_examples.
    """
    # Normalize all metric values by the total count of examples seen.
    normalized_metrics: Metrics = {}
    for metric_name, metric_value in aggregated_metrics.items():
        if isinstance(metric_value, float) or isinstance(metric_value, int):
            normalized_metrics[metric_name] = metric_value / total_examples
    return normalized_metrics


def uniform_normalize_metrics(
    total_client_count_by_metric: DefaultDict[str, int], aggregated_metrics: Metrics
) -> Metrics:
    """
    Function that normalizes metrics based on how many clients contributed to the metric.

    Args:
        total_client_count_by_metric (DefaultDict[str, int]): The count of clients that contributed to each metric.
        aggregated_metrics (Metrics): Metrics that have been aggregated across clients.

    Returns:
        Metrics: The normalized metrics.
    """
    # Normalize all metric values by the total count of clients that contributed to the metric.
    normalized_metrics: Metrics = {}
    for metric_name, metric_value in aggregated_metrics.items():
        if isinstance(metric_value, float) or isinstance(metric_value, int):
            normalized_metrics[metric_name] = metric_value / total_client_count_by_metric[metric_name]
    return normalized_metrics


def fit_metrics_aggregation_fn(
    all_client_metrics: List[Tuple[int, Metrics]],
) -> Metrics:
    """
    Function for fit that computes a weighted aggregation of the client metrics
    and normalizes by the total number of samples.

    Args:
        all_client_metrics (List[Tuple[int, Metrics]]): A list of tuples with the
            sample counts and metrics for each client.

    Returns:
        Metrics: The aggregated normalized metrics.
    """
    # This function is run by the server to aggregate metrics returned by each clients fit function
    # NOTE: The first value of the tuple is number of examples for FedAvg
    total_examples, aggregated_metrics = metric_aggregation(all_client_metrics)
    return normalize_metrics(total_examples, aggregated_metrics)


def evaluate_metrics_aggregation_fn(
    all_client_metrics: List[Tuple[int, Metrics]],
) -> Metrics:
    """
    Function for evaluate that computes a weighted aggregation of the client metrics
    and normalizes by the total number of samples.

    Args:
        all_client_metrics (List[Tuple[int, Metrics]]): A list of tuples with the
            sample counts and metrics for each client.

    Returns:
        Metrics: The aggregated normalized metrics.
    """
    # This function is run by the server to aggregate metrics returned by each clients evaluate function
    # NOTE: The first value of the tuple is number of examples for FedAvg
    total_examples, aggregated_metrics = metric_aggregation(all_client_metrics)
    return normalize_metrics(total_examples, aggregated_metrics)


def uniform_evaluate_metrics_aggregation_fn(
    all_client_metrics: List[Tuple[int, Metrics]],
) -> Metrics:
    """
    Function for evaluate that computes aggregation of the client metrics and normalizes by the number
    of clients that contributed to the metric.

    Args:
        all_client_metrics (List[Tuple[int, Metrics]]): A list of tuples with the
            sample counts and metrics for each client.

    Returns:
        Metrics: The aggregated normalized metrics.
    """
    # This function is run by the server to aggregate metrics returned by each clients evaluate function
    # NOTE: The first value of the tuple is number of examples for FedAvg, but it is not used here.
    total_client_count_by_metric, aggregated_metrics = uniform_metric_aggregation(all_client_metrics)
    return uniform_normalize_metrics(total_client_count_by_metric, aggregated_metrics)
