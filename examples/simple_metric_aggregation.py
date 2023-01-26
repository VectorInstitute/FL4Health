from typing import List, Tuple

from flwr.common.typing import Metrics


def metric_aggregation(all_client_metrics: List[Tuple[int, Metrics]]) -> Tuple[int, Metrics]:
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
                aggregated_metrics[metric_name] = current_metric_value + metric_value
            elif isinstance(metric_value, int):
                current_metric_value = aggregated_metrics.get(metric_name, 0)
                assert isinstance(current_metric_value, int)
                aggregated_metrics[metric_name] = current_metric_value + metric_value
            else:
                raise ValueError("Metric type is not supported")
    return total_examples, aggregated_metrics


def normalize_metrics(total_examples: int, aggregated_metrics: Metrics) -> Metrics:
    # Normalize all metric values by the total count of examples seen.
    normalized_metrics: Metrics = {}
    for metric_name, metric_value in aggregated_metrics.items():
        if isinstance(metric_value, float) or isinstance(metric_value, int):
            normalized_metrics[metric_name] = metric_value / total_examples
    return normalized_metrics
