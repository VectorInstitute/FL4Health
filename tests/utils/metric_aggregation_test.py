from flwr.common.typing import Metrics

from fl4health.metrics.metric_aggregation import (
    evaluate_metrics_aggregation_fn,
    fit_metrics_aggregation_fn,
    metric_aggregation,
    normalize_metrics,
    uniform_evaluate_metrics_aggregation_fn,
    uniform_metric_aggregation,
)


def test_metric_aggregation() -> None:
    n_clients = 10
    int_metric_counts = [10 for _ in range(n_clients)]
    int_metric_vals: list[Metrics] = []
    for i in range(n_clients):
        metric: Metrics = {"score": 1} if i < 5 else {"score": 2}
        int_metric_vals.append(metric)
    int_metrics: list[tuple[int, Metrics]] = [(count, val) for count, val in zip(int_metric_counts, int_metric_vals)]
    int_total_examples, int_aggregated_metrics = metric_aggregation(int_metrics)
    assert int_total_examples == sum(int_metric_counts)
    gt_int_metrics: Metrics = {"score": 150.0}
    assert int_aggregated_metrics == gt_int_metrics

    float_metric_counts = [20 for _ in range(n_clients)]
    float_metric_vals: list[Metrics] = []
    for i in range(n_clients):
        float_metric: Metrics = {"score": float(i)}
        float_metric_vals.append(float_metric)
    float_metrics = [(count, vals) for (count, vals) in zip(float_metric_counts, float_metric_vals)]
    float_total_samples, float_aggregated_metrics = metric_aggregation(float_metrics)
    assert float_total_samples == sum(float_metric_counts)
    gt_float_metrics: Metrics = {"score": 900.0}
    assert float_aggregated_metrics == gt_float_metrics


def test_normalize_metrics() -> None:
    aggregated_metrics: Metrics = {"score1": 1000.0, "score2": 500.0}
    total_samples = 100

    gt_normalized_metrics: Metrics = {"score1": 10.0, "score2": 5.0}
    normalized_metrics = normalize_metrics(total_samples, aggregated_metrics)
    assert normalized_metrics == gt_normalized_metrics


def test_fit_metrics_aggregation_fn() -> None:
    n_clients = 10
    metric_counts = [10 for _ in range(n_clients)]
    metric_vals: list[Metrics] = []
    for i in range(n_clients):
        metric: Metrics = {"score": float(i)}
        metric_vals.append(metric)
    metric_vals = [{"score": float(i)} for i in range(n_clients)]
    metrics = [(count, vals) for (count, vals) in zip(metric_counts, metric_vals)]
    normalized_metrics = fit_metrics_aggregation_fn(metrics)
    gt_normalized_metric: Metrics = {"score": 4.5}
    assert normalized_metrics == gt_normalized_metric


def test_evaluate_metrics_aggregation_fn() -> None:
    n_clients = 5
    metric_counts = [20 for _ in range(n_clients)]
    metric_vals: list[Metrics] = []
    for i in range(n_clients):
        metric: Metrics = {"score": float(i)}
        metric_vals.append(metric)
    metrics = [(count, vals) for (count, vals) in zip(metric_counts, metric_vals)]
    normalized_metrics = evaluate_metrics_aggregation_fn(metrics)
    gt_normalized_metric: Metrics = {"score": 2.0}
    assert normalized_metrics == gt_normalized_metric


def test_uniform_metric_aggregation() -> None:
    client_sample_counts = [100, 200, 100, 200, 100]
    vals = [5.0, 10.0, 20.0, 10.0, 10.0]
    client_metric_vals: list[tuple[int, Metrics]] = []
    for count, val in zip(client_sample_counts, vals):
        client_metrics: Metrics = {"score": val}
        client_metric_vals.append((count, client_metrics))
    client_count_by_metric, uniform_agg_metrics = uniform_metric_aggregation(client_metric_vals)
    gt_uniform_agg_metrics = {"score": 55.0}
    assert uniform_agg_metrics == gt_uniform_agg_metrics
    assert client_count_by_metric == {"score": 5}


def test_uniform_evaluate_metrics_aggregation_fn() -> None:
    client_sample_counts = [100, 200, 100, 200, 100]
    vals = [5.0, 10.0, 20.0, 10.0, 10.0]
    client_metric_vals: list[tuple[int, Metrics]] = []
    for count, val in zip(client_sample_counts, vals):
        client_metrics: Metrics = {"score": val}
        client_metric_vals.append((count, client_metrics))
    metrics = uniform_evaluate_metrics_aggregation_fn(client_metric_vals)
    gt_metrics = {"score": 11.0}
    assert metrics == gt_metrics
