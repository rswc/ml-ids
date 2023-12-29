from river.metrics.base import Metrics

def extract_metric_name(metric) -> str:
    """Extract name of the metric using method compliant with `MetricWrapper` based on `Metric` __repr__"""
    return str(metric).split(':')[0]

def get_metrics_dict(metrics: Metrics) -> dict:
    """Current values of the metrics, as a Python dict"""
    return dict(zip([extract_metric_name(metric) for metric in metrics], metrics.get()))
