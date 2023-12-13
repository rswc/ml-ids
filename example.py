from cicids import CICIDS2017
from river.datasets import ImageSegments, SMSSpam

from river import tree
from river import metrics
from river.metrics.base import Metrics
from framework import ExperimentRunner
from metrics import MetricWrapper, FalseDiscoveryRate, FalsePositiveRate, AttackDetectionRate

dataset_mul = CICIDS2017(filename='all_days_noatt_idx=400000_n=400000.csv')
model_mul = tree.HoeffdingTreeClassifier()

WINDOW_SIZE = 8000

bin_metric_params = { 
    'window_size': WINDOW_SIZE, 
    'collapse_label': False, 
    'collapse_classes': ['BENIGN']
}

metrics_mul = Metrics([
    MetricWrapper(
        metric=metrics.GeometricMean(),
        window_size=WINDOW_SIZE,
    ),
    MetricWrapper(
        metrics.GeometricMean()
    ),
    MetricWrapper(
        metric=AttackDetectionRate(),
        collapse_label=False,
        collapse_classes=['BENIGN']
    ),
    MetricWrapper(
        metric=AttackDetectionRate(),
        **bin_metric_params
    ),
    MetricWrapper(
        metric=FalseDiscoveryRate(),
        **bin_metric_params
    ),
    MetricWrapper(
        metric=FalsePositiveRate(),
        **bin_metric_params
    )
])

runner = ExperimentRunner(model_mul, dataset_mul, metrics_mul, "./out", project="test-ml-ids", enable_tracker=False)
runner.run()
