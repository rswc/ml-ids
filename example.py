from cicids import CICIDS2017
from river.datasets import ImageSegments, SMSSpam

from river import tree
from river import metrics
from river.metrics.base import Metrics
from framework import ExperimentRunner
from metrics import MetricWrapper, FalseDiscoveryRate, FalsePositiveRate, AttackDetectionRate


dataset_mul = CICIDS2017(filename='test_ready.csv')
#dataset_bin = SMSSpam()

model_mul = tree.HoeffdingTreeClassifier()
#model_bin = tree.HoeffdingTreeClassifier()

metrics_mul = Metrics([
    MetricWrapper(
        metric=metrics.GeometricMean(),
        window_size=8000,
    ),
    MetricWrapper(
        metrics.GeometricMean()
    ),
    MetricWrapper(
        metric=AttackDetectionRate(),
        collapse_label=True,
        collapse_classes=['BENIGN']
    ),
    MetricWrapper(
        metric=AttackDetectionRate(),
        window_size=8000,
        collapse_label=True,
        collapse_classes=['BENIGN']
    ),
    MetricWrapper(
        metric=FalseDiscoveryRate(),
        window_size=8000,
        collapse_label=True,
        collapse_classes=['BENIGN']
    ),
    MetricWrapper(
        metric=FalsePositiveRate(),
        window_size=8000,
        collapse_label=True,
        collapse_classes=['BENIGN']
    )
])

metrics_bin = Metrics([
    metrics.F1(),
    MetricWrapper(
        metric=AttackDetectionRate(),
        window_size=8000,
    ),
    MetricWrapper(
        metric=FalsePositiveRate(),
        window_size=8000,
    ),
    MetricWrapper(
        metric=FalseDiscoveryRate(),
        window_size=8000,
    ),
])

#runner_bin = ExperimentRunner(model_bin, dataset_bin, metrics_bin, "./out", project="ml-ids", enable_tracker=True)
runner_mul = ExperimentRunner(model_mul, dataset_mul, metrics_mul, "./out", project="ml-ids", enable_tracker=True)

runner_mul.run()
#runner_bin.run()
