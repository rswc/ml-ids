# from cicids import CICIDS2017
from river.datasets import ImageSegments, SMSSpam

from river import tree
from river import metrics
from river.metrics.base import Metrics
from framework import ExperimentRunner
from metrics import MetricWrapper, FalseDiscoveryRate, FalsePositiveRate, AttackDetectionRate


dataset_mul = ImageSegments()
dataset_bin = SMSSpam()

model_mul = tree.HoeffdingTreeClassifier()
model_bin = tree.HoeffdingTreeClassifier()

metrics_mul = Metrics([
    MetricWrapper(
        metric=metrics.GeometricMean(),
        window_size=300,
    ),
    MetricWrapper(
        metrics.GeometricMean()
    ),
    MetricWrapper(
        metric=AttackDetectionRate(),
        collapse_label=False,
        collapse_classes=['sky']
    ),
    MetricWrapper(
        metric=AttackDetectionRate(),
        window_size=100,
        collapse_label=False,
        collapse_classes=['sky']
    ),
    MetricWrapper(
        metric=FalseDiscoveryRate(),
        window_size=500,
        collapse_label=False,
        collapse_classes=['sky']
    ),
    MetricWrapper(
        metric=FalsePositiveRate(),
        window_size=500,
        collapse_label=False,
        collapse_classes=['sky']
    )
])

metrics_bin = Metrics([
    metrics.F1(),
    MetricWrapper(
        metric=AttackDetectionRate(),
        window_size=1000,
    ),
    MetricWrapper(
        metric=FalsePositiveRate(),
        window_size=1000,
    ),
    MetricWrapper(
        metric=FalseDiscoveryRate(),
        window_size=1000,
    ),
])

runner_bin = ExperimentRunner(model_bin, dataset_bin, metrics_bin, "./out", project="ml-ids", enable_tracker=True)
runner_mul = ExperimentRunner(model_mul, dataset_mul, metrics_mul, "./out", project="ml-ids", enable_tracker=True)

runner_mul.run()
# runner_bin.run()
