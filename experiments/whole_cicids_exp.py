from river.drift.binary import DDM
from river.ensemble import BaggingClassifier
from river.forest import ARFClassifier
from river.tree import HoeffdingTreeClassifier
from cbce import CBCE
from cicids import CICIDS2017
from river import metrics, linear_model
from river.metrics.base import Metrics
from framework import ExperimentRunner
from metrics import MetricWrapper, FalseDiscoveryRate, FalsePositiveRate, AttackDetectionRate

dataset = CICIDS2017(filename='cicids_final.csv')

SEED = 42

cbce = CBCE(linear_model.LogisticRegression(), seed=SEED)
cbce_ddm = CBCE(linear_model.LogisticRegression(), seed=SEED, drift_detector=DDM())
ob = BaggingClassifier(HoeffdingTreeClassifier(), seed=SEED)
arf = ARFClassifier(seed=SEED)


metrics = Metrics([
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

runner1 = ExperimentRunner(cbce, dataset, metrics, "../out", project="cicids-test", enable_tracker=True)
runner2 = ExperimentRunner(ob, dataset, metrics, "../out", project="cicids-test", enable_tracker=True)
runner3 = ExperimentRunner(arf, dataset, metrics, "../out", project="cicids-test", enable_tracker=True)
runner4 = ExperimentRunner(cbce_ddm, dataset, metrics, "../out", project="cicids-test", enable_tracker=True)

runner1.run()
runner2.run()
runner3.run()
runner4.run()
