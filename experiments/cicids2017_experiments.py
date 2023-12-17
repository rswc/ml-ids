from cicids import CICIDS2017, generate_cicids_file, predict_cicids_filename
from framework import ExperimentRunner, DatasetAnalyzer
from metrics import (
    MetricWrapper,
    AttackDetectionRate,
    FalseDiscoveryRate,
    FalsePositiveRate,
)
from pathlib import Path
from river import metrics, linear_model
from river.forest import ARFClassifier
from river.metrics.base import Metrics
from river.drift.no_drift import NoDrift

OUT_DIR = "./out"
PROJECT = "cicids-test"

# Substitute `None` with dataset dir root path if needed
# defaults to %project-root%/cicids2017
DATASET_ROOT: Path = None
# Take care! Changing n_samples and start_idx produces new dataset
# start_idx = 0, n_samples = None -> all dataset, will strip suffix _idx=...
START_IDX = 400_000
N_SAMPLES = 400_000
# Convert "Attempted" to "BENIGN" - will use _noatt dataset if set
CONVERT_ATTEMPTED = False
WINDOW_SIZE = 8000

bin_metric_params = {
    "window_size": WINDOW_SIZE,
    "collapse_label": False,
    "collapse_classes": ["BENIGN"],
}

METRICS = Metrics(
    [
        MetricWrapper(
            metric=metrics.GeometricMean(),
            window_size=WINDOW_SIZE,
        ),
        MetricWrapper(metrics.GeometricMean()),
        MetricWrapper(
            metric=AttackDetectionRate(),
            collapse_label=False,
            collapse_classes=["BENIGN"],
        ),
        MetricWrapper(metric=AttackDetectionRate(), **bin_metric_params),
        MetricWrapper(metric=FalseDiscoveryRate(), **bin_metric_params),
        MetricWrapper(metric=FalsePositiveRate(), **bin_metric_params),
    ]
)

if __name__ == "__main__":
    cicids_file = generate_cicids_file(
        dataset_path=DATASET_ROOT,
        n_samples=N_SAMPLES,
        start_sample=START_IDX,
        convert_attempted=CONVERT_ATTEMPTED,
    )

    dataset = CICIDS2017(
        cicids_file.name,
        dataset_dir=DATASET_ROOT,
        n_samples=N_SAMPLES,
        convert_attempted=CONVERT_ATTEMPTED,
    )

    analyzer = DatasetAnalyzer(dataset, WINDOW_SIZE, OUT_DIR, project=PROJECT)
    # analyzer.analyze()

    model = ARFClassifier(drift_detector=NoDrift())
    runner = ExperimentRunner(
        model, dataset, METRICS, OUT_DIR, project=PROJECT, enable_tracker=False
    )
    runner.run()
