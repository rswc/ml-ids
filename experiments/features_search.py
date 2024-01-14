from cbce import CBCE
from cicids import CICIDS2017, generate_cicids_file
from framework import ExperimentRunner
from metrics import (
    FalseDiscoveryRate,
    FalsePositiveRate,
    AttackDetectionRate,
    MetricWrapper,
)
from pathlib import Path
from rbc import ResamplingBaggingClassifier
from river import metrics
from river.drift.binary import DDM
from river.forest import ARFClassifier
from river.metrics.base import Metrics
from river.tree import HoeffdingTreeClassifier

OUT_DIR = "./out"
PROJECT = "features-search"
SEED = 42

# CICIDS2017 Setup
# Substitute `None` with dataset dir root path if needed
# defaults to %project-root%/cicids2017
DATASET_ROOT: Path = None

if __name__ == "__main__":
    # Subset 400k_400k
    convert_att = True
    enable_tracker = True
    start_idx = 400_000
    n_samples = 400_000
    window_size = 4_000
    summary_metric = "mean"

    bin_metric_params = {
        "window_size": window_size,
        "collapse_label": False,
        "collapse_classes": ["BENIGN"],
    }
    
    features = [ 
        (CICIDS2017.Features.DEFAULT, "DEFAULT"),
        (CICIDS2017.Features.KURNIABUDI2020, "KURNIABUDI2020"),
        (CICIDS2017.Features.YULIANTO2019, "YULIANTO2019")
    ]
    
    cicids_file = generate_cicids_file(
        dataset_path=DATASET_ROOT,
        start_sample=start_idx,
        n_samples=n_samples,
        convert_attempted=convert_att,
    )

    for feature_set, fs_name in features:

        dataset = CICIDS2017(
            cicids_file.name,
            dataset_dir=DATASET_ROOT,
            n_samples=n_samples,
            convert_attempted=convert_att,
            used_features=feature_set
        )

        global_metrics = Metrics(
            [
                MetricWrapper(
                    metric=metrics.GeometricMean(),
                    window_size=window_size,
                ),
                metrics.GeometricMean(),
                MetricWrapper(metric=AttackDetectionRate(), **bin_metric_params),
                MetricWrapper(metric=FalseDiscoveryRate(), **bin_metric_params),
                MetricWrapper(metric=FalsePositiveRate(), **bin_metric_params),
            ]
        )

        models = [
            HoeffdingTreeClassifier(),
            CBCE(
                classifier=HoeffdingTreeClassifier(),
                drift_detector=DDM(),
                seed=SEED,
            ),
            ARFClassifier(seed=SEED),
            ResamplingBaggingClassifier(model=HoeffdingTreeClassifier(), seed=SEED),
        ]
        

        for model in models:
            runner = ExperimentRunner(
                model,
                dataset,
                global_metrics,
                OUT_DIR,
                summary_metric=summary_metric,
                enable_tracker=enable_tracker,
                project=PROJECT,
                tags=["cicsub:400k_400k", f"features:{fs_name}"]
            )
            runner.run()

