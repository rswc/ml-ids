from cbce import CBCE
from cicids import CICIDS2017, generate_cicids_file, predict_cicids_filename
from framework import ExperimentRunner, DatasetAnalyzer
from framework.adapters.model import CBCEAdapter, ARFAdapter, RBCAdapter
from framework.adapters.drift import ADWINAdapter, DDMAdapter
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
from river.metrics import Recall
from river.metrics.base import Metrics
from river.tree import HoeffdingTreeClassifier
from river.utils import Rolling

OUT_DIR = "./out"
PROJECT = "new-year-experiments"
SEED = 42

# CICIDS2017 Setup
# Substitute `None` with dataset dir root path if needed
# defaults to %project-root%/cicids2017
DATASET_ROOT: Path = None

if __name__ == "__main__":
    # Subset 400k_400k
    analyze_cicids_sub1 = True
    # Subset 1200k_400k
    analyze_cicids_sub2 = True
    convert_att = True
    enable_tracker = True
    n_samples = 400_000

    datasets_with_tags = []
    if analyze_cicids_sub2:
        cicids_file = generate_cicids_file(
            dataset_path=DATASET_ROOT,
            start_sample=1_200_000,
            n_samples=n_samples,
            convert_attempted=convert_att,
        )

        dataset10 = CICIDS2017(
            cicids_file.name,
            dataset_dir=DATASET_ROOT,
            n_samples=n_samples,
            convert_attempted=convert_att,
        )
        datasets_with_tags.append(
            (dataset10, ["cicsub:1200k_400k", "features:DEFAULT"])
        )

        dataset25 = CICIDS2017(
            cicids_file.name,
            dataset_dir=DATASET_ROOT,
            n_samples=n_samples,
            convert_attempted=convert_att,
            used_features=CICIDS2017.Features.YULIANTO2019,
        )
        datasets_with_tags.append(
            (dataset25, ["cicsub:1200k_400k", "features:YULIANTO2019"])
        )

    if analyze_cicids_sub1:
        cicids_file = generate_cicids_file(
            dataset_path=DATASET_ROOT,
            start_sample=400_000,
            n_samples=n_samples,
            convert_attempted=convert_att,
        )

        dataset10 = CICIDS2017(
            cicids_file.name,
            dataset_dir=DATASET_ROOT,
            n_samples=n_samples,
            convert_attempted=convert_att,
        )
        datasets_with_tags.append((dataset10, ["cicsub:400k_400k", "features:DEFAULT"]))

        dataset25 = CICIDS2017(
            cicids_file.name,
            dataset_dir=DATASET_ROOT,
            n_samples=n_samples,
            convert_attempted=convert_att,
            used_features=CICIDS2017.Features.YULIANTO2019,
        )
        datasets_with_tags.append(
            (dataset25, ["cicsub:400k_400k", "features:YULIANTO2019"])
        )

    window_sizes = [4_000]

    for dataset, dataset_tags in datasets_with_tags:
        for window_size in window_sizes:
            analyzer = DatasetAnalyzer(
                dataset,
                window_size=window_size,
                out_dir=OUT_DIR,
                project=PROJECT,
                enable_tracker=enable_tracker,
            )
            analyzer.analyze()

            bin_metric_params = {
                "window_size": window_size,
                "collapse_label": False,
                "collapse_classes": ["BENIGN"],
            }
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

            pc_metrics = Metrics(
                [
                    Rolling(Recall(), window_size),
                    Rolling(FalsePositiveRate(), window_size),
                    Rolling(FalseDiscoveryRate(), window_size),
                    Rolling(AttackDetectionRate(), window_size),
                ]
            )

            models = [
                CBCE(
                    classifier=HoeffdingTreeClassifier(),
                    drift_detector=DDM(),
                    seed=SEED,
                ),
                ARFClassifier(seed=SEED, warning_detector=DDM(), drift_detector=DDM()),
                ResamplingBaggingClassifier(model=HoeffdingTreeClassifier(), seed=SEED),
            ]
            adapters = [
                CBCEAdapter(per_class_metrics=pc_metrics, drift_adapter=DDMAdapter),
                ARFAdapter(per_class_metrics=pc_metrics),
                RBCAdapter(per_class_metrics=pc_metrics),
            ]

            for model, adapter in zip(models, adapters):
                runner = ExperimentRunner(
                    model,
                    dataset,
                    global_metrics,
                    OUT_DIR,
                    model_adapter=adapter,
                    enable_tracker=enable_tracker,
                    project=PROJECT,
                    tags=dataset_tags,
                )
                runner.run()
