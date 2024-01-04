from river import metrics, linear_model
from river.forest import ARFClassifier
from river.ensemble import BaggingClassifier
from river.tree import HoeffdingTreeClassifier
from river.metrics.base import Metrics
from framework import ExperimentRunner, DatasetAnalyzer
from metrics import MetricWrapper
from cbce import CBCE
from synthstream import ClassSampler, SyntheticStream
from ucimlrepo import fetch_ucirepo
from copy import deepcopy
import math

CHUNK_SIZE = 150
N_SAMPLES = 11 * CHUNK_SIZE
SEED_CBCE = 42
SEED_SS = 42
SEED_OB = 42
SEED_RF = 42
ENABLE_TRACKER = True
OUT_DIR = "./out"
PROJECT = "test-ml-ids"


def extract_samples(X, Y, labels: list[str]) -> dict:
    samples = {label: [] for label in labels}
    n_samples = 0

    for x, y in zip(X, Y):
        if y in samples.keys():
            samples[y].append(x)
            n_samples += 1
    return samples, n_samples


if __name__ == "__main__":
    letter_recognition = fetch_ucirepo(id=59)

    X = [x[1].to_dict() for x in letter_recognition.data.features.iterrows()]
    y = [y[1].to_dict()["lettr"] for y in letter_recognition.data.targets.iterrows()]

    letter_samples, cnt_samples = extract_samples(X, y, labels=["A", "B", "C"])

    c_prior = lambda t: 1 / (1 + math.exp(-(t - N_SAMPLES / 2) / (N_SAMPLES / 16)))
    ss = SyntheticStream(
        n_features=16,
        max_samples=N_SAMPLES,
        seed=SEED_SS,
        init_csamplers=[
            ClassSampler("A", samples=letter_samples["A"], weight_func=lambda t: 1),
            ClassSampler("B", samples=letter_samples["B"], weight_func=lambda t: 1),
            ClassSampler("C", samples=letter_samples["C"], weight_func=c_prior),
        ],
    )

    my_metrics = Metrics(
        [
            MetricWrapper(
                metric=metrics.F1(),
                collapse_classes=["A"],
                collapse_label=True,
                window_size=CHUNK_SIZE,  # one-eleventh as described in paper
            ),
            MetricWrapper(
                metric=metrics.F1(),
                collapse_classes=["B"],
                collapse_label=True,
                window_size=CHUNK_SIZE,  # one-eleventh as described in paper
            ),
            MetricWrapper(
                metric=metrics.F1(),
                collapse_classes=["C"],
                collapse_label=True,
                window_size=CHUNK_SIZE,  # one-eleventh as described in paper
            ),
            MetricWrapper(
                metric=metrics.GeometricMean(),
                window_size=CHUNK_SIZE,
            ),
        ]
    )

    ss_copy = deepcopy(ss)
    analyzer = DatasetAnalyzer(
        ss_copy,
        window_size=CHUNK_SIZE,
        out_dir=OUT_DIR,
        project=PROJECT,
        enable_tracker=ENABLE_TRACKER,
    )
    analyzer.analyze()

    ss_copy = deepcopy(ss)
    cbce = CBCE(linear_model.LogisticRegression(), seed=SEED_CBCE)
    runner = ExperimentRunner(
        cbce,
        ss_copy,
        my_metrics,
        out_dir=OUT_DIR,
        project=PROJECT,
        enable_tracker=ENABLE_TRACKER,
    )
    runner.run()

    ss_copy = deepcopy(ss)
    arf = ARFClassifier(seed=SEED_RF)
    runner = ExperimentRunner(
        arf,
        ss_copy,
        my_metrics,
        out_dir=OUT_DIR,
        project=PROJECT,
        enable_tracker=ENABLE_TRACKER,
    )
    runner.run()

    ss_copy = deepcopy(ss)
    ob = BaggingClassifier(HoeffdingTreeClassifier(), seed=SEED_OB)
    runner = ExperimentRunner(
        ob,
        ss_copy,
        my_metrics,
        out_dir=OUT_DIR,
        project=PROJECT,
        enable_tracker=ENABLE_TRACKER,
    )
    runner.run()
