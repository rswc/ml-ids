from river import tree
from river import metrics
from river.metrics.base import Metrics
from framework import ExperimentRunner
from metrics import MetricWrapper

from cbce import CBCE
from synthstream import ClassSampler, SyntheticStream
import math
from ucimlrepo import fetch_ucirepo 
from river import linear_model

CHUNK_SIZE = 150
N_SAMPLES = 11 * CHUNK_SIZE
SEED = 42

def extract_samples(X, Y, labels: list[str]) -> dict:
    samples = {label: [] for label in labels}
    n_samples = 0
    
    for x, y in zip(X, Y):
        if y in samples.keys():
            samples[y].append(x)
            n_samples += 1
    return samples, n_samples

def c_prior(t: int, n: int = N_SAMPLES): 
    z1 = (t - n/4)/(n/64)
    z2 = -(t - n*3/4)/(n/64)
    f1 = 1/(1 + math.exp(z1))
    f2 = 1/(1 + math.exp(z2))
    return f1 + f2

if __name__ == '__main__':
    # fetch dataset 
    letter_recognition = fetch_ucirepo(id=59) 

    X = [ x[1].to_dict() for x in letter_recognition.data.features.iterrows() ]
    y = [ y[1].to_dict()['lettr'] for y in letter_recognition.data.targets.iterrows() ]

    letter_samples, cnt_samples = extract_samples(X, y, labels=['A', 'B', 'C'])

    ss = SyntheticStream(
        max_samples=N_SAMPLES,
        seed=42, 
        init_csamplers=[
            ClassSampler('A', samples=letter_samples['A'], weight_func=lambda t: 1),
            ClassSampler('B', samples=letter_samples['B'], weight_func=lambda t: 1),
            ClassSampler('C', samples=letter_samples['C'], weight_func=c_prior),
        ]
    )
    
    model = CBCE(linear_model.LogisticRegression(), seed=42)

    my_metrics = Metrics([
        MetricWrapper(
            metric=metrics.F1(),
            collapse_classes=['A'],
            collapse_label=True,
            window_size=CHUNK_SIZE,# one-eleventh as described in paper
        ),
        MetricWrapper(
            metric=metrics.F1(),
            collapse_classes=['B'],
            collapse_label=True,
            window_size=CHUNK_SIZE,# one-eleventh as described in paper
        ),
        MetricWrapper(
            metric=metrics.F1(),
            collapse_classes=['C'],
            collapse_label=True,
            window_size=CHUNK_SIZE, # one-eleventh as described in paper
        ),
        MetricWrapper(
            metric=metrics.GeometricMean(),
            window_size=CHUNK_SIZE,
        )
    ])
    runner = ExperimentRunner(model, ss, my_metrics, "./out", project='ml-ids', enable_tracker=True)
    runner.run()