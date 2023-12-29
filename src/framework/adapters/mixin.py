import abc
from river.metrics.base import Metrics
from framework.util import get_metrics_dict

class PerClassMetricsMixin(abc.ABC):

    def __init__(self, per_class_metrics: Metrics = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._per_class_metrics_prototype = per_class_metrics
        self._per_class_metrics: dict[str, Metrics] = dict()
    
    @abc.abstractmethod
    def _get_inner_classifiers(self) -> dict:
        """Collection of the per-class "inner" or "base" classifiers of the model."""

    def update(self, y, y_pred, *args, **kwargs) -> None:
        super().update(y, y_pred, *args, **kwargs)

        if y not in self._per_class_metrics:
            self._per_class_metrics[y] = self._per_class_metrics_prototype.clone()
            self._per_class_metrics[y]
        
        for cls, metric in self._per_class_metrics.items():
            metric.update(y == cls, y_pred == cls)

    def add_per_class_state(self, state):
        state["class"] = {cls: get_metrics_dict(metrics) for cls, metrics in self._per_class_metrics.items()}

        return state
