from .base import ModelAdapterBase
from .mixin import PerClassMetricsMixin
from cbce import CBCE
from river.forest import ARFClassifier

def extract_mean(prop, models):
    return sum(map(lambda m: getattr(m, prop, 0.0), models)) / len(models)

class CBCEAdapter(PerClassMetricsMixin, ModelAdapterBase[CBCE]):

    @classmethod
    def get_target_class(self):
        return CBCE
    
    def _get_inner_classifiers(self) -> dict:
        return self._model.classifiers
    
    def get_loggable_state(self) -> dict:
        state = {
            "class_priors": self._model._class_priors
        }

        return self.add_per_class_state(state)

class ARFAdapter(PerClassMetricsMixin, ModelAdapterBase[ARFClassifier]):

    @classmethod
    def get_target_class(self) -> type[ARFClassifier]:
        return ARFClassifier

    def _get_inner_classifiers(self) -> dict:
        return self._model.models
    
    def get_loggable_state(self) -> dict:
        state = {
            "mean_active_leaves": extract_mean("_n_active_leaves", self.model._background),
            "mean_inactive_leaves": extract_mean("_n_inactive_leaves", self.model._background),
        }

        return self.add_per_class_state(state)
