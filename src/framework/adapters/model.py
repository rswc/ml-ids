from typing import Type

from river.base.drift_detector import DriftDetector
from .base import ModelAdapterBase, DriftModelAdapterBase
from cbce import CBCE
from river.forest import ARFClassifier

def extract_mean(prop, models):
    return sum(map(lambda m: getattr(m, prop, 0.0), models)) / len(models)

class CBCEAdapter(DriftModelAdapterBase[CBCE]):

    @classmethod
    def get_target_class(self):
        return CBCE
    
    def _get_drift_prototype(self, model: CBCE):
        return model.drift_detector
    
    def _get_drift_detectors(self) -> dict[str, DriftDetector]:
        return self._model.drift_detectors
    
    def get_loggable_state(self) -> dict:
        state = {
            "class_priors": self._model._class_priors,
        }

        return self.add_drift_state(state, active_classes=self._model.classifiers.keys())

class ARFAdapter(ModelAdapterBase[ARFClassifier]):

    @classmethod
    def get_target_class(self) -> type[ARFClassifier]:
        return ARFClassifier
    
    def get_loggable_state(self) -> dict:
        return {
            "mean_active_leaves": extract_mean("_n_active_leaves", self.model._background),
            "mean_inactive_leaves": extract_mean("_n_inactive_leaves", self.model._background),
        }
