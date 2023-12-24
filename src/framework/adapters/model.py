from .base import ModelAdapterBase
from cbce import CBCE
from river.forest import ARFClassifier
from typing import Type

def extract_mean(prop, models):
    return sum(map(lambda m: getattr(m, prop, 0.0), models)) / len(models)

class CBCEAdapter(ModelAdapterBase[CBCE]):

    def __init__(self, drift_adapter: Type[ModelAdapterBase] = None) -> None:
        super().__init__()
        self._drift_adapter = drift_adapter
        self._drift_adapters: dict[str, ModelAdapterBase] = dict()
    
    @ModelAdapterBase.model.setter
    def model(self, model: CBCE):
        if self._drift_adapter is not None and not isinstance(model.drift_detector, self._drift_adapter.get_target_class()):
            raise ValueError(f"Model with {model.drift_detector.__class__.__name__} as drift detector was provided to an adapter expecting {self._drift_adapter.__name__}")

        super(__class__, self.__class__).model.__set__(self, model)

    @classmethod
    def get_target_class(self):
        return CBCE
    
    def get_loggable_state(self) -> dict:
        state = {
            "class_priors": self._model._class_priors,
        }

        if self._drift_adapter is not None:
            for cls in self._model.drift_detectors.keys():
                if cls not in self._drift_adapters:
                    self._drift_adapters[cls] = self._drift_adapter()
                
                self._drift_adapters[cls].model = self._model.drift_detectors[cls]

            drift = {cls: self._drift_adapters[cls].get_loggable_state() for cls in self._drift_adapters if cls in self._model.classifiers.keys()}

            # Log state for active classes only
            state["drift"] = {cls: stats for cls, stats in drift.items() if stats is not None}

        return state

class ARFAdapter(ModelAdapterBase[ARFClassifier]):

    @classmethod
    def get_target_class(self) -> type[ARFClassifier]:
        return ARFClassifier
    
    def get_loggable_state(self) -> dict:
        return {
            "mean_active_leaves": extract_mean("_n_active_leaves", self.model._background),
            "mean_inactive_leaves": extract_mean("_n_inactive_leaves", self.model._background),
        }
