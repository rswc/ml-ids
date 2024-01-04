import abc
from typing import Generic, TypeVar, Type
from river.base.drift_detector import DriftDetector

MODEL = TypeVar("MODEL")

class ModelAdapterBase(abc.ABC, Generic[MODEL]):

    def __init__(self) -> None:
        self._model = None
    
    def get_parameters(self) -> dict:
        """The adapter's parameters"""
        return {
            "name": self.__class__.__name__
        }

    @property
    def model(self) -> MODEL:
        """The specific model instance this adapter is tracking."""
        return self._model
    
    @model.setter
    def model(self, model: MODEL):
        if not isinstance(model, self.get_target_class()):
            raise ValueError(f"An instance of {model.__class__.__name__} was provided to an adapter of {self.get_target_class().__name__}")

        self._model = model

    @classmethod
    @abc.abstractmethod
    def get_target_class(self) -> Type[MODEL]:
        """The Python class of the model this adapter was made for."""

    @abc.abstractmethod
    def get_loggable_state(self) -> dict:
        """Return a dict with a subset of the model's state determined interesting to log."""
    
    def update(self, y, y_pred) -> None:
        """Call this at the end of each time step, to let the adapter know it needs to update its state."""

class DriftModelAdapterBase(Generic[MODEL], ModelAdapterBase[MODEL]):
    """Helper extension to support integrating adapters for model's inner drift detector."""

    def __init__(self, drift_adapter: Type[ModelAdapterBase] = None, warning_adapter: Type[ModelAdapterBase] = None) -> None:
        super().__init__()

        if drift_adapter is None:
            print("WARNING: Drift-detecting model's adapter initialized without specifying drift_adapter. The drift detector's state will not be logged.")

        if not isinstance(drift_adapter, type):
            print("WARNING: The drift detector adapter is expected to be a type, not an instance.")
            drift_adapter = drift_adapter.__class__

        if not isinstance(warning_adapter, type):
            print("WARNING: The drift detector warning adapter is expected to be a type, not an instance.")
            warning_adapter = warning_adapter.__class__

        self._drift_adapter = drift_adapter
        self._drift_adapters: dict[str, ModelAdapterBase] = dict()

        if warning_adapter is not None and self._get_drift_warning_detectors() is not None:
            print("WARNING: warning_adapter parameter passed to adapter of model which does not separate drift and warning instances. Warning adapter will be ignored.")
            warning_adapter = None
        
        elif warning_adapter is None and self._get_drift_warning_detectors() is not None:
            warning_adapter = drift_adapter

        self._warning_adapter = warning_adapter
        self._warning_adapters: dict[str, ModelAdapterBase] = dict()
    
    def get_parameters(self) -> dict:
        return {
            **super().get_parameters(),
            "drift_adapter_name": self._drift_adapter.__name__ if self._drift_adapter is not None else None
        }

    @abc.abstractmethod
    def _get_drift_prototype(self, model: MODEL) -> DriftDetector:
        """Access the model's "prototype" drift detector."""

    @abc.abstractmethod
    def _get_drift_detectors(self) -> dict[str, DriftDetector]:
        """Access the model's drift detector list."""

    def _get_drift_warning_prototype(self, model: MODEL) -> DriftDetector:
        """Access the model's "prototype" drift warning detector. This is useful for models which use separate
        detector instances for warning (as opposed to a single DriftAndWarning instance)."""

        return None

    def _get_drift_warning_detectors(self) -> dict[str, DriftDetector]:
        """Access the model's drift warning detector list. This is useful for models which use separate
        detector instances for warning (as opposed to a single DriftAndWarning instance)."""

        return None

    def add_drift_state(self, state: dict, active_classes: list[str] = None):
        if self._drift_adapter is not None:
            for cls in self._get_drift_detectors().keys():
                if cls not in self._drift_adapters:
                    self._drift_adapters[cls] = self._drift_adapter()
                
                self._drift_adapters[cls].model = self._get_drift_detectors()[cls]

            drift = {cls: self._drift_adapters[cls].get_loggable_state() for cls in self._drift_adapters}

            # Log state for active classes only
            state[f"drift.{self._get_drift_prototype(self._model).__class__.__name__}"] = {
                cls: stats
                for cls, stats in drift.items()
                if stats is not None and (active_classes is None or cls in active_classes)
            }
        
        if self._warning_adapter is not None:
            for cls in self._get_drift_warning_detectors().keys():
                if cls not in self._warning_adapters:
                    self._warning_adapters[cls] = self._warning_adapter()
                
                self._warning_adapters[cls].model = self._get_drift_warning_detectors()[cls]

            drift = {cls: self._warning_adapters[cls].get_loggable_state() for cls in self._warning_adapters}

            # Log state for active classes only
            state[f"drift_warning.{self._get_drift_warning_prototype(self._model).__class__.__name__}"] = {
                cls: stats
                for cls, stats in drift.items()
                if stats is not None and (active_classes is None or cls in active_classes)
            }

        return state

    @ModelAdapterBase.model.setter
    def model(self, model: MODEL):
        if self._drift_adapter is not None and not isinstance(self._get_drift_prototype(model), self._drift_adapter.get_target_class()):
            raise ValueError(f"Model with {self._get_drift_prototype(model).__class__.__name__} as drift detector was provided to an adapter expecting {self._drift_adapter.__name__}")

        super(__class__, self.__class__).model.__set__(self, model)
