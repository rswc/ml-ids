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
    """Helper extension to support integrating adapters for model's inner drift detector.
    Some models use `DriftAndWarningDetector` instances, while others use separate `DriftDetector` instances,
    designated as "warning" and "drift". This class of adapters support both cases. 
    
    Parameters
    ----------
    drift_adapter
        (optional) Class (not instance!) of the drift adapter. If None, disables the drift detector adapter portion.
    warning_adapter
        (optional) Class of the warning adapter. If None, assumed to be the same as `drift_adapter`. If the model
        does not use separate drift and warning detectors, this has no effect.
    
    """

    def __init__(self, drift_adapter: Type[ModelAdapterBase] = None, warning_adapter: Type[ModelAdapterBase] = None) -> None:
        super().__init__()

        if drift_adapter is None:
            print("WARNING: Drift-detecting model's adapter initialized without specifying drift_adapter. The drift detector's state will not be logged.")

        if drift_adapter is not None and not isinstance(drift_adapter, type):
            print("WARNING: The drift detector adapter is expected to be a type, not an instance.")
            drift_adapter = drift_adapter.__class__

        if warning_adapter is not None and not isinstance(warning_adapter, type):
            print("WARNING: The drift detector warning adapter is expected to be a type, not an instance.")
            warning_adapter = warning_adapter.__class__

        self._drift_adapter = drift_adapter
        self._drift_adapters: dict[str, ModelAdapterBase] = dict()

        self._warning_adapter = warning_adapter or drift_adapter
        self._warning_adapters: dict[str, ModelAdapterBase] = dict()

        if warning_adapter is not None and not self._warning_detectors_separate():
            print("WARNING: warning_adapter parameter passed to adapter of model which does not separate drift and warning instances. Warning adapter will be ignored.")
            self._warning_adapter = None
    
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

    @abc.abstractmethod
    def _warning_detectors_separate(self) -> bool:
        """Define whether the model uses separate warning and drift detectors or not."""

    def _get_drift_warning_prototype(self, model: MODEL) -> DriftDetector:
        """Access the model's "prototype" drift warning detector. This is useful for models which use separate
        detector instances for warning (as opposed to a single DriftAndWarning instance)."""

        return None

    def _get_drift_warning_detectors(self) -> dict[str, DriftDetector]:
        """Access the model's drift warning detector list. This is useful for models which use separate
        detector instances for warning (as opposed to a single DriftAndWarning instance)."""

        return None
    
    def __extract_state(self, proto: Type[ModelAdapterBase], detectors: dict[str, DriftDetector], adapters: dict[str, ModelAdapterBase], active_classes: list[str] = None):
        for cls in detectors.keys():
            if cls not in adapters:
                adapters[cls] = proto()
            
            adapters[cls].model = detectors[cls]

        drift = {cls: adapters[cls].get_loggable_state() for cls in adapters}

        # Log state for active classes only
        return {
            cls: stats
            for cls, stats in drift.items()
            if stats is not None and (active_classes is None or cls in active_classes)
        }

    def add_drift_state(self, state: dict, active_classes: list[str] = None):
        """Extend state object with drift detector information.
        
        Parameters
        ----------
        state
            State object to be extended.
        active_classes
            (optional) If specified, will only log state of detectors corresponding to these labels.
            This can be useful for model esembles which activate and deactivate certain base classifiers.
        
        """

        if self._drift_adapter is not None:
            state[f"drift.{self._get_drift_prototype(self._model).__class__.__name__}"] = self.__extract_state(
                self._drift_adapter,
                self._get_drift_detectors(),
                self._drift_adapters,
                active_classes
            )
        
        if self._warning_adapter is not None:
            state[f"drift_warning.{self._get_drift_warning_prototype(self._model).__class__.__name__}"] = self.__extract_state(
                self._warning_adapter,
                self._get_drift_warning_detectors(),
                self._warning_adapters,
                active_classes
            )

        return state

    @ModelAdapterBase.model.setter
    def model(self, model: MODEL):
        if self._drift_adapter is not None and not isinstance(self._get_drift_prototype(model), self._drift_adapter.get_target_class()):
            raise ValueError(f"Model with {self._get_drift_prototype(model).__class__.__name__} as drift detector was provided to an adapter expecting {self._drift_adapter.__name__}")

        if self._warning_adapter is not None and not isinstance(self._get_drift_warning_prototype(model), self._warning_adapter.get_target_class()):
            raise ValueError(f"Model with {self._get_drift_warning_prototype(model).__class__.__name__} as drift warning detector was provided to an adapter expecting {self._warning_adapter.__name__}")

        super(__class__, self.__class__).model.__set__(self, model)
