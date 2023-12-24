import abc
from typing import Generic, TypeVar, Type

MODEL = TypeVar("MODEL")
DRIFT = TypeVar("DRIFT")

class ModelAdapterBase(abc.ABC, Generic[MODEL]):

    def __init__(self) -> None:
        self._model = None

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

class DriftModelAdapterBase(Generic[MODEL, DRIFT], ModelAdapterBase[MODEL]):
    """Helper extension to support integrating adapters for model's inner drift detector."""

    def __init__(self, drift_adapter: Type[ModelAdapterBase] = None) -> None:
        super().__init__()
        self._drift_adapter = drift_adapter
        self._drift_adapters: dict[str, ModelAdapterBase] = dict()

    @abc.abstractmethod
    def _get_drift_prototype(self) -> Type[DRIFT]:
        """Access the model's "prototype" drift detector."""

    @abc.abstractmethod
    def _get_drift_detectors(self) -> dict[str, DRIFT]:
        """Access the model's drift detector list."""

    def add_drift_state(self, state: dict, active_classes: list[str] = None):
        if self._drift_adapter is not None:
            for cls in self._get_drift_detectors().keys():
                if cls not in self._drift_adapters:
                    self._drift_adapters[cls] = self._drift_adapter()
                
                self._drift_adapters[cls].model = self._get_drift_detectors()[cls]

            drift = {cls: self._drift_adapters[cls].get_loggable_state() for cls in self._drift_adapters}

            # Log state for active classes only
            state["drift"] = {cls: stats for cls, stats in drift.items() if stats is not None and (active_classes is None or cls in active_classes)}
        
        return state

    @ModelAdapterBase.model.setter
    def model(self, model: MODEL):
        if self._drift_adapter is not None and not isinstance(self._get_drift_prototype(), self._drift_adapter.get_target_class()):
            raise ValueError(f"Model with {self._get_drift_prototype().__class__.__name__} as drift detector was provided to an adapter expecting {self._drift_adapter.__name__}")

        super(__class__, self.__class__).model.__set__(self, model)
