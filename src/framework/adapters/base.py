import abc
from typing import Generic, TypeVar, Type

T = TypeVar("T")

class ModelAdapterBase(abc.ABC, Generic[T]):

    def __init__(self) -> None:
        self._model = None

    @property
    def model(self) -> T:
        """The specific model instance this adapter is tracking."""
        return self._model
    
    @model.setter
    def model(self, model: T):
        if not isinstance(model, self.get_target_class()):
            raise ValueError(f"An instance of {model.__class__.__name__} was provided to an adapter of {self.get_target_class().__name__}")

        self._model = model

    @classmethod
    @abc.abstractmethod
    def get_target_class(self) -> Type[T]:
        """The Python class of the model this adapter was made for."""

    @abc.abstractmethod
    def get_loggable_state(self) -> dict:
        """Return a dict with a subset of the model's state determined interesting to log."""
    
    def update(self, y, y_pred) -> None:
        """Call this at the end of each time step, to let the adapter know it needs to update its state."""
