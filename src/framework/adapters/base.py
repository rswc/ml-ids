import abc

class ModelAdapterBase(abc.ABC):

    def __init__(self) -> None:
        self._model = None

    @property
    def model(self):
        """The specific model instance this adapter is tracking."""
        return self._model
    
    @model.setter
    def model(self, model):
        if not isinstance(model, self.get_target_class()):
            raise ValueError(f"An instance of {model.__class__.__name__} was provided to an adapter of {self.get_target_class().__name__}")

        self._model = model

    @classmethod
    @abc.abstractmethod
    def get_target_class(self):
        """The Python class of the model this adapter was made for."""

    @abc.abstractmethod
    def get_loggable_state(self) -> dict:
        """Return a dict with a subset of the model's state determined interesting to log."""
