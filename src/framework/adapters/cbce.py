from .base import ModelAdapterBase
from cbce import CBCE

class CBCEAdapter(ModelAdapterBase[CBCE]):

    @classmethod
    def get_target_class(cls):
        return CBCE
    
    def get_loggable_state(self) -> dict:
        return {
            "class_priors": self._model._class_priors
        }
