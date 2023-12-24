from .base import ModelAdapterBase
import math
from river.drift.binary import DDM

class DDMAdapter(ModelAdapterBase[DDM]):

    @classmethod
    def get_target_class(self):
        return DDM
    
    def get_loggable_state(self) -> dict:
        p = self._model._p.get()
        n = self._model._p.n
        s = math.sqrt(p * (1 - p) / n)

        return {
            "error_probability": p,
            "ps": p + s,
            "warn_threshold": self._model._p_min + self._model.warning_threshold * self._model._s_min,
            "alarm_threshold": self._model._p_min + self._model.drift_threshold * self._model._s_min,
        }
