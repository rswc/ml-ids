from .base import ModelAdapterBase
import math
from river.drift.binary import DDM
from river.drift import ADWIN

class DDMAdapter(ModelAdapterBase[DDM]):

    @classmethod
    def get_target_class(self):
        return DDM
    
    def get_loggable_state(self) -> dict:
        p = self._model._p.get()
        n = self._model._p.n

        if n < 1:
            return None

        s = math.sqrt(p * (1 - p) / n)

        if self._model._p_min is None:
            return {
                "error_probability": p,
                "ps": p + s,
            }

        return {
            "error_probability": p,
            "ps": p + s,
            "warn_threshold": self._model._p_min + self._model.warning_threshold * self._model._s_min,
            "alarm_threshold": self._model._p_min + self._model.drift_threshold * self._model._s_min,
        }

class ADWINAdapter(ModelAdapterBase[ADWIN]):
    
    @classmethod
    def get_target_class(self):
        return ADWIN
    
    def get_loggable_state(self) -> dict:
        return {
            "width": self._model.width,
            "n_detections": self._model.n_detections,
            "variance": self._model.variance,
            "total": self._model.total,
            "estimation": self._model.estimation,
        }
