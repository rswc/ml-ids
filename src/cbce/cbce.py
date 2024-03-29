import random
from river import base
from river.base.drift_detector import BinaryDriftDetector

class NoDrift(base.BinaryDriftAndWarningDetector):
    
    def update(self, x: bool) -> BinaryDriftDetector:
        return self

class CBCE(base.Wrapper, base.Classifier):

    def __init__(
            self,
            classifier: base.Classifier,
            drift_detector: base.BinaryDriftAndWarningDetector = NoDrift(),
            decay_factor: float = 0.9,
            disappearance_threshold: float = 0.9 ** 1000,
            seed: int = None,
            reset_buffer_on_warning_lowered: bool = True
        ) -> None:
        self.classifier = classifier
        self.drift_detector = drift_detector

        self.classifiers: dict[base.typing.ClfTarget, base.Classifier] = {}
        self.inactive_classifiers: dict[base.typing.ClfTarget, base.Classifier] = {}
        self.drift_detectors: dict[base.typing.ClfTarget, base.BinaryDriftAndWarningDetector] = {}
        
        self.decay_factor = decay_factor
        self.disappearance_threshold = disappearance_threshold
        
        self._class_priors: dict[base.typing.ClfTarget, float] = {}
        self._sample_buffer: dict[base.typing.ClfTarget, list[dict]] = {}
        self._random = random.Random(seed)
        self.seed = seed

        self.__classes_to_reset = []
        self.reset_buffer_on_warning_lowered = reset_buffer_on_warning_lowered

    @property
    def _wrapped_model(self):
        return self.classifier
    
    @property
    def _multiclass(self):
        return True
    
    def is_active(self, cls: base.typing.ClfTarget) -> bool:
        return self._class_priors.get(cls, 0) > 0
    
    def learn_one(self, x: dict, y: base.typing.ClfTarget, **kwargs) -> base.Classifier:
        for label in self._sample_buffer:
            self._sample_buffer[label].append(x)

        # Class emergence
        if y not in self._class_priors:
            if y not in self._sample_buffer:
                # First sample arrived, start buffering
                self._sample_buffer[y] = [x]

                self.drift_detectors[y] = self.drift_detector.clone()

            else:
                # Second sample arrived, initilize model
                self.classifiers[y] = self.__init_cb_model(y, **kwargs)

                # Sample buffer contains the two positive samples, hence the -1
                self._class_priors[y] = 1 / (len(self._sample_buffer[y]) - 1)

                # Stop buffering
                del self._sample_buffer[y]
        
        # Class reoccurrence
        elif self._class_priors[y] == 0:
            if y not in self._sample_buffer:
                # First sample arrived, start buffering
                self._sample_buffer[y] = [x]

                # Activate classifier
                self.classifiers[y] = self.inactive_classifiers[y]
                del self.inactive_classifiers[y]
            else:
                # Second sample arrived, initilize model
                buffer_len = len(self._sample_buffer[y])

                # Sample buffer contains the two positive samples, hence the -1
                self._class_priors[y] = 1 / (buffer_len - 1)

                # Stop buffering
                del self._sample_buffer[y]
        
        # Class disappearance
        disappeared_labels = []
        for label, model in self.classifiers.items():
            if self._class_priors[label] < self.disappearance_threshold and label not in self._sample_buffer:
                self.inactive_classifiers[label] = model
                self._class_priors[label] = 0
                disappeared_labels.append(label)
            
        for label in disappeared_labels:
            del self.classifiers[label]

        self.__update_cb_models(x, y, **kwargs)

        for cls in self.__classes_to_reset:
            self.__reset_model(cls)

        self.__classes_to_reset.clear()

        return self
    
    def __update_cb_models(self, x: dict, y: base.typing.ClfTarget, **kwargs):
        for label, model in self.classifiers.items():
            if y == label:
                self._class_priors[y] = self.decay_factor * self._class_priors[y] + 1 - self.decay_factor
                model.learn_one(x, 1, **kwargs)
                self.__update_drift_detector(x, label, **kwargs)

            else:
                self._class_priors[label] *= self.decay_factor
                p = self._class_priors[label] / (1 - self._class_priors[label])

                if self._random.random() < p:
                    model.learn_one(x, -1, **kwargs)
                    self.__update_drift_detector(x, label, **kwargs)
    
    def __update_drift_detector(self, x: dict, y: base.typing.ClfTarget, **kwargs):
        pred = self.predict_one(x)
        self.drift_detectors[y].update(pred != y)

        if self.drift_detectors[y].warning_detected and y not in self._sample_buffer:
            self._sample_buffer[y] = [x]

        elif y in self._sample_buffer and not self.drift_detectors[y].warning_detected and self.reset_buffer_on_warning_lowered:
            del self._sample_buffer[y]

        if self.drift_detectors[y].drift_detected:
            if y in self._sample_buffer:
                self.drift_detectors[y]._reset()

                model = self.__init_cb_model(y, **kwargs)
                    
                if self.is_active(y):
                    self.classifiers[y] = model
                else:
                    self.inactive_classifiers[y] = model

                # Stop buffering
                del self._sample_buffer[y]
            
            else:
                self.__classes_to_reset.append(y)

    def __reset_model(self, y: base.typing.ClfTarget):
        self.classifiers.pop(y, None)
        self.inactive_classifiers.pop(y, None)
        self.drift_detectors.pop(y, None)
        self._class_priors.pop(y, None)
        self._sample_buffer.pop(y, None)

    def __init_cb_model(self, y: base.typing.ClfTarget, **kwargs):
        buffer_len = len(self._sample_buffer[y])

        model: base.Classifier = self.classifier.clone()

        labels = [1 if i == 0 or i == buffer_len - 1 else -1 for i in range(buffer_len)]
        for buffered_x, buffered_y in zip(self._sample_buffer[y], labels):
            model.learn_one(buffered_x, buffered_y, **kwargs)

        return model

    def predict_proba_one(self, x: dict, **kwargs) -> dict[base.typing.ClfTarget, float]:
        y_pred = {}

        for label, model in self.classifiers.items():
            y_pred[label] = model.predict_proba_one(x, **kwargs)[1]

        total = sum(y_pred.values())
        
        if total:
            return {label: score / total for label, score in y_pred.items()}
        return {label: 1 / len(y_pred) for label in y_pred.keys()}
