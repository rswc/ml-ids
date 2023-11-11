import random
from river import base

class CBCE(base.Wrapper, base.Classifier):

    def __init__(self, classifier: base.Classifier, decay_factor: float = 0.9, disappearance_threshold = 1.7e-46) -> None:
        self.classifier = classifier
        self.classifiers: dict[base.typing.ClfTarget, base.Classifier] = {}
        self.inactive_classifiers: dict[base.typing.ClfTarget, base.Classifier] = {}
        self._class_priors: dict[base.typing.ClfTarget, float] = {}
        self._sample_buffer: dict[base.typing.ClfTarget, list[dict]] = {}
        self.decay_factor = decay_factor
        self.disappearance_threshold = disappearance_threshold

    @property
    def _wrapped_model(self):
        return self.classifier
    
    @property
    def _multiclass(self):
        return True
    
    def learn_one(self, x: dict, y: base.typing.ClfTarget, **kwargs) -> base.Classifier:
        for label in self._sample_buffer:
            self._sample_buffer[label].append(x)

        if y not in self.classifiers and y not in self.inactive_classifiers:
            if y not in self._sample_buffer:
                # First sample arrived, start buffering
                self._sample_buffer[y] = [x]
                self._class_priors[y] = 0
            else:
                # Second sample arrived, initilize model
                buffer_len = len(self._sample_buffer[y])

                model: base.Classifier = self.classifier.clone()
                labels = [1 if i == 0 or i == buffer_len - 1 else -1 for i in range(buffer_len)]
                for buffered_x, buffered_y in zip(self._sample_buffer[y], labels):
                    model.learn_one(buffered_x, buffered_y, **kwargs)
                    
                self.classifiers[y] = model

                # Sample buffer contains the two positive samples, hence the -1
                self._class_priors[y] = 1 / (buffer_len - 1)

                # Stop buffering
                del self._sample_buffer[y]

        #TODO: class reoccurrence
        
        # Class disappearance
        disappeared_labels = []
        for label, model in self.classifiers.items():
            if self._class_priors[label] < self.disappearance_threshold:
                self.inactive_classifiers[label] = model
                self._class_priors[label] = 0
                disappeared_labels.append(label)
            
        for label in disappeared_labels:
            del self.classifiers[label]

        self.__updateCBModels(x, y, **kwargs)

        return self
    
    def __updateCBModels(self, x: dict, y: base.typing.ClfTarget, **kwargs):
        for label, model in self.classifiers.items():
            if y == label:
                self._class_priors[y] = self.decay_factor * self._class_priors[y] + 1 - self.decay_factor
                model.learn_one(x, 1, **kwargs)
            else:
                self._class_priors[label] *= self.decay_factor
                p = self._class_priors[label] / (1 - self._class_priors[label])

                if random.random() < p:
                    model.learn_one(x, -1, **kwargs)

    def predict_proba_one(self, x: dict, **kwargs) -> dict[base.typing.ClfTarget, float]:
        y_pred = {}

        for label, model in self.classifiers.items():
            y_pred[label] = model.predict_proba_one(x, **kwargs)[1]

        total = sum(y_pred.values())
        
        if total:
            return {label: score / total for label, score in y_pred.items()}
        return {label: 1 / len(y_pred) for label in y_pred.keys()}
