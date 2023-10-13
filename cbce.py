import random
from river import base

class CBCE(base.Wrapper, base.Classifier):

    def __init__(self, classifier: base.Classifier, decay_factor: float = 0.9) -> None:
        self.classifier = classifier
        self.classifiers: dict[base.typing.ClfTarget, base.Classifier] = {}
        self._class_priors: dict[base.typing.ClfTarget, float] = {}
        self._sample_buffer: dict[base.typing.ClfTarget, list[dict]] = {}
        self.decay_factor = decay_factor

    @property
    def _wrapped_model(self):
        return self.classifier
    
    @property
    def _multiclass(self):
        return True
    
    def learn_one(self, x: dict, y: base.typing.ClfTarget, **kwargs) -> base.Classifier:
        for label in self._sample_buffer:
            self._sample_buffer[label].append(x)

        if y not in self.classifiers:
            if y not in self._sample_buffer:
                # First sample arrived, start buffering
                self._sample_buffer[y] = [x]
            else:
                # Second sample arrived, initilize model
                buffer_len = len(self._sample_buffer[y])

                model = self.classifier.clone()
                labels = [1 if i == 0 or i == buffer_len - 1 else -1 for i in range(buffer_len)]
                for x, y in zip(self._sample_buffer[y], labels):
                    model.learn_one(x, y, **kwargs)
                    
                self.classifiers[y] = model

                #TODO: class reoccurrence
                #TODO: class disappearance

                # Sample buffer contains the two positive samples, hence the -1
                self._class_priors[y] = 1 / (buffer_len - 1)

                del self._sample_buffer[y]

        self.__updateCBModels(x, y, **kwargs)

        return self
    
    def __updateCBModels(self, x: dict, y: base.typing.ClfTarget, **kwargs):
        for label, model in self.classifiers.items():
            if y == label:
                self._class_priors[y] = self.decay_factor * self._class_priors[y] + 1 - self.decay_factor
                model.learn_one(x, 1, **kwargs)
            else:
                self._class_priors[y] *= self.decay_factor
                p = self._class_priors[y] / (1 - self._class_priors[y])

                if random.random() < p:
                    model.learn_one(x, -1, **kwargs)


    def predict_proba_one(self, x: dict) -> dict[base.typing.ClfTarget, float]:
        raise NotImplementedError()

    def predict_one(self, x: dict) -> base.typing.ClfTarget | None:
        raise NotImplementedError()    
