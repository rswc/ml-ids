from river import ensemble, utils, base
import random

class MOOB(ensemble.BaggingClassifier):

    def __init__(
            self,
            model=None,
            n_models=10,
            seed=None,
            decay_factor=0.9,
    ) -> None:
        super().__init__(model=model, n_models=n_models, seed=seed)
        self.decay_factor = decay_factor
        self._class_priors: dict[base.typing.ClfTarget, float] = {}
        self._random = random.Random(seed)

    def learn_one(self, x, y, **kwargs):
        for model in self:
            for _ in range(utils.random.poisson(1, self._rng)):
                model.learn_one(x, y, **kwargs)
        return self

    def __update_models(self, x: dict, y: base.typing.ClfTarget, **kwargs):
        for label, model in self.classifiers.items():
            if y == label:
                self._class_priors[y] = self.decay_factor * self._class_priors[y] + 1 - self.decay_factor
                model.learn_one(x, 1, **kwargs)

            else:
                self._class_priors[label] *= self.decay_factor
                p = self._class_priors[label] / (1 - self._class_priors[label])

                if self._random.random() < p:
                    model.learn_one(x, -1, **kwargs)
