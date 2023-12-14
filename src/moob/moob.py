from river import ensemble, utils, base


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
        self.class_priors: dict[base.typing.ClfTarget, float] = {}

    def learn_one(self, x, y, **kwargs):
        if y not in self.class_priors:
            self.class_priors[y] = 0

        for cls in self.class_priors:
            self.class_priors[cls] = self.decay_factor * self.class_priors[cls] + ((1 - self.decay_factor) * (y == cls))

        w_max = max(self.class_priors.values())

        for model in self:
            for _ in range(utils.random.poisson(w_max / self.class_priors[y], self._rng)):
                model.learn_one(x, y, **kwargs)
        return self
