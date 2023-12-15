from river import ensemble, utils, base


class ResamplingBaggingClassifier(ensemble.BaggingClassifier):

    def __init__(
            self,
            model=None,
            n_models=10,
            seed=None,
            decay_factor=0.9,
            resampling="oversampling"
    ) -> None:
        super().__init__(model=model, n_models=n_models, seed=seed)
        self.decay_factor: float = decay_factor
        self._class_priors: dict[base.typing.ClfTarget, float] = {}
        self.resampling: str = resampling

    def learn_one(self, x, y, **kwargs):
        if y not in self._class_priors:
            self._class_priors[y] = 0

        for cls in self._class_priors:
            self._class_priors[cls] = self.decay_factor * self._class_priors[cls] + ((1 - self.decay_factor) * (y == cls))

        if self.resampling == "undersampling":
            min_prior = min(self._class_priors.values())
            for model in self:
                for _ in range(utils.random.poisson(min_prior / self._class_priors[y], self._rng)):
                    model.learn_one(x, y, **kwargs)
        else:
            max_prior = max(self._class_priors.values())
            for model in self:
                for _ in range(utils.random.poisson(max_prior / self._class_priors[y], self._rng)):
                    model.learn_one(x, y, **kwargs)
        return self
