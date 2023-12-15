from river import tree
from math import isclose
from moob import MOOB


class TestMOOB:

    def test_default_oversampling(self):
        model = MOOB(tree.HoeffdingTreeClassifier(), seed=42)
        assert model.resampling == "oversampling"

    def test_class_priors(self):
        model = MOOB(tree.HoeffdingTreeClassifier(), seed=42)

        DATA = [
            ({'x': 1}, 'A'),
            ({'x': 1}, 'A'),
            ({'x': 0}, 'B'),
            ({'x': 0}, 'B'),
            ({'x': 0}, 'B'),
            ({'x': 0}, 'B'),
            ({'x': 0}, 'B'),
            ({'x': 0}, 'B'),
            ({'x': 0}, 'B'),
            ({'x': 0}, 'B'),
        ]

        for x, y in DATA:
            model.learn_one(x, y)

        assert len(model._class_priors) == 2
        assert round(model._class_priors['A'], 2) < 0.2

    def test_oversampling(self):
        model = MOOB(tree.HoeffdingTreeClassifier(), resampling="oversampling", seed=42)

        DATA = [
            ({'x': 1}, 'A'),
            ({'x': 0}, 'B'),
            ({'x': 0}, 'B'),
            ({'x': 0}, 'B'),
            ({'x': 0}, 'B'),
            ({'x': 0}, 'B'),
            ({'x': 0}, 'B'),
            ({'x': 0}, 'B'),
            ({'x': 0}, 'B'),
            ({'x': 1}, 'A'),
            ({'x': 1}, 'A'),
        ]

        for x, y in DATA:
            model.learn_one(x, y)

        assert isclose(model.predict_proba_one({'x': 1})['A'], 0.5, abs_tol=0.05)
        assert isclose(model.predict_proba_one({'x': 1})['B'], 0.5, abs_tol=0.05)

    def test_undersampling(self):
        model = MOOB(tree.HoeffdingTreeClassifier(), resampling="undersampling", seed=42)

        DATA = [
            ({'x': 1}, 'A'),
            ({'x': 0}, 'B'),
            ({'x': 0}, 'B'),
            ({'x': 0}, 'B'),
            ({'x': 0}, 'B'),
            ({'x': 0}, 'B'),
            ({'x': 0}, 'B'),
            ({'x': 0}, 'B'),
            ({'x': 0}, 'B'),
            ({'x': 1}, 'A'),
            ({'x': 1}, 'A'),
        ]

        for x, y in DATA:
            model.learn_one(x, y)

        assert isclose(model.predict_proba_one({'x': 1})['A'], 0.5, abs_tol=0.05)
        assert isclose(model.predict_proba_one({'x': 1})['B'], 0.5, abs_tol=0.05)
