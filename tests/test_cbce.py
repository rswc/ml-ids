from cbce import CBCE
from river import linear_model

class TestCBCE:

    def test_learn_classify_sign(self):
        """
        Model based on Logistic Regression should be able to learn a basic binary problem.
        """
        model = CBCE(linear_model.LogisticRegression())
        
        DATA = [
            ({"x": 1}, "pos"),
            ({"x": -3}, "neg"),
            ({"x": 3}, "pos"),
            ({"x": -1}, "neg"),
            ({"x": -2}, "neg"),
            ({"x": 2}, "pos")
        ]

        for x, y in DATA:
            model.learn_one(x, y)

        assert model.predict_one({"x": -7}) == "neg"
        assert model.predict_one({"x": 7}) == "pos"
    
    def test_dont_activate_class_until_second_sample(self):
        """
        Receiving the first sample of a novel class should have no impact on the predictions.
        """

        model = CBCE(linear_model.LogisticRegression())

        DATA = [
            ({"x": 1}, "majority"),
            ({"x": 2}, "majority"),
            ({"x": 3}, "A"),
            ({"x": 1}, "majority"),
            ({"x": 5}, "B"),
            ({"x": 6}, "C"),
            ({"x": 2}, "majority"),
            ({"x": 1}, "majority"),
        ]

        for x, y in DATA:
            model.learn_one(x, y)
        
        assert model.predict_proba_one({"x": 5})["majority"] + 1e-7 > 1

        model.learn_one({"x": 6}, "B")

        assert model.predict_proba_one({"x": 6})["majority"] < 1.0


