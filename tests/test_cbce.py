from cbce import CBCE
from river import linear_model
from river.drift.binary import DDM

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

        assert model.predict_one({"x": -7}) == "neg", "Misclassified negative sample"
        assert model.predict_one({"x": 7}) == "pos", "Misclassified positive sample"
    
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
        
        assert model.predict_proba_one({"x": 5})["majority"] + 1e-7 > 1, "Failed to predict majority class, despite no others being available"

        model.learn_one({"x": 6}, "B")

        assert model.predict_proba_one({"x": 6})["majority"] < 1.0, "Failed to adjust prediction after receiving second sample"

    def test_class_disappearance(self):
        """
        Classes which haven't been seen in a long time should be deactivated, and reactivated
        once new samples belonging to them arrive.
        """

        model = CBCE(linear_model.LogisticRegression(), disappearance_threshold=0.9**100)

        DATA = [
            ({"x": 1}, "A"),
            ({"x": 2}, "A"),
            ({"x": 3}, "A"),
            ({"x": -2}, "B"),
            ({"x": -4}, "B"),
            ({"x": -2}, "B"),
        ]

        for x, y in DATA:
            model.learn_one(x, y)

        assert model.predict_proba_one({"x": 9})["A"] > 0.5, "Failed to learn first class"

        DATA = [({"x": 2}, "B")] * 100

        for x, y in DATA:
            model.learn_one(x, y)

        assert model._class_priors["A"] == 0, "Provided wrong prior value for disappeared class"
        assert "A" not in model.predict_proba_one({"x": 9}), "Provided disappeared class during prediction"

        DATA = [
            ({"x": 1}, "A"),
            ({"x": -2}, "B"),
            ({"x": 2}, "A"),
        ]

        for x, y in DATA:
            model.learn_one(x, y)

        assert model._class_priors["A"] > 0, "Provided wrong prior for reappeared class"
        assert "A" in model.predict_proba_one({"x": 9}), "Failed to provide reappeared class during prediction"

    def test_drift_detector(self):
        """
        The model should react to concept drift in the classes it tracks.
        """

        model = CBCE(linear_model.LogisticRegression(), drift_detector=DDM())

        DATA = [
            ({"x": 1}, "A"),
            ({"x": 2}, "A"),
            ({"x": 3}, "A"),
            ({"x": -2}, "B"),
            ({"x": -4}, "B"),
            ({"x": -2}, "B"),
        ]

        for x, y in DATA:
            model.learn_one(x, y)

        assert model.predict_proba_one({"x": 9})["A"] > 0.5, "Failed to learn first class"

        VALUE_MUL = [1, -1]
        LABEL = ["A", "B"]
        DATA = [({"x": i * VALUE_MUL[i & 1]}, LABEL[i & 1]) for i in range(100)]

        num_classes = 0
        for x, y in DATA:
            model.learn_one(x, y)
            num_classes += len(model._class_priors)
        
        # According to the paper, whenever drift is detected in a certain class,
        # the model for that class has to be reinitialized. At the time of writing, 
        # in our implementation this means removing that class completely
        assert num_classes < 200, "Failed to evict models for classes under drift"
