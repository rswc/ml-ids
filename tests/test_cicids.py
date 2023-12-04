from cicids import CICIDS2017
CICIDS_TEST = False

class TestCICIDS:

    def test_number_of_samples(self):
        if CICIDS_TEST:
            dataset = CICIDS2017()
            samples = 0

            for x, y in iter(dataset):
                samples += 1

            assert samples == dataset.n_samples
        else:
            pass

    def test_number_of_features(self):
        if CICIDS_TEST:
            dataset = CICIDS2017()
            x, y = next(iter(dataset))

            assert len(x) == dataset.n_features
        else:
            pass

    def test_number_of_classes(self):
        if CICIDS_TEST:
            dataset = CICIDS2017()
            classes = []

            for x, y in iter(dataset):
                if y not in classes:
                    classes.append(y)

            assert len(classes) == dataset.n_classes
        else:
            pass
