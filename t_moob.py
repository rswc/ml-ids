from river.ensemble import BaggingClassifier
from river.tree import HoeffdingTreeClassifier
from moob import MOOB
from cicids import CICIDS2017

dataset = CICIDS2017("subset_test_ready.csv")

moob = MOOB(HoeffdingTreeClassifier(), n_models=10, seed=42)
ob = BaggingClassifier(HoeffdingTreeClassifier(), n_models=10, seed=42)


