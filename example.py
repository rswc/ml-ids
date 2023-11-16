# from cicids import CICIDS2017
from river.datasets.sms_spam import SMSSpam

# dataset = CICIDS2017()
dataset = SMSSpam()
print(dataset)

x, y = next(iter(dataset))
print(x)
print(y)

from river import tree
from river import metrics
from river.metrics.base import Metrics
from framework import ExperimentRunner

model = tree.HoeffdingTreeClassifier()

runner = ExperimentRunner(model, dataset, Metrics([metrics.Precision(), metrics.Recall()]), "./out", project="ml-ids")

runner.run()
