from river import datasets
from cicids import CICIDS2017
from tqdm import tqdm

dataset = CICIDS2017()
print(dataset)

x, y = next(iter(dataset))
print(x)
print(y)

from river import tree

model = tree.HoeffdingTreeClassifier()
print(model.predict_one(x))


print(model.learn_one(x, y))
print(model.predict_proba_one(x))


from river import metrics
metric = metrics.ClassificationReport()

for x, y in tqdm(dataset):
    y_pred = model.predict_one(x)
    model.learn_one(x, y)
    if y_pred is not None:
        metric.update(y, y_pred)

print(metric)

from river import evaluate
res = evaluate.progressive_val_score(dataset, model, metric)
print(res)