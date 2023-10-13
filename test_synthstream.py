
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
letter_recognition = fetch_ucirepo(id=59) 

X = letter_recognition.data.features 
y = letter_recognition.data.targets 

from synthstream import SyntheticStream, EmptyClassStream
from collections import defaultdict
import math

examples = defaultdict(list)

n_examples = 0
for ((i, x_row), (_, y_row)) in zip(X.iterrows(), y.iterrows()):
    label =  y_row.to_dict()['lettr']
    if label in ['A', 'B', 'C']:
        examples[label].append(x_row.to_dict())
        n_examples += 1

ss = SyntheticStream()
ss.add_class(
    label='A',
    prob_sampler=lambda t: 1,
    x_generator=(x for x in examples['A'])
)
ss.add_class(
    label='B',
    prob_sampler=lambda t: 1,
    x_generator=(x for x in examples['B'])
)

c_prob_sampler = lambda t: 1.0/(1.0 + math.exp(-((t - n_examples/2) / (n_examples/8))))
ss.add_class(
    label='C',
    prob_sampler=c_prob_sampler,
    x_generator=(x for x in examples['C'])
)

cnts = defaultdict(int)
idx = 0

try:
    for x,y in ss:
        idx += 1
        # print(n_examples - idx)
        cnts[y] += 1
except EmptyClassStream:
    pass