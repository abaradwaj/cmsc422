from numpy import *
from pylab import *
import util, datasets, runClassifier, binary, dt

h = dt.DT({'maxDepth': 1})
print(h)

# print(h.predict(h, X=2))
h.train(datasets.TennisData.X, datasets.TennisData.Y)
print(h)

h = dt.DT({'maxDepth': 2})
h.train(datasets.TennisData.X, datasets.TennisData.Y)
print(h)

h = dt.DT({'maxDepth': 5})
h.train(datasets.TennisData.X, datasets.TennisData.Y)
print(h)

