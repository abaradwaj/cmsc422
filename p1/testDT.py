from numpy import *
from pylab import *
import util, datasets, runClassifier, binary, dt

h = dt.DT({'maxDepth': 1})
print(h)

print(h.predict(self, 2))
h.train(datasets.TennisData.X, datasets.TennisData.Y)
print(h)

