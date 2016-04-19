import gd
import util
import mlGraphics
from pylab import *
import binary
import linear
import runClassifier
import datasets
from numpy import *

# GD Test

gd.gd(lambda x: x**2, lambda x: 2*x, 10, 10, 0.2)
x, trajectory = gd.gd(lambda x: x**2, lambda x: 2*x, 10, 100, 0.2)
x
plot(trajectory)
show(False)

# Linear Test

f = linear.LinearClassifier({'lossFunction': linear.SquaredLoss(), 'lambda': 0, 'numIter': 100, 'stepSize': 0.5})
runClassifier.trainTestSet(f, datasets.TwoDAxisAligned)
# Training accuracy 0.91, test accuracy 0.86
# f
# w=array([ 2.73466371, -0.29563932])
# mlGraphics.plotLinearClassifier(f, datasets.TwoDAxisAligned.X, datasets.TwoDAxisAligned.Y)
# show(False)