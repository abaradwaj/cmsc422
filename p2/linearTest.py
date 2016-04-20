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

# gd.gd(lambda x: x**2, lambda x: 2*x, 10, 10, 0.2)
# x, trajectory = gd.gd(lambda x: x**2, lambda x: 2*x, 10, 100, 0.2)
# print(x)
# plot(trajectory)
# show(False)

# Linear Test

# SquaredLoss
# f = linear.LinearClassifier({'lossFunction': linear.SquaredLoss(), 'lambda': 0, 'numIter': 100, 'stepSize': 0.5})
# runClassifier.trainTestSet(f, datasets.TwoDAxisAligned)
# # Training accuracy 0.91, test accuracy 0.86
# print(f)
# # w=array([ 2.73466371, -0.29563932])
# mlGraphics.plotLinearClassifier(f, datasets.TwoDAxisAligned.X, datasets.TwoDAxisAligned.Y)
# show(False)
#
# f = linear.LinearClassifier({'lossFunction': linear.SquaredLoss(), 'lambda': 10, 'numIter': 100, 'stepSize': 0.5})
# runClassifier.trainTestSet(f, datasets.TwoDAxisAligned)
# # Training accuracy 0.9, test accuracy 0.86
# print(f)
# # w=array([ 1.30221546, -0.06764756])
#
# # HingeLoss
# f = linear.LinearClassifier({'lossFunction': linear.HingeLoss(), 'lambda': 1, 'numIter': 100, 'stepSize': 0.5})
# runClassifier.trainTestSet(f, datasets.TwoDDiagonal)
# # Training accuracy 0.98, test accuracy 0.86
# print(f)
# # w=array([ 1.17110065,  4.67288657])
#
# # LogisticLoss
#
# f = linear.LinearClassifier({'lossFunction': linear.LogisticLoss(), 'lambda': 10, 'numIter': 100, 'stepSize': 0.5})
# runClassifier.trainTestSet(f, datasets.TwoDDiagonal)
# # Training accuracy 0.99, test accuracy 0.86
# print(f)
# # w=array([ 0.29809083,  1.01287561])

# WU5
print("Logistic:")
# f = linear.LinearClassifier({'lossFunction': linear.LogisticLoss(), 'lambda': 1, 'numIter': 100, 'stepSize': 0.5})
# runClassifier.trainTestSet(f, datasets.WineDataBinary)
# print(f)

print("Hinge")
# f = linear.LinearClassifier({'lossFunction': linear.HingeLoss(), 'lambda': 1, 'numIter': 100, 'stepSize': 0.5})
# runClassifier.trainTestSet(f, datasets.WineDataBinary)
# print(f)

print("Squared")
f = linear.LinearClassifier({'lossFunction': linear.SquaredLoss(), 'lambda': 1, 'numIter': 100, 'stepSize': 0.05})
runClassifier.trainTestSet(f, datasets.WineDataBinary)
print(f)