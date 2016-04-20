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
f = linear.LinearClassifier({'lossFunction': linear.LogisticLoss(), 'lambda': 1, 'numIter': 100, 'stepSize': 0.5})
runClassifier.trainTestSet(f, datasets.WineDataBinary)

large = [   0.606423261902,
            0.689199007903,
            0.710890552154,
            0.770124769156,
            0.883289753118  ]

small = [   -1.1695212164,
            -0.765309390643,
            -0.683593167789,
            -0.629590728143,
            -0.532191672468 ]

# f.weights.sort()
# print(f.weights)

print("Printing WineDataBinary.words:")
print(datasets.WineDataBinary.words)
print("Words array length: %d\n" % len(datasets.WineDataBinary.words))

print("Printing small items:")
for i,j in enumerate(f.weights):
    for smallItem in small:
        if ("%.8f" % j) == ("%.8f" % smallItem):
            print datasets.WineDataBinary.words[i]

print("\n")
print("Printing large items:")
for i,j in enumerate(f.weights):
    for largeItem in large:
        if ("%.8f" % j) == ("%.8f" % largeItem):
            print datasets.WineDataBinary.words[i]
#
# print("Hinge")
# f = linear.LinearClassifier({'lossFunction': linear.HingeLoss(), 'lambda': 1, 'numIter': 100, 'stepSize': 0.5})
# runClassifier.trainTestSet(f, datasets.WineDataBinary)
# print(f)
#
# print("Squared")
# f = linear.LinearClassifier({'lossFunction': linear.SquaredLoss(), 'lambda': 1, 'numIter': 100, 'stepSize': 0.05})
# runClassifier.trainTestSet(f, datasets.WineDataBinary)
# print(f)