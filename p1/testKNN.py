from numpy import *
from pylab import *
import util, datasets, runClassifier, binary, knn

runClassifier.trainTestSet(knn.KNN({'isKNN': False, 'eps': 0.5}), datasets.TennisData)
runClassifier.trainTestSet(knn.KNN({'isKNN': False, 'eps': 1.0}), datasets.TennisData)
runClassifier.trainTestSet(knn.KNN({'isKNN': False, 'eps': 2.0}), datasets.TennisData)

runClassifier.trainTestSet(knn.KNN({'isKNN': True, 'K': 1}), datasets.TennisData)
runClassifier.trainTestSet(knn.KNN({'isKNN': True, 'K': 3}), datasets.TennisData)
runClassifier.trainTestSet(knn.KNN({'isKNN': True, 'K': 5}), datasets.TennisData)

runClassifier.trainTestSet(knn.KNN({'isKNN': False, 'eps': 6.0}), datasets.DigitData)
runClassifier.trainTestSet(knn.KNN({'isKNN': False, 'eps': 8.0}), datasets.DigitData)
runClassifier.trainTestSet(knn.KNN({'isKNN': False, 'eps': 10.0}), datasets.DigitData)

runClassifier.trainTestSet(knn.KNN({'isKNN': True, 'K': 1}), datasets.DigitData)
runClassifier.trainTestSet(knn.KNN({'isKNN': True, 'K': 3}), datasets.DigitData)
runClassifier.trainTestSet(knn.KNN({'isKNN': True, 'K': 5}), datasets.DigitData)

#knnCurve = runClassifier.hyperparamCurveSet(knn.KNN({'isKNN':True, 'K':1}), 'K', range(0, 21), datasets.DigitData)
#runClassifier.plotCurve('KNN vs. train/test Accuracy', knnCurve)

#epsCurve = runClassifier.hyperparamCurveSet(knn.KNN({'isKNN':False, 'eps':0.5}), 'eps', arange(0,14.5,0.5), datasets.DigitData)
#runClassifier.plotCurve('Epsilon Ball vs. train/test accuracy', epsCurve)

learning = runClassifier.learningCurveSet(knn.KNN({'isKNN':True, 'K':5}), datasets.DigitData)
runClassifier.plotCurve('Training examples vs. train/test accuracy', learning)

