# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
    """
    See the project description for the specifications of the Naive Bayes classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "naivebayes"
        self.k = 1 # this is the smoothing parameter, ** use it in your train method **
        self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **

    def setSmoothing(self, k):
        """
        This is used by the main method to change the smoothing parameter before training.
        Do not modify this method.
        """
        self.k = k

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Outside shell to call your method. Do not modify this method.
        """

        # might be useful in your code later...
        # this is a list of all features in the training set.
        self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));

        if (self.automaticTuning):
            kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
        else:
            kgrid = [self.k]

        self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
        """
        Trains the classifier by collecting counts over the training data, and
        stores the Laplace smoothed estimates so that they can be used to classify.
        Evaluate each value of k in kgrid to choose the smoothing parameter
        that gives the best accuracy on the held-out validationData.

        trainingData and validationData are lists of feature Counters.  The corresponding
        label lists contain the correct label for each datum.

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """
        maxcount = 0
        finalpfy = None
        finalpy = None
        for k in kgrid:
            """ P(Y) """
            labelcounter = util.Counter()
            """ P(F|Y) """
            featureprob = util.Counter()

            seenFeatureValues = {}

            for label in trainingLabels:
                labelcounter[label] += 1.0/len(trainingLabels)
            i = 0
            for datum in trainingData:
                for feat, value in datum.iteritems():
                    featureprob[(feat, value, trainingLabels[i])] += 1
                    if feat not in seenFeatureValues:
                        seenFeatureValues[feat] = [value]
                    elif value not in seenFeatureValues[feat]:
                        seenFeatureValues[feat].append(value)
                i += 1

            for label in self.legalLabels:
                for feature in self.features:
                    for val in seenFeatureValues[feature]:
                        featureprob[(feature, val, label)] += k

            for label in self.legalLabels:
                for feature in self.features:
                    vallist = seenFeatureValues[feature]
                    s = sum([featureprob[(feature, val, label)] for val in vallist])
                    for val in vallist:
                        featureprob[(feature, val, label)] /= float(s)

            self.pfy = featureprob
            self.py = labelcounter
            vd = self.classify(validationData)
            count = 0
            for i, item in enumerate(vd):
                if item == validationLabels[i]:
                    count += 1
            if count > maxcount:
                finalpfy = self.pfy
                finalpy = self.py
                maxcount = count
        self.pfy = finalpfy
        self.py = finalpy



    def classify(self, testData):
        """
        Classify the data based on the posterior distribution over labels.

        You shouldn't modify this method.
        """
        guesses = []
        self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum)
            guesses.append(posterior.argMax())
            self.posteriors.append(posterior)
        return guesses

    def calculateLogJointProbabilities(self, datum):
        """
        Returns the log-joint distribution over legal labels and the datum.
        Each log-probability should be stored in the log-joint counter, e.g.
        logJoint[3] = <Estimate of log( P(Label = 3, datum) )>

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """
        import math
        logJoint = util.Counter()
        for item in self.legalLabels:
            logJoint[item] = math.log(self.py[item])
            for feat, val in datum.iteritems():
                prod = self.pfy[(feat, val, item)]
                if prod != 0:
                    logJoint[item] += math.log(prod)
        return logJoint


    def findHighOddsFeatures(self, label1, label2):
        """
        Returns the 100 best features for the odds ratio:
                P(feature=1 | label1)/P(feature=1 | label2)

        Note: you may find 'self.features' a useful way to loop through all possible features
        """
        featuresOdds = []

        for feature in self.features:
            odds = self.pfy[(feature, 1, label1)]/self.pfy[(feature, 1, label2)]
            featuresOdds.append((odds, feature))

        return [b for a,b in sorted(featuresOdds, reverse=True)[:100]]
