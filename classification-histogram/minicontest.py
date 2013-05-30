#minicontest.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

import util
import classificationMethod
import mira
import perceptron
import naiveBayes
import math
class contestClassifier(classificationMethod.ClassificationMethod):
    """
    Create any sort of classifier you want. You might copy over one of your
    existing classifiers and improve it.
    """
    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "minicontest"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = 5
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."
        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            Cgrid = [5, 10, 15, 25, 50]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """
        bestK = 0
        maxsofar = 0
        self.xMatrix = trainingData
        self.yMatrix = trainingLabels
        self.bestK = 3
        Cgrid = [5, 10, 15, 25, 50]


    def classify(self, data ):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        """for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())"""
	"""xMat = numpy.array( [tuple(self.xMatrix[i].values()) for i in range(len(self.xMatrix))])
	yMat = numpy.array([ tuple(data[i].values()) for i in range(len(data))])"""	
	x = []
	xScores = []
	for datum in data:
		for i in range(len(self.xMatrix)):
			x.append( self.xMatrix[i] - datum)
		for i in range(len(x)):
			for key in x[i]:
				x[i][key] = x[i][key] ** 2
			xScores.append((i,math.sqrt(x[i].totalCount())))	
		xScores = sorted(xScores, key = lambda x: x[1])
		del xScores[self.bestK:]	
	    	kNearest = [self.yMatrix[labelInd] for (labelInd, eucDist) in xScores]	
	    	kNearestNeighbors = util.Counter()
	    	for label in kNearest:
			kNearestNeighbors[label] += 1		
	    	guesses.append(kNearestNeighbors.argMax())
	"""for i in range(len(yMat)): 
	    datumDist = []
	    diff = xMat - yMat[i]
	    squareDiff = numpy.square(diff)
	    datumDist = [(j,math.sqrt(squareDiff[j].sum())) for j in range(len(squareDiff))]
	    datumDist = sorted(datumDist, key = lambda x: x[1])
	    del datumDist[self.bestK:]
	    kNearest = [self.yMatrix[labelInd] for (labelInd, eucDist) in datumDist]	
	    kNearestNeighbors = util.Counter()
	    for label in kNearest:
		kNearestNeighbors[label] += 1		
	    guesses.append(kNearestNeighbors.argMax())"""
=======
        xMat = numpy.array( [tuple(self.xMatrix[i].values()) for i in range(len(self.xMatrix))])

        yMat = numpy.array([ tuple(data[i].values()) for i in range(len(data))])
        for i in range(len(yMat)):
            datumDist = []
            diff = xMat - yMat[i]
            squareDiff = numpy.square(diff)
            datumDist = [(j,math.sqrt(squareDiff[j].sum())) for j in range(len(squareDiff))]
            datumDist = sorted(datumDist, key = lambda x: x[1])
            del datumDist[self.bestK:]
            kNearest = [self.yMatrix[labelInd] for (labelInd, eucDist) in datumDist]
            kNearestNeighbors = util.Counter()
            for label in kNearest:
                kNearestNeighbors[label] += 1
            guesses.append(kNearestNeighbors.argMax())
>>>>>>> 5a58624e790ca85f7cf32814b03ffb7a10d91b43
        return guesses
