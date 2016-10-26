#!usr/bin/python
# -*- coding: utf-8 -*-
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet():
	dataMat = []; labelMat = [];
	fr = open('testSet.txt')
	for line in fr.readlines():
		lineArr = line.strip().split()
		dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
		labelMat.append(int(lineArr[2]))
	return dataMat, labelMat

def sigmoid(inX):
	return 1.0 / (1 + exp(-inX))

def gradAscent(dataArr, classLabels):
	dataMatrix = mat(dataArr)
	labelMat = mat(classLabels).transpose()
	m, n = shape(dataMatrix)
	alpha = 0.001
	maxCycles = 500
	weights = ones((n, 1))
	for k in range(maxCycles):
		h = sigmoid(dataMatrix * weights)
		error = (labelMat - h)
		weights = weights + alpha * dataMatrix.transpose()* error  #vectorization in gradAscent
	return weights

def stocGradAscent0(dataMatrix, classLabels):
	m, n = shape(dataMatrix)
	alpha = 0.01
	weights = ones(n)
	for i in range(m):
		h = sigmoid(sum(dataMatrix[i] * weights))
		error = classLabels[i] - h
		weights = weights + alpha * error * dataMatrix[i]
	return mat(weights).transpose()

def stocGradAscent1(dataMatrix, classLabels, numIter = 150):
	m, n = shape(dataMatrix)
	weights = ones(n)
	for j in range(numIter):
		dataIndex = range(m)
		for i in range(m):
			alpha = 4 / (1.0 + j + i) + 0.01
			randIndex = int(random.uniform(0, len(dataIndex)))
			h = sigmoid(sum(dataMatrix[randIndex] * weights))
			error = classLabels[randIndex] - h
			weights = weights + alpha * error * dataMatrix[randIndex]
			del(dataIndex[randIndex])
		print weights[0], weights[1], weights[2]
	return mat(weights).transpose()

def plotBestFit(dataArr, classLabels, wei):
	weights = wei.getA()
	dataArr = array(dataArr)
	n = shape(dataArr)[0]
	xcord1 = []; ycord1 = []
	xcord2 = []; ycord2 = []
	for i in range(n):
		if int(classLabels[i]) == 1:
			xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i, 2])
		else:
			xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1, ycord1, s = 30, c = 'red', marker = 's')
	ax.scatter(xcord2, ycord2, s = 30, c = 'green')
	x = arange(-3.0, 3.0, 0.1)
	y = (-weights[0] - weights[1] * x) / weights[2]
	ax.plot(x, y)
	plt.xlabel('X1'); plt.ylabel('X2')
	plt.show()

def classifyVector(inX, weights):
	prob = sigmoid(sum(inX * weights))
	if prob > 0.5 : return 1.0
	else : return 0.0

def colicTest():
	frTrain = open('horseColicTraining.txt')
	frTest = open('horseColicTest.txt')
	trainingSet = []; trainingLabels = []
	for line in frTrain.readlines():
		currline = line.strip().split('\t')
		lineArr = []
		for i in range(21):
			lineArr.append(float(currline[i]))
		trainingSet.append(lineArr)
		trainingLabels.append(float(currline[21]))
		
	trainingWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)

	errorCount = 0; numTestVec = 0.0
	for line in frTest.readlines():
		numTestVec += 1.0
		currline = line.strip().split('\t')
		lineArr = []
		for i in range(21):
			lineArr.append(float(currline[i]))
		if int(classifyVector(array(lineArr), trainingWeights)) != int(currline[21]):
			errorCount += 1
	errorRate = (float(errorCount) / numTestVec)
	print "the error rate of this test is: %f" % errorRate
	return errorRate

def multiTest():
	numTests = 10; errorSum = 0.0
	for k in range(numTests):
		errorSum += colicTest()
	print "after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests)) 





dataArr, labelMat = loadDataSet()
# weights = gradAscent(dataArr, labelMat)
# weights = stocGradAscent0(array(dataArr), labelMat)
weights = stocGradAscent1(array(dataArr), labelMat, 200)
print weights

# print type(weights)
plotBestFit(dataArr, labelMat, weights)
# multiTest()