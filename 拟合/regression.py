from numpy import *
import matplotlib.pyplot as plt
from time import sleep
from bs4 import BeautifulSoup

def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t')) - 1
    dataMat = []; labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

#using linear regression to find the best fitting straight line
def standRegres(xArr, yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse."
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws

# using local weighted linear regression to find the best fitting straight line
def lwlr(testPoint, xArr, yArr, k = 1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye(m))
    # Generate the weight matrix
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print "This matirx is singular, cannot do inverse."
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k = 1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

# regularize by columns
def regularize(xMat):
    inMat = xMat.copy()
    inMeans = mean(inMat, 0) # calc the mean then subtract it off
    inVar = var(inMat, 0)    # calc the variance of the Xi then divide it
    inMat = (inMat - inMeans) / inVar
    return inMat

# xiangqianzhubuhuigui
# @eps iterator step size
# @numIt iterator number
def stageWise(xArr, yArr, eps = 0.01, numIt = 100):
    xMat = mat(xArr); yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean  # can also regularize ys but will get smaller coef
    xMat = regularize(xMat)
    m,n = shape(xMat)
    returnMat = zeros((numIt, n))
    ws = zeros((n,1)); wsTest = ws.copy(); wsMax = ws.copy()
    for i in range(numIt):
        print ws.T
        lowestError = inf;
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat * wsTest
                rssE = ressError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat

def ridgeRegres(xMat, yMat, lam = 0.2):
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return 
    ws = denom.I * (xMat.T * yMat)
    return ws

def searchForSet(retX, retY, infile, yr, numPce, origPrc):
    fr = open(infile)
    soup = BeautifulSoup(fr.read())
    i = 1

    currentRow = soup.findAll('table', r="%d" % i)
    while(len(currentRow) != 0):
        currentRow = soup.findAll('table', r="%d" % i)
        title = currentRow[0].find('a', {'class' : 'vip'}).text
        lwrTitle = title.lower()

        if(lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
            newFlag = 1.0
        else:
            newFlag = 0.0

        soldUnicde = currentRow[0].find('span', {'class' : 'sold'})
        if soldUnicde is None:
            print 'item #%d did not sell' % i
        else:
            soldPrice = currentRow[0].findAll('td')[4]
            priceStr = soldPrice.text
            priceStr = priceStr.replace('$', '').replace(',', '')
            if len(soldPrice) > 1:
                priceStr = priceStr.replace('Free shipping', '')
            sellingPrice = float(priceStr)

            if sellingPrice > origPrc * 0.5:
                print '%d\t%d\t%d\t%f\t%f' % (yr, numPce, newFlag, origPrc, sellingPrice)
                retX.append([yr, numPce, newFlag, origPrc])
                retY.append(sellingPrice)
        i += 1
        currentRow = soup.findAll('table', r="%d" % i)

def crossValidation(xArr, yArr, numVal = 10):
    m = len(yArr)
    indexList = range(m)
    errorMat = zeros((numVal, 30))

    for i in range(numVal):
        trainX = []; trianY = []
        testX = []; testY = []

        random.shuffle(indexList)

        for j in range(m):
            if j < m * 0.9:
                trainX.append(xArr[indexList[j]])
                trianY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])

        wMat = ridgeTest(trainX, trianY)

        for k in range(30):
            # get testing set and training set
            matTestX = mat(testX); matTrianX = mat(trainX)
            meanTrain = mean(matTrianX, 0)
            varTrain = var(matTrianX, 0)
            matTestX = (matTestX - meanTrain) / varTrain

            yEst = matTestX * mat(wMat[k, :]).T + mean(trianY)

            errorMat[i, k] = ((yEst.T.A - array(testY))**2).sum()

    meanErrors = mean(errorMat, 0)
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors == minMean)]

    xMat = mat(xArr); yMat = mat(yArr).T
    meanX = mean(xMat, 0); varX = var(xMat, 0)
    unReg = bestWeights /varX

    print "the best model from Ridge Regression is:\n", unReg
    print "with constant term: ", -1 * sum(multiply(meanX, unReg)) + mean(yMat)

def setDataCollect(retX, retY):
    searchForSet(retX, retY, '/home/sunkg/project/ml/regression/setHtml/lego8288.html', 2006, 800, 49.99)
    searchForSet(retX, retY, '/home/sunkg/project/ml/regression/setHtml/lego10030.html', 2002, 3096, 269.99)
    searchForSet(retX, retY, '/home/sunkg/project/ml/regression/setHtml/lego10179.html', 2007, 5195, 499.99)
    searchForSet(retX, retY, '/home/sunkg/project/ml/regression/setHtml/lego10181.html', 2007, 3428, 199.99)
    searchForSet(retX, retY, '/home/sunkg/project/ml/regression/setHtml/lego10189.html', 2008, 5922, 299.99)
    searchForSet(retX, retY, '/home/sunkg/project/ml/regression/setHtml/lego10196.html', 2009, 3263, 249.99)

def ressError(yArr, yHatArr):
    return ((yArr - yHatArr)**2).sum()

def ridgeTest(xArr, yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = mean(xMat, 0)
    xVar = var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, exp(i - 10))
        wMat[i, :] = ws.T
    return wMat


# TEST1 standRegression
xArr, yArr = loadDataSet('ex0.txt')
ws = standRegres(xArr, yArr)
xMat = mat(xArr); yMat = mat(yArr)
yHat = xMat * ws
print ws
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
xCopy = xMat.copy()
xCopy.sort(0)
yHat = xCopy * ws
ax.plot(xCopy[:, 1], yHat)
# plt.show()
yHat = xMat * ws

# '''calculate the correlation coefficient 
# between predicted value and real value'''
print corrcoef(yHat.T, yMat)  

# TEST2 local weighted linear regression
# xArr, yArr = loadDataSet('ex0.txt')

# print 'Predict value for sample point', xArr[0]
# print 'Real value is ', yArr[0]
# print 'When k is 1.0, predict value is', lwlr(xArr[0], xArr, yArr, 1.0)[0, 0]
# print 'When k is 0,001, predict value is', lwlr(xArr[0], xArr, yArr, 0.001)[0, 0]


# print 'Predict value for test point [1.0, 0.8] while k is 0.001, the predict value is', \
#   lwlr([1.0, 0.8], xArr, yArr, 0.001)[0, 0]
# # calculate the predict value for all sample points
# yHat = lwlrTest(xArr, xArr, yArr, 0.01)

# xMat = mat(xArr)
# srtInd = xMat[:, 1].argsort(0)
# xSort = xMat[srtInd][:, 0, :]
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(xSort[:, 1], yHat[srtInd])
# ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).T.flatten().A[0], s = 2, c = 'red')
# plt.show()

# TEST3 predict the abalone's age
# abX, abY = loadDataSet('abalone.txt')
# yHat01 = lwlrTest(abX[0 : 99], abX[0 : 99], abY[0 : 99], 0.1)
# yHat1 = lwlrTest(abX[0 : 99], abX[0 : 99], abY[0 : 99], 1.0)
# yHat10 = lwlrTest(abX[0 : 99], abX[0 : 99], abY[0 : 99], 10)

# print 'Now calculate the training error:'
# print 'when k is 0.1, ressError is', ressError(abY[0 : 99], yHat01.T)
# print 'when k is 1.0, ressError is', ressError(abY[0 : 99], yHat1.T)
# print 'when k is 10, ressError is', ressError(abY[0 : 99], yHat10.T)

# print '\nNow calculate the testing error:'
# yHat01 = lwlrTest(abX[100 : 199], abX[0 : 99], abY[0 : 99], 0.1)
# yHat1 = lwlrTest(abX[100 : 199], abX[0 : 99], abY[0 : 99], 1.0)
# yHat10 = lwlrTest(abX[100 : 199], abX[0 : 99], abY[0 : 99], 10)

# print 'when k is 0.1, ressError is', ressError(abY[100 : 199], yHat01.T)
# print 'when k is 1.0, ressError is', ressError(abY[100 : 199], yHat1.T)
# print 'when k is 10, ressError is', ressError(abY[100 : 199], yHat10.T)

# print '\nnow compare with standRegression:'
# ws = standRegres(abX[0 : 99], abY[0 : 99])
# yHat = mat(abX[100 : 199]) * ws
# print 'the ressError of standRegression is', ressError(abY[100 : 199], yHat.T.A)

# TEST4 ridgeRegres
# abX, abY = loadDataSet('abalone.txt')
# ridgeWeights = ridgeTest(abX, abY)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(ridgeWeights)
# ax.grid(True)
# plt.xlabel("log(lambda)")
# plt.show()

# TEST5.0 stageWise
# xArr, yArr = loadDataSet('abalone.txt')
# stageWise(xArr, yArr, 0.001, 5000)

# TEST5.1 compare the weights with 5000 iterations above 
# xMat = mat(xArr)
# yMat = mat(yArr).T
# xMat = regularize(xMat)
# yM = mean(yMat, 0)
# yMat = yMat - yM
# weights = standRegres(xMat, yMat.T)
# print weights.T

# TEST5.2 draw 1000 iterations map
# xArr, yArr = loadDataSet('abalone.txt')
# wMat = stageWise(xArr, yArr, 0.005, 1000)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(wMat)
# ax.grid(True)
# plt.xlabel('Iteration')
# plt.show()

# TEST6 lego price prediction (crossValidaion and redgeRegression)
# lgX = []; lgY = []
# setDataCollect(lgX, lgY)
# crossValidation(lgX, lgY, 10)

#TEST6.1 lego price prediction (standRegression)
# xArr = []; yArr = []
# setDataCollect(xArr, yArr)
# trainX = []; trainY = []
# testX = []; testY = []
# m = len(yArr)
# indexList = range(m)
# random.shuffle(indexList)

# for j in range(m):
#     if j < m * 0.9:
#         trainX.append(xArr[indexList[j]])
#         trainY.append(yArr[indexList[j]])
#     else:
#         testX.append(xArr[indexList[j]])
#         testY.append(yArr[indexList[j]])

# wMat = standRegres(trainX, trainY)
# print wMat.T
# print 'RessError for standRegre is ', ressError(testY, yHat)
