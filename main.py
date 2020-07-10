from kDKNN import randomlySplitData, createD_i, classifyListComp, classifyListCompFromKNN, classifyListiNNfromKNN
from helperMethods import numpyReader, dataBeautifier
from KDTreeV2 import kdTree, knn
import time
import numpy as np
from listKNN import listKNN

'''
just some tests
'''


def buildTreeandD_i(filename, i, l):
    if i >= l:
        print("i = {} has to be strictly smaller than l = {}".format(i, l))
        exit()
    data = numpyReader(filename)
    splitData = randomlySplitData(data, l)
    D_noI, D_i = createD_i(splitData, i)
    tree = kdTree(D_noI)
    return D_i, tree


def classifyList2d(k, i, l):
    filename = 'bananas-1-2d'
    t1 = time.time()
    D_i, tree = buildTreeandD_i(filename, i, l)
    t2 = time.time()
    result = classifyListComp(D_i, tree, k)
    t3 = time.time()
    print("building and splitting the tree took ", t2 - t1, " seconds")
    print("classifying took ", t3 - t2, " seconds")
    return result


def classifyList2dKNN(knnList, i, l):
    filename = 'bananas-1-4d'
    t1 = time.time()
    D_i, tree = buildTreeandD_i(filename, i, l)
    t2 = time.time()
    result = classifyListCompFromKNN(D_i, knnList)
    t3 = time.time()
    print("building and splitting the tree took ", t2 - t1, " seconds")
    print("classifying took ", t3 - t2, " seconds")
    return result


def classifyList10d(k, i, l):
    filename = 'toy-10d'
    D_i, tree = buildTreeandD_i(filename, i, l)
    result = classifyListComp(D_i, tree, k)
    return result


def testKNNKD_10d():
    filename = 'toy-10d'
    data = numpyReader(filename)[1:]
    tree = kdTree(data)
    randomInt = np.random.randint(0, 10000)
    testPoint = data[randomInt]
    KNN = np.array(knn(tree, testPoint, 100))[:, 0]  # distances
    KNN2 = listKNN(data, testPoint, 100)[1]  # distances
    return all(KNN - KNN2 < 0.0001)


def errorRate(classifiedList, k):
    n = len(classifiedList)
    results, actualValue = dataBeautifier(classifiedList)[1], dataBeautifier(classifiedList)[0][:, 0]
    return np.sum(results != actualValue) / n, k


t1 = time.time()

testK = 20
testL = 5
filename = 'bananas-1-2d'
# filename = 'toy-10d'
print("testing for k <= ", testK, "and i <= ", testL, "and data ", filename)

data = numpyReader(filename)

splittedData = randomlySplitData(data, testL)
errorRateListSum = [0] * testK

for lval in range(testL):

    dnoi, di = createD_i(splittedData, lval)
    tree = kdTree(dnoi)
    kNNList = np.array([np.array(knn(tree, point, testK)) for point in di])

    errorrateList = [None] * (testK)
    for i in range(1, testK + 1):
        testresults = classifyListiNNfromKNN(di, kNNList, i)
        errorrate = errorRate(testresults, i)
        errorrateList[i - 1] = errorrate
        errorRateListSum[i - 1] += errorrate[0]

    errorrateList.sort(key=lambda x: x[0])
errorRateAVG = [(errorRateListSum[i] / testL, i + 1) for i in range(len(errorRateListSum))]
print(errorRateAVG)
print("k* = ", min(errorRateAVG, key=lambda x: x[0]))
print("knn not broken yet: ", testKNNKD_10d())

print("Test took ", time.time() - t1, " seconds")
