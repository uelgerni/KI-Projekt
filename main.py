from V4 import randomlySplitData, createD_i, classifyListComp, classifyPoint, dataBeautifier, sign
from helperMethods import numpyReader, pandasPlotter, dataBeautifier, sortListByKey
from KDTreeV2 import kdTree, knn
import time
import numpy as np
from listKNN import listKNN
import matplotlib.pyplot as plt

'''
just some tests
'''


def buildTreeandD_i(filename):
    data = numpyReader(filename)
    splitData = randomlySplitData(data, 5)
    D_noI, D_i = createD_i(splitData, 3)
    tree = kdTree(D_noI)
    return D_i, tree


def classifyList2d(k):
    filename = 'bananas-1-2d'
    t1 = time.time()
    D_i, tree = buildTreeandD_i(filename)
    t2 = time.time()
    result = classifyListComp(D_i, tree, k)
    t3 = time.time()
    print("building and splitting the tree took ", t2 - t1, " seconds")
    print("classifying took ", t3 - t2, " seconds")
    return result


def classifyList10d(k):
    filename = 'toy-10d'
    D_i, tree = buildTreeandD_i(filename)
    result = classifyListComp(D_i, tree, k)
    return result


def testKNNKD_10d():
    filename = 'toy-10d'
    data = numpyReader(filename)[1:]
    tree = kdTree(data)
    testPoint = data[0]
    KNN = np.array(knn(tree, testPoint, 100))[:, 0]  # distances
    KNN2 = listKNN(data, testPoint, 100)[1]  # distances
    return all(KNN - KNN2 < 0.0001)


def errorRate(classifiedList, k):
    n = len(classifiedList)
    results, actualValue = dataBeautifier(classifiedList)[1], dataBeautifier(classifiedList)[0][:, 0]
    return np.sum(results != actualValue) / n, k


t1 = time.time()
ourList = classifyList10d()
testStuffy, testStuffy2 = dataBeautifier(ourList)[1], dataBeautifier(ourList)[0][:, 0]

print("knn not broken yet: ", testKNNKD_10d())

print("Test took ", time.time() - t1, " seconds")
