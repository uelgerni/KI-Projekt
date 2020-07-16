# coding: utf-8

import numpy as np

from helperMethods import dataBeautifier

'''
our classification functions for points and lists of points

for classifying from the number k or classifying from the actual k nearest neigbours
also for classifying using only i <= k of the k nearest neighbours 
'''

'''
calculates the "average sign" of data, if sign == 0 chooses nearest neighbour instead
'''


def sign(neighbours):
    signum = sum(neighbours[:, 0])
    return 1 if signum == 0 else signum / abs(signum)


'''
takes a classification which includes chosen values, actual values and of course the points themselves
also takes k just so it can give it back
returns error rate and k

used if you only have 1 k
'''


def errorRate(classifiedList, k):
    n = len(classifiedList)
    results, actualValue = dataBeautifier(classifiedList)[1], dataBeautifier(classifiedList)[0][:, 0]
    return np.sum(results != actualValue) / n, k


'''
takes a numpy array of classifications which includes chosen values and actual values for each point for all k's
returns classification error rate for each k

used if you have to test a lot of k's
'''


def errorRateList(classifiedList):
    n = len(classifiedList)
    pointList = np.array(classifiedList)
    errorRateList = [np.sum(pointList[:, i][:, 0] != pointList[:, i][:, 1]) / n for i in
                     range(len(pointList[0]))]
    return np.array(errorRateList)


'''
creates a copy of data, shuffles it and partitions it into l parts of (almost) same size
returns a list with l entries
'''


def randomlySplitData(data, l):
    n = len(data)
    copyData = data.copy()
    np.random.shuffle(copyData)
    returnList = [copyData[n * i // l:n * (i + 1) // l] for i in range(l)]
    return np.array(returnList)


'''
takes a list of "datablocks" according to randomly split data
returns an array with all datablocks but the ith one concatenated, and the ith datablock
'''


def createD_i(listOfData, i):
    dim = len(listOfData[0][0])
    returnArray = np.ndarray((0, dim))
    for k in range(len(listOfData)):
        if k != i:
            returnArray = np.concatenate((returnArray, listOfData[k]))

    return returnArray, listOfData[i]


'''
given a point, a kdTree and a number k classifies the point according to the sign of the k nearest neighbours

returns the point and the classification -1 or 1
'''

'''
same but from knn instead of k
'''


def classifyPointFromKNN(point, knn):
    kNN = dataBeautifier(knn)[0]
    classification = sign(kNN)
    return classification, point


'''
classifies an array of points using list comprehension and classifiyPoint
returns a list of tuples (point, classification € {-1,1}) 
'''

'''
does the same but expects knn and not k 
'''


def classifyListCompFromKNN(sample, knnList):
    workingList = [(sample[i], knnList[i]) for i in range(sample.shape[0])]
    resultList = [classifyPointFromKNN(point, knn) for point, knn in workingList]
    return resultList


'''
gets knn list but only uses nearest i neighbours
'''


def classifyPointiNNfromKNN2(point, knn):
    kNN = dataBeautifier(knn)[0]
    classification = np.array([(sign(kNN[:i]), point[0]) for i in range(1, len(kNN) + 1)])
    return classification


'''
list comprehension version for nearest i from k neighbours 
'''


def classifyListiNNfromKNN2(sample, knnList):
    workingList = [(sample[j], knnList[j]) for j in range(sample.shape[0])]
    resultList = [classifyPointiNNfromKNN2(point, knn) for point, knn in workingList]
    return resultList
