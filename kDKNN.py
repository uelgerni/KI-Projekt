# coding: utf-8
import numpy as np
from KDTree import knn

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


def classifyPoint(point, tree, k):
    kNN = knn(tree, point, k)
    kNN = dataBeautifier(kNN)[0]
    classification = sign(kNN)
    return classification, point


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


def classifyListComp(sample, tree, k):
    resultList = [classifyPoint(point, tree, k) for point in sample]
    return np.array(resultList)


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


def classifyPointiNNfromKNN(point, knn, i):
    kNN = dataBeautifier(knn)[0]
    classification = sign(kNN[:i])
    return classification, point


'''
list comprehension version for nearest i from k neighbours 
'''


def classifyListiNNfromKNN(sample, knnList, i):
    workingList = [(sample[j], knnList[j]) for j in range(sample.shape[0])]
    resultList = [classifyPointiNNfromKNN(point, knn, i) for point, knn in workingList]
    return resultList
