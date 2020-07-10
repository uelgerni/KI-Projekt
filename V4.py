import numpy as np
from KDTreeV2 import kdTree, knn
from helperMethods import dataBeautifier
'''
calculates the "average sign" of data, according to what we should do
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
    returnList = [None] * l
    for i in range(l):
        returnList[i] = copyData[n * i // l:n * (i + 1) // l]
    return returnList


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

returns the point and the classification € {-1,1}
'''





def classifyPoint(point, tree, k):
    kNN = knn(tree, point, k)
    # some data strange beautification:
    kNN = dataBeautifier(kNN)[0]
    #print(kNN)
    classification = sign(kNN)
    return classification, point


'''
classifies an array of points using list comprehension and classifiyPoint
returns a list of tuples (point, classification € {-1,1}) 
'''


def classifyListComp(sample, tree, k):
    resultList = [classifyPoint(point, tree, k) for point in sample]
    return np.array(resultList)
