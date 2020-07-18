import os

import numpy as np
import pandas as pd
import time
import warnings
from math import *
'''
python file for our "helper functions" so our other methods and files aren't too cluttered
'''


# calculates euclidean distance between two points, ignoring the first and last entry (flag and key
# and not taking the root since the root is monotone anyways HAHAHAH JUST KIDDING IT WONT WORK THAT WAY :))
cdef double distance(p1, p2):
    usefulP1 = p1[1:-1]
    usefulP2 = p2[1:-1]
    return sqrt(sum((usefulP1 - usefulP2) ** 2))

# function to read our data and add keys
def numpyTrainingData(filename):
    data = np.genfromtxt('./data/{}.train.csv'.format(filename), delimiter=',', dtype=float)
    keys = np.arange(len(data))
    t = time.time()
    keyedData = np.c_[data, keys]

    return keyedData


# function to read our test data
def numpyTestData(filename):
    data = np.genfromtxt('./data/{}.test.csv'.format(filename), delimiter=',', dtype=float)
    keys = np.arange(len(data))
    keyedData = np.c_[data, keys]
    return keyedData


# same but as a dataframe, only needed for plotting 2d
def pandasReader(filename):
    dataframe = pd.read_csv('./data/{}.train.csv'.format(filename), header=None)
    names = ["Colour"]
    for i in range(dataframe.shape[1] - 1):
        names.append("dim{}".format(i + 1))
    dataframe.columns = names
    return dataframe


# gives our df colour, -1 corresponds to red, 1 to blue
def pandasPlotter(filename):
    dataframe = pandasReader(filename)
    dataframe['Colour'] = dataframe['Colour'].apply(lambda a: 'r' if a == -1 else 'b')
    return dataframe


'''
just a little data beautification thats needed multiple times
'''


def dataBeautifier(data):
    return np.array(np.array(data)[:, 1].tolist()), np.array(np.array(data)[:, 0].tolist())


'''
sorts matrix by last column
'''


def sortListByKey(listToSort):
    sortedList = listToSort[dataBeautifier(listToSort)[0][:, -1].argsort()]
    return sortedList


'''
lists all training files in ./data directory
'''


def listData():
    for file in os.listdir('data'):
        if file.endswith('train.csv'):
            print(file[:-10])


'''
reads a file name, checks if exists, if not reads a file name ...
'''


def readAndTestFilename():
    while True:
        filename = input('please choose one of the above files by typing its name:\n')
        file = 'data/' + filename + '.train.csv'
        if not os.path.isfile(file):
            print('The file {} does not exist in the data directory, please try again with one of the above'.format(
                filename))
            continue
        else:
            break
    return filename


'''
checks if user input is int, reads again until it is
'''


def testIntUserInput(prompt):
    while True:
        try:
            number = int(input(prompt + '\n'))
        except ValueError:
            print('not an integer, try again')
            continue
        else:
            break
    return number


import heapq
from collections import namedtuple
from math import inf
from pprint import pformat

import numpy as np



# simple function that returns true if the hypersphere around a point intersects the hyperrectangle
# also returns false if the hypersphere is totally outside of the hyperrectangle
def intersects(point, radius, bb):
    usefulPoint = point[1:-1]
    minVal = bb[0, :]
    maxVal = bb[1, :]
    # if fully outside return false
    if any(maxVal < (usefulPoint - radius)) or any(minVal > (usefulPoint + radius)):
        return False
    # else check if intersects
    maxIntersect = maxVal < usefulPoint + radius
    minIntersect = minVal > usefulPoint - radius
    return any(maxIntersect) or any(minIntersect)


# computes minimum boundingbox for data
# where boundingbox is an axes aligned hyperrectangle defined by its two opposing outmost vertices
def computeBB(data):
    bb = np.zeros((2, data.shape[1] - 2))
    bb[0, :] = data[:, 1:-1].min(axis=0)
    bb[1, :] = data[:, 1:-1].max(axis=0)
    return bb


# just a simple container for our nodes
class Node(namedtuple('Node', 'value left right boundingBox')):
    def __repr__(self):
        return pformat((tuple(self)))

    def isLeaf(self):
        return self.left is None and self.right is None


'''
recursively creates kd tree from input data
'''


def kdTree(data, depth=0, boundingBox=None):
    boundingBox = computeBB(data) if boundingBox is None else boundingBox
    n = len(data)
    # if node is not leaf
    if n > 1:
        half = n // 2
        dim = len(data[0]) - 2  # flag and key
        axis = depth % dim
        sortedData = data[data[:, axis + 1].argsort(kind='mergesort')]
        value = sortedData[half]
        # creates bounding boxes for left and right child tree
        leftBB = boundingBox.copy()
        rightBB = boundingBox.copy()
        leftBB[1, axis] = value[axis + 1]
        rightBB[0, axis] = value[axis + 1]
        # recursively create children
        return Node(
            value=value,
            left=kdTree(data=sortedData[:half], depth=depth + 1, boundingBox=leftBB),
            right=kdTree(data=sortedData[half + 1:], depth=depth + 1, boundingBox=rightBB),
            boundingBox=boundingBox
        )
    # base case for leaves
    elif n == 1:
        return Node(
            value=data[0],
            left=None,
            right=None,
            boundingBox=boundingBox
        )


'''
recursive knn search for kd trees

only searches paths where closer nodes could possibly be
'''


def knn(root, point, k, axis=0, results=None):
    cdef double dist
    cdef double maxDistance
    dim = len(point) - 2  # ignore classification flag and key
    axis = axis % dim  # cycle through axes

    # init results if None, so only before first recursion
    if results is None:
        results = [(inf, inf)] * k
        heapq.heapify(results)
        maxDistance = inf
    else:
        maxDistance = results[-1][0]

    # basic case
    if root is None:
        return

    # only go into trees where closer points could be
    if not intersects(point, maxDistance, root.boundingBox):
        return
    dist = distance(root.value, point)
    if dist < maxDistance:
        results[-1] = dist, root.value
        results.sort(key=lambda x: x[0])

    # go to subtree with higher probability of finding closer nodes first
    # higher chances of skipping (parts of) other subtree
    if point[axis + 1] < root.value[axis + 1]:
        knn(root=root.left, point=point, k=k, axis=axis + 1, results=results)
        knn(root=root.right, point=point, k=k, axis=axis + 1, results=results)

    else:
        knn(root=root.right, point=point, k=k, axis=axis + 1, results=results)
        knn(root=root.left, point=point, k=k, axis=axis + 1, results=results)

    return results


# coding: utf-8

import numpy as np



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


def classifyPointiNNfromKNN2(point, knn):
    kNN = dataBeautifier(knn)[0]
    classification = np.array([(sign(kNN[:i]), point[0]) for i in range(1, len(kNN) + 1)])
    return classification


'''
list comprehension version for nearest i from k neighbours
'''


def classifyListiNNfromKNN(sample, knnList, i):
    workingList = [(sample[j], knnList[j]) for j in range(sample.shape[0])]
    resultList = [classifyPointiNNfromKNN(point, knn, i) for point, knn in workingList]
    return resultList


def classifyListiNNfromKNN2(sample, knnList):
    workingList = [(sample[j], knnList[j]) for j in range(sample.shape[0])]
    resultList = [classifyPointiNNfromKNN2(point, knn) for point, knn in workingList]
    return resultList



import os
import time


import numpy as np




# supress deprecated warnings from terminal output
warnings.filterwarnings("ignore")

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
finds k* using our train data
'''


def trainData(maxK, blockNum, name):
    data = numpyTrainingData(name)
    splitData = randomlySplitData(data, blockNum)
    errorRateListSum = np.zeros(maxK)

    for lval in range(blockNum):
        dnoi, di = createD_i(splitData, lval)
        tree = kdTree(dnoi)
        kNNList = np.array([np.array(knn(tree, point, maxK)) for point in di])
        classifiedList = classifyListiNNfromKNN2(di, kNNList)

        errorrateList = errorRateList(classifiedList)
        errorRateListSum = errorRateListSum + errorrateList

    # get avg error over i = 0...L-1 for each k
    errorRateAVG = [(errorRateListSum[i] / blockNum, i + 1) for i in range(len(errorRateListSum))]
    # save error for all k for graphic interpretations
    # np.savetxt("{}.errorAVG.csv".format(name),np.array(errorRateAVG))
    # return minimum avg error and corresponding k
    return min(errorRateAVG, key=lambda x: x[0])


'''
tests data name.test.csv with given k
prints error rate
saves result in ./results/name.results.csv
'''


def testData(name, k):
    # builds tree from data, finds k nearest neighbours for all points in data
    data = numpyTestData(name)
    tree = kdTree(data)

    # searches for k + 1 nearest neighbours and ignores closest, since closest is the point itself
    kNNList = np.array([np.array(knn(tree, point, k + 1)) for point in data])[:, 1:]

    # classifies data accordingly and casts to ndarray
    testResults = np.array(classifyListCompFromKNN(data, kNNList))

    # saves classification and coordinates of points together
    csvResults = np.c_[testResults[:, 0].astype(int), data[:, 1:-1]]
    # results including old classification for graphic interpretation
    # graphicResults = np.c_[testResults[:, 0].astype(int), data[:,:-1]]

    # saves results with at most 1 trailing zero for a prettier csv
    formatList = ['%4d']
    formatList.extend(['%1.7f'] * (data.shape[1] - 2))

    # make dir results if not exists
    if not os.path.exists('results'):
        os.mkdir('results')

    # save results as csv
    np.savetxt("results/{}.results.csv".format(name), csvResults, delimiter=', ', fmt=formatList)
    # extend formatlist for graphic results and save those results
    # formatList.extend(['%1.7f'])
    # np.savetxt("{}.graphicresults.csv".format(name), graphicResults, delimiter=', ', fmt=formatList)

    # calculates error rate and prints it
    error = errorRate(testResults, k)
    print("The error rate is: {:1.3f}%".format(error[0] * 100))


def classify(name, maxK, blockNum=5):
    t1 = time.time()
    k_star = trainData(maxK=maxK, name=name, blockNum=blockNum)[1]
    print("k* chosen is:", k_star)
    print("Choosing k* took {:1.3f} seconds.".format(time.time() - t1))
    t2 = time.time()
    testData(name=name, k=k_star)
    print("Testing our data took {:1.3f} seconds".format(time.time() - t2))


def main():
    listData()
    filename = readAndTestFilename()
    testK = testIntUserInput('Please choose a maximum value for k:')
    testL = testIntUserInput('please choose a number of partitions for our training data, 5 is suggested:')
    print("")
    t1 = time.time()
    print("testing for k <= {} and i <= {} and data {}".format(testK, testL, filename))
    classify(filename, testK, testL)
    print("Total runtime was {:1.3f} seconds".format(time.time() - t1))


main()
