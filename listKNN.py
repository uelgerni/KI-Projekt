import numpy as np
from helperMethods import pandasReader, pandasPlotter, numpyTrainingData, distance
import time

t1 = time.time()


def listKNN(dataList, point, k):
    n, dim = dataList.shape
    distances = np.zeros(n)
    for key, entry in enumerate(dataList):
        distances[key] = distance(entry, point)

    b = np.zeros((n, dim + 1))
    b[:, :-1] = point
    b[:, -1] = distances
    b = b[b[:, -1].argsort(kind='mergesort')]
    return b[:k, :-1], b[:k, -1]


def sign(neighbours):
    signum = sum(neighbours[0][:, 0])
    return 1 if signum == 0 else signum / abs(signum)


def classifyPoint(point, data, k):
    kNN = listKNN(point, data, k)
    classification = sign(kNN)
    return point, classification


def classifyListComp(sample, data, k):
    resultList = [classifyPoint(point, data, k) for point in sample]
    return resultList
