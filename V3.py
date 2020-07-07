import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
reads filename from directory
"""


def numpyReader(filename):
    return np.genfromtxt('./data/{}'.format(filename), delimiter=',', dtype=float)


def pandasReader(filename):
    dataframe = pd.read_csv('./data/{}'.format(filename), header=None)
    names = ["Colour"]
    for i in range(dataframe.shape[1] - 1):
        names.append("dim{}".format(i + 1))
    dataframe.columns = names
    return dataframe


def pandasPlotter(filename):
    dataframe = pandasReader(filename)
    dataframe['Colour'] = dataframe['Colour'].apply(lambda a: 'r' if a == -1 else 'b')
    return dataframe


"""
returns distance from points without using the "colour flag"
"""


def distance(p1, p2):
    usefulp1 = p1[1:]
    usefulp2 = p2[1:]
    return np.linalg.norm(usefulp1 - usefulp2)


import time

t1 = time.time()


def listKNN(point, data, k):
    n, dim = data.shape
    distances = np.zeros(n)
    for key, entry in enumerate(data):
        distances[key] = distance(entry, point)
    b = np.zeros((n, data.shape[1] + 1))
    b[:, :-1] = data
    b[:, -1] = distances
    b = b[b[:, -1].argsort(kind='mergesort')]
    return b[:k, :-1], b[:k, -1]


def sign(neighbours):
    signum = sum(neighbours[0][:, 0])
    return 1 if signum == 0 else signum / abs(signum)


def randomlySplitData(data, l):
    n = len(data)
    copyData = data.copy()
    np.random.shuffle(copyData)
    returnList = [None] * l
    for i in range(l):
        returnList[i] = copyData[n * i // l:n * (i + 1) // l]
    return returnList


def createD_i(listOfData, i):
    dim = len(listOfData[0][0])
    returnArray = np.ndarray((0, dim))
    for k in range(len(listOfData)):
        if k != i:
            returnArray = np.concatenate((returnArray, listOfData[k]))

    return returnArray, listOfData[i]


def classifyPoint(point, data, k):
    kNN = listKNN(point, data, k)
    classification = sign(kNN)
    return point, classification


def classifyForLoop(sample, data, k):
    resultArray = np.zeros((sample.shape[0], 2))
    for point in sample:
        yield classifyPoint(point, data, k)


def classifyListComp(sample, data, k):
    resultList = [classifyPoint(point, data, k) for point in sample]
    return resultList


data = numpyReader('bananas-1-2d.train.csv')


# plotData = pandasPlotter('bananas-1-2d.train.csv')
# plotData.plot.scatter(x='dim1', y='dim2', c='Colour')
# plt.show()
point = (1, 0.1, 0.2)
# data = numpyReader('bananas-1-4d.train.csv')
# data = numpyReader('toy-10d.train.csv')
t1 = time.time()
nnn = listKNN(point, data, 5)
print(nnn)
print(time.time()-t1)