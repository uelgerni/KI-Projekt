import numpy as np
import pandas as pd
import time

'''
python file for our "helper functions" so for example kdtree.py is not too cluttered
'''


# calculates euclidean distance between two points, ignoring the first entry
def distance(p1, p2):
    usefulP1 = p1[1:-1]
    usefulP2 = p2[1:-1]
    return np.linalg.norm(usefulP1 - usefulP2)


# function to read our data and add keys
def numpyTrainingData(filename):
    data = np.genfromtxt('./data/{}.train.csv'.format(filename), delimiter=',', dtype=float)
    keys = np.arange(len(data))
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
