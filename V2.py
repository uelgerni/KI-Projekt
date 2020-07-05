import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import time
import math

b1 = 'bananas-1-2d.train.csv'

'''
reads csv filename from data directory and returns as dataframe with first column mapped to red and blue for plotting
'''


def pandasPlotter(filename):
    dataframe = pandasReader(filename)
    dataframe[0] = dataframe[0].map(lambda a: 'r' if a == -1 else 'b')
    return dataframe


'''
reads csv filename from data directory and returns as dataframe
'''


def pandasReader(filename):
    dataframe = pd.read_csv('./data/{}'.format(filename), header=None)
    return dataframe


def numpyReader(filename):
    return np.genfromtxt('./data/{}'.format(filename), delimiter=',', dtype=float)


def distance(p1, p2):
    return np.linalg.norm(p1 - p2)


'''
def kd_tree(dataframe, depth=0):
    n = len(dataframe)
    #if no points are given something went wrong so just exit before more goes wrong
    if n <= 0:
        print("No points given")
        exit()

    dim = (len(dataframe.columns)-1) % depth
'''


def kd_tree(points, i=0):
    dim = points.shape[1]
    if len(points) > 1:
        points[points[:,i].argsort()]
        i = (i + 1) % dim
        half = len(points) >> 1
        return [
            kd_tree(points[: half], i),
            kd_tree(points[half + 1:], i),
            points[half]
        ]
    elif len(points) == 1:
        return [None, None, points[0]]


'''
df = pandasReader(b1)
points = df.drop(columns=0)
'''

data = numpyReader(b1)
points = data[1:]
#a=points
#print(points)
#print(points[points[:,1].argsort()])
#a = a[a[:,0].argsort(kind='mergesort')]
#print(kd_tree(points))
tree = kd_tree(points)
print(tree)
'''ax1 = dataframe.plot.scatter(1, 2, c=dataframe[0])
plt.show()
'''
