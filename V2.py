import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# standard libraries
import math, sys, time, os
# printing and heapqeue
import pprint, heapq
from collections import defaultdict

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


def checkNeighbour(pivot, p1, p2):
    # if either one doesnt exist return the other, if both dont exist return None
    if p1 is None:
        return p2
    if p2 is None:
        return p1
    # checks whether p1 is actually closer or only in the nearest leaf and returns the closer one
    return p1 if distance(pivot, p1) < distance(pivot, p2) else p2

