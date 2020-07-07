import numpy as np
import pandas as pd

'''
python file for our "helper functions" so for example kdtree.py is not too cluttered
'''


# calculates euclidean distance between two points, ignoring the first entry
def distance(p1, p2):
    usefulP1 = p1[1:]
    usefulP2 = p2[1:]
    return np.linalg.norm(usefulP1 - usefulP2)


# simple function that returns true if the hypersphere around a point intersects the hyperrectangle
# also returns false if the hypersphere is totally outside of the hyperrectangle
def intersects(point, radius, bb):
    usefulPoint = point[1:]
    minVal = bb[0, :]
    maxVal = bb[1, :]
    # if fully outside return false
    if any(maxVal < (usefulPoint - radius)) or any(minVal > (usefulPoint + radius)):
        return False
    # else check if intersects
    maxIntersect = maxVal < usefulPoint + radius
    minIntersect = minVal > usefulPoint - radius
    return any(maxIntersect) or any(minIntersect)


# function to read our data
def numpyReader(filename):
    return np.genfromtxt('./data/{}'.format(filename), delimiter=',', dtype=float)


# same but as a dataframe, not needed
def pandasReader(filename):
    dataframe = pd.read_csv('./data/{}'.format(filename), header=None)
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
