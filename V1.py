# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 17:26:36 2020

@author: Suri
"""
import numpy as np
import matplotlib.pyplot as plt
import warnings
from collections import Counter
from matplotlib import style
import random
import pandas as pd

trainingData={'b':[[1,4],[3,9],[6,0]],'r':[[8,4],[5,7],[1,1]]}
newData=[4,4]
def k_nearest_neighbour(data,prediction,k=3):
    distances = []
    if len(data)>=3:
        warnings.warn('This is a binary classification, you have too many data groups')
    for colour in data:
        for coordinates in data[colour]:
            distance = np.linalg.norm(np.array(coordinates)-np.array(prediction))
            distances.append([distance,colour])
            votes = [i[1] for i in sorted(distances)[:k]]
            result = Counter(votes).most_common(1)[0][0]
            
    return result

def openData(name):
        if name.endswith("-2d.train.csv"):
            dataFrame= pd.read_csv(name, header=-1,index_col=False)
            for i in range(dataFrame.shape[0]):
                if dataFrame.at[i,0] == 1:
                    plt.scatter(dataFrame.at[i,1],dataFrame.at[i,2],color='b')
                elif dataFrame.at[i,0] == -1:
                    plt.scatter(dataFrame.at[i,1],dataFrame.at[i,2],color='r')
        elif name.endswith("-3d.train.csv"):
            dataFrame=pd.read_csv(name,header=-1,index_col=False)
            for i in range(dataFrame.shape[0]):
                if dataFrame.at[i,0] == 1:
                    plt.scatter(dataFrame.at[i,1],dataFrame.at[i,2],dataFrame.at[i,3],color='b')
                elif dataFrame.at[i,0] == -1:
                    plt.scatter(dataFrame.at[i,1],dataFrame.at[i,2],dataFrame.at[i,3],color='r')
        elif name.endswith("-4d.train.csv"):
            dataFrame= pd.read_csv(name, header=-1,index_col = False)
        elif name.endswith("-10d.train.csv"):
            dataFrame = pd.read_csv(name, header =-1,index_col=False)
        plt.show()


#def kNearestNeighbour(data,point,k):
 #   distances = []
  #  for row in data:
            

#[[plt.scatter(ii[1],ii[2],color = i) for ii in dataFrame[i]]for i in dataFrame]
#plt.scatter(newData[0],newData[1],color= a)
#plt.show()
openData('bananas-1-2d.train.csv')