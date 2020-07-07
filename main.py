from V4 import randomlySplitData, createD_i, classifyListComp, classifyPoint, dataBeautifier, sign
from helperMethods import numpyReader, pandasPlotter
from KDTreeV2 import kdTree, knn
import time
import numpy as np
import matplotlib.pyplot as plt

filename = 'bananas-1-2d.train.csv'
data = numpyReader(filename)
splitData = randomlySplitData(data, 5)
D_noI, D_i = createD_i(splitData, 3)
tree = kdTree(data)



'''
just some tests
'''
for i in range(100):
    rando = np.random.randint(0, 5000)
    p = np.array(data[rando])
    classifiedPoint = classifyPoint(p, tree, 5)

    plotData = pandasPlotter(filename)
    # print(classiefiedPoints)
    # print(dataBeautifier(classiefiedPoints)[0])

    pNearest = dataBeautifier(knn(tree, p, 5))[0]
    xVals = pNearest[:, 1]
    yVals = pNearest[:, 2]
    cmap = ['black' for val in pNearest[:, 0]]
    cmap2 = ['r' if val == -1 else 'blue' for val in pNearest[:, 0]]  # else 'b']

    plotData.plot.scatter(x='dim1', y='dim2', c='Colour', s=15)
    plt.xlim((p[1] - .03, p[1] + .03))
    plt.ylim((p[2] - .03, p[2] + .03))
    pcol = 'r' if p[0] == -1 else 'blue'
    plt.scatter(x=p[1], y=p[2], c=pcol, s=100)
    pguessedcol = 'r' if sign(dataBeautifier(knn(tree, p, 5))[0]) == -1 else 'blue'
    plt.scatter(x=p[1], y=p[2], c=pcol, s=10)

    plt.scatter(x=xVals, y=yVals, c=cmap, s=50)
    plt.scatter(x=xVals, y=yVals, c=cmap2, s=10)
    plt.savefig('pictures/{}'.format(i))
    plt.close()
