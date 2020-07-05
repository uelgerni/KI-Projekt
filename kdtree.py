import numpy as np
from V2 import *


# simple function that returns true if the hypersphere around a point intersects the hyperrectangle
def intersects(point, radius, bb):
    usefulPoint = point[1:]
    minVal = bb[0, :]
    maxVal = bb[1, :]
    maxIntersect = maxVal - radius < usefulPoint
    minIntersect = minVal + radius > usefulPoint

    return any(maxIntersect) or any(minIntersect)


class kdTree:
    # inner class node
    class Node:
        def __init__(self, bb, currentAxis, parent=None):
            self.loc = None  # location for inner nodes
            self.right = None  # subtree
            self.left = None  # subtree
            self.data = None  # data for leaves
            self.currentAxis = currentAxis  # current dimensional axis for inner nodes
            self.bb = bb  # boundingbox for nn/knn search
            self.parent = parent  # for backwards traversal

        def isLeaf(self):
            return self.data is not None

    def __init__(self, data, leafsize):
        copydata = np.copy(data)
        self.kdNode = self.buildTree(copydata, leafsize=leafsize)
        self.left = self.kdNode.left
        self.right = self.kdNode.right

    def buildTree(self, data, depth=0, leafsize=10, parent=None):
        n = data.shape[0]
        half = n // 2
        if n == 0:
            return None

        dim = data.shape[1] - 1  # dont want to lose the grouping flag
        axis = depth % dim + 1  # cycle through dimensions

        if n == 1:
            bb = self.computeBB(data)
            node = self.Node(bb, axis, None)
            node.data = data
            print(bb)
            return node
        # sort along given axis with mergesort in O(nlogn)
        # + 1 is because of "colour"
        sortedData = data[data[:, axis].argsort(kind='mergesort')]
        splitPoint = sortedData[half]

        # create next node
        boundingBox = self.computeBB(data)
        node = self.Node(bb=boundingBox, currentAxis=axis, parent=parent)
        node.loc = splitPoint
        # if still too big -> split again

        leftHalf = sortedData[:half]
        rightHalf = sortedData[half:]
        if half >= leafsize:
            node.left = self.buildTree(leftHalf, depth=depth + 1, leafsize=leafsize, parent=self)
            node.right = self.buildTree(rightHalf, depth=depth + 1, leafsize=leafsize, parent=self)
        # else create left and right with data
        else:
            BB1 = self.computeBB(leftHalf)
            node.left = self.Node(bb=BB1, currentAxis=axis, parent=self)
            node.left.data = leftHalf

            BB2 = self.computeBB(rightHalf)
            node.right = self.Node(bb=BB2, currentAxis=axis, parent=self)
            node.right.data = rightHalf
        return node

    def computeBB(self, data):
        bb = np.zeros((2, data.shape[1] - 1))
        bb[0, :] = data[:, 1:].min(axis=0)
        bb[1, :] = data[:, 1:].max(axis=0)
        return bb


def nn(node, pivot):
    while not node.isLeaf():
        axis = node.currentAxis
        print("Axis:", axis)
        print(pivot[axis])
        if pivot[axis] < node.loc[axis]:
            return nn(node.left, pivot)
        else:
            return nn(node.right, pivot)

    mindist = float('inf')
    minPoint = None

    for dataPoint in node.data:
        dist = distance(dataPoint, pivot)
        if dist < mindist:
            mindist = dist
            minPoint = dataPoint
    if intersects(minPoint, mindist, node.bb):
        print(intersects(minPoint, mindist, node.bb))

    return minPoint, mindist


def knn(self, point, k):
    pass


data = numpyReader('bananas-1-2d.train.csv')
points = data.copy()
head = kdTree(points, leafsize=50).kdNode
x = points[:, 1],
y = points[:, 2]

point = np.asarray((1, .05645, 0.6473))

print(nn(head, point))
plt.scatter(x=x, y=y, s=1)
plt.show()
