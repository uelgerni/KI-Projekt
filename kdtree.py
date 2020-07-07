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


'''
this was so so hard to program and yet it is so bad that it actually takes longer than a simple bruteforce sort and search :))

builds a kdTree out of our data so we can search a LOT faster: O(log(n)) instead of O(n)
'''


class kdTree:
    # inner class node
    class Node:
        def __init__(self, bb, currentAxis, parent):
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
        self.bb = self.computeBB(data)
        self.kdNode = self.buildTree(copydata, self.bb, parent=self, leafsize=leafsize)
        self.left = self.kdNode.left
        self.right = self.kdNode.right
        self.data = data
        self.isLeaf = lambda: False  # just to catch some errors that won't exist outside of testing stuff that's not necessary

    def buildTree(self, data, bb, parent, depth=0, leafsize=10):
        n = data.shape[0]
        half = n // 2
        if n == 0:
            return None

        dim = data.shape[1] - 1  # dont want to lose the grouping flag
        axis = depth % dim + 1  # cycle through dimensions

        if n == 1:
            node = self.Node(bb, axis, parent)
            node.data = data
            return node
        # sort along given axis with mergesort in O(nlogn)
        # + 1 is because of "colour"
        sortedData = data[data[:, axis].argsort(kind='mergesort')]
        splitPoint = sortedData[half]

        leftBB = bb.copy()
        rightBB = bb.copy()
        leftBB[1] = splitPoint[1:]
        rightBB[0] = splitPoint[1:]
        # create next node
        node = self.Node(parent=parent, bb=bb, currentAxis=axis)
        node.loc = splitPoint

        # we're using this split either way
        leftHalf = sortedData[:half]
        rightHalf = sortedData[half:]

        # if still too big -> split again
        if half >= leafsize:
            node.left = self.buildTree(leftHalf, bb=leftBB, parent=node, depth=depth + 1, leafsize=leafsize)
            node.right = self.buildTree(rightHalf, bb=rightBB, parent=node, depth=depth + 1, leafsize=leafsize)
        # else create left and right with data
        else:
            node.left = self.Node(bb=leftBB, currentAxis=axis, parent=node)
            node.left.data = leftHalf

            node.right = self.Node(bb=rightBB, currentAxis=axis, parent=node)
            node.right.data = rightHalf
        return node

    # computes axes aligned hyperrectangle around date
    def computeBB(self, data):
        bb = np.zeros((2, data.shape[1] - 1))
        bb[0, :] = data[:, 1:].min(axis=0)
        bb[1, :] = data[:, 1:].max(axis=0)
        return bb


'''
finds all leaves under a given node, very similar to inorder traversal
'''


def findAllLeaves(pivot, radius, node, foundData=[]):
    if node.left is not None:
        findAllLeaves(pivot=pivot, radius=radius, node=node.left, foundData=foundData)
    if node.isLeaf() & intersects(pivot, radius, node.bb):
        for dp in node.data:
            foundData.append(dp)
    if node.right is not None:
        findAllLeaves(pivot=pivot, radius=radius, node=node.right, foundData=foundData)
    return np.asarray(foundData)


'''
nearest neighbour search in O(log(n)) time using our kdtree
'''


def nn(node, pivot):
    # go to the leaf where the pivot would be
    while not node.isLeaf():
        axis = node.currentAxis
        if pivot[axis] < node.loc[axis]:
            return nn(node.left, pivot)
        else:
            return nn(node.right, pivot)

    # init mindist as inf so all other points are closer
    mindist = float('inf')
    minPoint = None

    # check for closest point in leaf
    for dataPoint in node.data:
        dist = distance(dataPoint, pivot)
        if dist < mindist:
            mindist = dist
            minPoint = dataPoint

    # if hypersphere around minpoint intersects neighbouring leafs -> there could be closer points
    # -> go up to parent enough times, so we check all neighbouring leaves
    if intersects(minPoint, mindist, node.bb):
        # jump once for leaving the leaf,
        # then 2**(dim+1) == 2** shape[0] to reach last common (grandgrand..)parent of all neighbours
        jumps = 2 ** (minPoint.shape[0]) + 1
        parentNode = node
        for i in range(jumps):
            # if check just to catch errors in testing, not really needed
            if isinstance(parentNode, kdTree.Node):
                parentNode = parentNode.parent
        # append all neighbours
        dataList = findAllLeaves(parentNode)
        # search all those neighbours
        for dataPoint in dataList:

            dist = distance(dataPoint, pivot)
            # if smaller dist ->  replace
            if dist < mindist:
                mindist = dist
                minPoint = dataPoint
    return minPoint, mindist


'''
reaally similar to nn 
'''


def knn(node, pivot, k):
    dim = len(pivot) - 1
    # go to the leaf where the pivot would be
    while not node.isLeaf():
        axis = node.currentAxis
        if pivot[axis] < node.loc[axis]:
            return knn(node.left, pivot, k)
        else:
            return knn(node.right, pivot, k)

    # init kmindist as inf so all other points are closer
    kmindist = float('inf')
    # init minArray with correct size so we dont have to copy all the time
    # and with infinity distances so they all get replaced
    minArray = np.full((k, 2), np.inf, dtype=object)

    # check for closest point in leaf
    for dataPoint in node.data:
        dist = distance(dataPoint, pivot)
        if dist < kmindist:
            minArray[k - 1] = np.asmatrix((dataPoint, dist))
            # sort so we just have to replace last point
            minArray = minArray[minArray[:, 1].argsort(kind='mergesort')]
            kmindist = minArray[-1, 1]

    # if hypersphere around minpoint intersects neighbouring leafs -> there could be closer points
    # -> go up to parent enough times, so we check all neighbouring leaves
    parentNode = node.parent
    i = 0
    print(minArray[-1])
    while isinstance(parentNode, kdTree.Node) & intersects(pivot, kmindist, parentNode.bb):
        # while distance(pivot, parentNode.loc) < kmindist:
        parentNode = parentNode.parent
    # append all neighbours
    print("now finding all leaves")
    dataList = findAllLeaves(pivot=pivot, radius=kmindist, node=parentNode)
    # search all those neighbours
    print(len(dataList))
    for dataPoint in dataList:
        dist = distance(dataPoint, pivot)
        if dist < kmindist:
            minArray[k - 1] = [dataPoint, dist]
            # sort so we just have to replace last point
            minArray = minArray[minArray[:, 1].argsort(kind='mergesort')]
            kmindist = minArray[-1, 1]
    return minArray


t1 = time.time()
data = numpyReader('bananas-1-2d.train.csv')
points = data.copy()
head = kdTree(points, leafsize=64).kdNode
x = points[:, 1]
y = points[:, 2]

point = (1, 0.1, 0.3)
t3 = time.time()
dah = knn(head, point, 20)
print(dah[:, 1:])
t4 = time.time()
plt.scatter(x=x, y=y, s=1)
for entry in dah[:, :1]:
    plt.scatter(entry[0][1:][0], entry[0][1:][1], c='red')

t2 = time.time()

print("alles insgesamt hat {} sekunden gebraucht".format(t2 - t1))
print("und die knn suche hat {} sekunden gedauert".format(t4 - t3))
plt.scatter(point[1], point[2], c='g')
plt.xlim(point[1] - 0.1, point[1] + 0.1)
plt.ylim(point[2] - 0.1, point[2] + 0.1)
plt.show()
