import numpy as np
from V2 import *


class kdTree:
    # inner class node
    class Node:
        def __init__(self):
            self.loc = None
            self.right = None
            self.left = None
            self.data = None

        def isLeaf(self):
            return self.data is not None

    def __init__(self, data, leafsize=10):
        self.node = self.buildTree(data, leafsize=leafsize)
        self.left = self.node.left
        self.right = self.node.right

    def buildTree(self, data, depth=0, leafsize=10):
        n = data.shape[0]
        half = n // 2
        if n == 0:
            return None
        dim = data.shape[1]
        axis = depth % dim  # cycle through dimensions

        # sort along given axis "=dimension" with mergesort in O(nlogn)
        sortedData = data[data[:, axis].argsort(kind='mergesort')]
        splitpoint = sortedData[half]

        # create next node
        node = self.Node()
        node.loc = splitpoint
        # if still too big -> split again
        if half >= leafsize:
            node.left = self.buildTree(sortedData[:half], depth=depth + 1)
            node.right = self.buildTree(sortedData[half:], depth=depth + 1)
        # else create left and right with data
        else:
            node.left = self.Node()
            node.left.data = sortedData[:half]
            node.right = self.Node()
            node.right.data = sortedData[half:]
        return node


data = numpyReader('bananas-1-2d.train.csv')
points = data[:, 1:]
head = kdTree(points).node

while not head.isLeaf():
    head= head.left
print(head.data)
