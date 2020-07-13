import heapq
from collections import namedtuple
from math import inf
from pprint import pformat

import numpy as np

from helperMethods import distance


# simple function that returns true if the hypersphere around a point intersects the hyperrectangle
# also returns false if the hypersphere is totally outside of the hyperrectangle
def intersects(point, radius, bb):
    usefulPoint = point[1:-1]
    minVal = bb[0, :]
    maxVal = bb[1, :]
    # if fully outside return false
    if any(maxVal < (usefulPoint - radius)) or any(minVal > (usefulPoint + radius)):
        return False
    # else check if intersects
    maxIntersect = maxVal < usefulPoint + radius
    minIntersect = minVal > usefulPoint - radius
    return any(maxIntersect) or any(minIntersect)


# computes minimum boundingbox for data
# where boundingbox is an axes aligned hyperrectangle defined by its two opposing outmost vertices
def computeBB(data):
    bb = np.zeros((2, data.shape[1] - 2))
    bb[0, :] = data[:, 1:-1].min(axis=0)
    bb[1, :] = data[:, 1:-1].max(axis=0)
    return bb


# just a simple container for our nodes
class Node(namedtuple('Node', 'value left right boundingBox')):
    def __repr__(self):
        return pformat((tuple(self)))

    def isLeaf(self):
        return self.left is None and self.right is None


'''
recursively creates kd tree from input data
'''


def kdTree(data, depth=0, boundingBox=None):
    boundingBox = computeBB(data) if boundingBox is None else boundingBox
    n = len(data)
    # if node is not leaf
    if n > 1:
        half = n // 2
        dim = len(data[0]) - 2  # flag and key
        axis = depth % dim
        sortedData = data[data[:, axis + 1].argsort(kind='mergesort')]
        value = sortedData[half]
        # creates bounding boxes for left and right child tree
        leftBB = boundingBox.copy()
        rightBB = boundingBox.copy()
        leftBB[1, axis] = value[axis + 1]
        rightBB[0, axis] = value[axis + 1]
        # recursively create children
        return Node(
            value=value,
            left=kdTree(data=sortedData[:half], depth=depth + 1, boundingBox=leftBB),
            right=kdTree(data=sortedData[half + 1:], depth=depth + 1, boundingBox=rightBB),
            boundingBox=boundingBox
        )
    # base case for leaves
    elif n == 1:
        return Node(
            value=data[0],
            left=None,
            right=None,
            boundingBox=boundingBox
        )


'''
recursive knn search for kd trees

only searches paths where closer nodes could possibly be
'''


def knn(root, point, k, axis=0, results=None):
    dim = len(point) - 2  # ignore classification flag and key
    axis = axis % dim  # cycle through axes

    # init results if None, so only before first recursion
    if results is None:
        results = [(inf, inf)] * k
        heapq.heapify(results)
        maxDistance = inf
    else:
        maxDistance = results[-1][0]

    # basic case
    if root is None:
        return

    # only go into trees where closer points could be
    if not intersects(point, maxDistance, root.boundingBox):
        return
    dist = distance(root.value, point)
    if dist < maxDistance:
        results[-1] = dist, root.value
        results.sort(key=lambda x: x[0])

    # go to subtree with higher probability of finding closer nodes first
    # higher chances of skipping (parts of) other subtree
    if point[axis + 1] < root.value[axis + 1]:
        knn(root=root.left, point=point, k=k, axis=axis + 1, results=results)
        knn(root=root.right, point=point, k=k, axis=axis + 1, results=results)

    else:
        knn(root=root.right, point=point, k=k, axis=axis + 1, results=results)
        knn(root=root.left, point=point, k=k, axis=axis + 1, results=results)

    return results


'''
uses heapq instead of sorting all the time, not really faster, also not fixed for latest versions
'''


def knn2(root, point, k, axis=0, results=None):
    dim = len(point) - 1  # ignore classification flag
    axis = axis % dim  # cycle through axes

    # init results if None, so only before first recursion
    if results is None:
        results = [(-inf, -inf)] * k
        heapq.heapify(results)
        maxDistance = -inf
    else:
        maxDistance = results[-1][0]
    # basic case
    if root is None:
        return

    # only go into trees where closer points could be
    if not intersects(point, -maxDistance, root.boundingBox):
        return
    dist = distance(root.value, point)
    if dist > maxDistance:
        heapq.heappushpop(results, (-dist, root.value))

    # go to subtree with higher probability of finding closer nodes first
    # higher chances of skipping (parts of) other subtree
    if point[axis + 1] < root.value[axis + 1]:
        knn2(root=root.left, point=point, k=k, axis=axis + 1, results=results)
        knn2(root=root.right, point=point, k=k, axis=axis + 1, results=results)

    else:
        knn2(root=root.right, point=point, k=k, axis=axis + 1, results=results)
        knn2(root=root.left, point=point, k=k, axis=axis + 1, results=results)

    return results
