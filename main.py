from kDKNN import randomlySplitData, createD_i, classifyListCompFromKNN, classifyListiNNfromKNN
from helperMethods import numpyTrainingData, dataBeautifier, numpyTestData
from KDTreeV2 import kdTree, knn
import time
import numpy as np

'''
takes a classification which includes chosen values, actual values and of course the points themselves
also takes k just so it can give it back
returns error rate and k
'''


def errorRate(classifiedList, k):
    n = len(classifiedList)
    results, actualValue = dataBeautifier(classifiedList)[1], dataBeautifier(classifiedList)[0][:, 0]
    return np.sum(results != actualValue) / n, k


'''
finds k* using our train data
'''


def trainData(maxK, blockNum, name):
    data = numpyTrainingData(name)
    splittedData = randomlySplitData(data, blockNum)
    errorRateListSum = [0] * maxK
    # for i = 0, .. L-1 build our sample and tree from rest of training data
    # also build our knn for each point in our sample D_i
    for lval in range(blockNum):

        dnoi, di = createD_i(splittedData, lval)
        tree = kdTree(dnoi)
        kNNList = np.array([np.array(knn(tree, point, maxK)) for point in di])

        errorrateList = [None] * maxK

        # for each k in KSET = (1, ..., maxK) check our error rate and sum it up
        for i in range(1, maxK + 1):
            testresults = classifyListiNNfromKNN(di, kNNList, i)
            errorrate = errorRate(testresults, i)
            errorrateList[i - 1] = errorrate
            errorRateListSum[i - 1] += errorrate[0]

    # get avg error over i = 0...L-1 for each k
    errorRateAVG = [(errorRateListSum[i] / blockNum, i + 1) for i in range(len(errorRateListSum))]
    # return minimum avg error and corresponding k
    return min(errorRateAVG, key=lambda x: x[0])


def testData(name, k):
    # builds tree from data, finds k nearest neighbours for all points in data
    data = numpyTestData(filename)
    tree = kdTree(data)
    kNNList = np.array([np.array(knn(tree, point, k)) for point in data])

    # classifies data accordingly and casts to ndarray
    testResults = np.array(classifyListCompFromKNN(data, kNNList))

    # saves classification and coordinates of points together
    csvResults = np.c_[testResults[:, 0].astype(int), data[:, 1:-1]]

    # saves results with at most 1 trailing zero for a prettier csv
    formatList = ['%4d']
    formatList.extend(['%1.7f'] * (data.shape[1] - 2))
    np.savetxt("results/{}.results.csv".format(name), csvResults, delimiter=', ', fmt=formatList)

    # calculates error rate and prints it
    error = errorRate(testResults, k)
    print("The error rate is: {:1.3f}%".format(error[0] * 100))


def classify(name, maxK, blockNum=5):
    t1 = time.time()
    k_star = trainData(maxK=maxK, name=name, blockNum=blockNum)[1]
    print("k* chosen is:", k_star)
    print("Choosing k* took {:1.3f} seconds.".format(time.time() - t1))
    t2 = time.time()
    testData(name=name, k=k_star)
    print("Testing our data took {:1.3f} seconds".format(time.time() - t2))


t1 = time.time()
testK = 20
testL = 5
filename = 'bananas-1-2d'
# filename = 'toy-10d'
print("testing for k <= {} and i <= {} and data {}".format(testK, testL, filename))

classify(filename, testK)

print("Total runtime was {:1.3f} seconds".format(time.time() - t1))
