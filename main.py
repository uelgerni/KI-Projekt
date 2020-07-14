import os
import time
import warnings

import numpy as np

from KDTree import kdTree, knn
from classification import randomlySplitData, createD_i, classifyListCompFromKNN, classifyListiNNfromKNN2
from helperMethods import numpyTrainingData, dataBeautifier, numpyTestData, listData, readAndTestFilename, \
    testIntUserInput

# supress deprecated warnings from terminal output
warnings.filterwarnings("ignore")

'''
takes a classification which includes chosen values, actual values and of course the points themselves
also takes k just so it can give it back
returns error rate and k

used if you only have 1 k
'''


def errorRate(classifiedList, k):
    n = len(classifiedList)
    results, actualValue = dataBeautifier(classifiedList)[1], dataBeautifier(classifiedList)[0][:, 0]
    return np.sum(results != actualValue) / n, k


'''
takes a numpy array of classifications which includes chosen values and actual values for each point for all k's
returns classification error rate for each k

used if you have to test a lot of k's
'''


def errorRateList(classifiedList):
    n = len(classifiedList)
    pointList = np.array(classifiedList)
    errorRateList = [np.sum(pointList[:, i][:, 0] != pointList[:, i][:, 1]) / n for i in
                     range(len(pointList[0]))]
    return np.array(errorRateList)


'''
finds k* using our train data
'''


def trainData(maxK, blockNum, name):
    data = numpyTrainingData(name)
    splitData = randomlySplitData(data, blockNum)
    errorRateListSum = np.zeros(maxK)

    for lval in range(blockNum):
        dnoi, di = createD_i(splitData, lval)
        tree = kdTree(dnoi)
        kNNList = np.array([np.array(knn(tree, point, maxK)) for point in di])
        classifiedList = classifyListiNNfromKNN2(di, kNNList)

        errorrateList = errorRateList(classifiedList)
        errorRateListSum = errorRateListSum + errorrateList

    # get avg error over i = 0...L-1 for each k
    errorRateAVG = [(errorRateListSum[i] / blockNum, i + 1) for i in range(len(errorRateListSum))]
    # save error for all k for graphic interpretations
    # np.savetxt("{}.errorAVG.csv".format(name),np.array(errorRateAVG))
    # return minimum avg error and corresponding k
    return min(errorRateAVG, key=lambda x: x[0])


'''
tests data name.test.csv with given k
prints error rate
saves result in ./results/name.results.csv
'''


def testData(name, k):
    # builds tree from data, finds k nearest neighbours for all points in data
    data = numpyTestData(name)
    tree = kdTree(data)

    # searches for k + 1 nearest neighbours and ignores closest, since closest is the point itself
    kNNList = np.array([np.array(knn(tree, point, k + 1)) for point in data])[:, 1:]

    # classifies data accordingly and casts to ndarray
    testResults = np.array(classifyListCompFromKNN(data, kNNList))

    # saves classification and coordinates of points together
    csvResults = np.c_[testResults[:, 0].astype(int), data[:, 1:-1]]
    # results including old classification for graphic interpretation
    # graphicResults = np.c_[testResults[:, 0].astype(int), data[:,:-1]]

    # saves results with at most 1 trailing zero for a prettier csv
    formatList = ['%4d']
    formatList.extend(['%1.7f'] * (data.shape[1] - 2))

    # make dir results if not exists
    if not os.path.exists('results'):
        os.mkdir('results')

    # save results as csv
    np.savetxt("results/{}.results.csv".format(name), csvResults, delimiter=', ', fmt=formatList)
    # extend formatlist for graphic results and save those results
    # formatList.extend(['%1.7f'])
    # np.savetxt("{}.graphicresults.csv".format(name), graphicResults, delimiter=', ', fmt=formatList)

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


def main():
    listData()
    filename = readAndTestFilename()
    testK = testIntUserInput('Please choose a maximum value for k:')
    testL = testIntUserInput('please choose a number of partitions for our training data, 5 is suggested:')
    print("")
    t1 = time.time()
    print("testing for k <= {} and i <= {} and data {}".format(testK, testL, filename))
    classify(filename, testK, testL)
    print("Total runtime was {:1.3f} seconds".format(time.time() - t1))


main()
