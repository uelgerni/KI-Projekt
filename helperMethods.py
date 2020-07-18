import os
from math import sqrt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
python file for our "helper functions" so our other methods and files aren't too cluttered
'''


# calculates euclidean distance between two points, ignoring the first and last entry (flag and key
# and not taking the root since the root is monotone anyways HAHAHAH JUST KIDDING IT WONT WORK THAT WAY :))
def distance(p1, p2):
    usefulP1 = p1[1:-1]
    usefulP2 = p2[1:-1]
    return sqrt(sum((usefulP1 - usefulP2) ** 2))  # 19 sec


# function to read our data and add keys
def numpyTrainingData(filename):
    data = np.genfromtxt('./data/{}.train.csv'.format(filename), delimiter=',', dtype=float)
    keys = np.arange(len(data))
    keyedData = np.c_[data, keys]

    return keyedData


# function to read our test data
def numpyTestData(filename):
    data = np.genfromtxt('./data/{}.test.csv'.format(filename), delimiter=',', dtype=float)
    keys = np.arange(len(data))
    keyedData = np.c_[data, keys]
    return keyedData


# same but as a dataframe, only needed for plotting 2d
def pandasReader(filename):
    dataframe = pd.read_csv('./data/{}.train.csv'.format(filename), header=None)
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


'''
just a little data beautification thats needed multiple times
'''


def dataBeautifier(data):
    return np.array(np.array(data)[:, 1].tolist()), np.array(np.array(data)[:, 0].tolist())


'''
lists all training files in ./data directory
'''


def listData():
    for file in os.listdir('data'):
        if file.endswith('train.csv'):
            print(file[:-10])


'''
reads a file name, checks if exists, if not reads a file name ...
'''


def readAndTestFilename():
    while True:
        filename = input('please choose one of the above files by typing its name:\n')
        file = 'data/' + filename + '.train.csv'
        if not os.path.isfile(file):
            print('The file {} does not exist in the data directory, please try again with one of the above'.format(
                filename))
            continue
        else:
            break
    return filename


'''
checks if user input is int, reads again until it is
'''


def testIntUserInput(prompt):
    while True:
        try:
            number = int(input(prompt + '\n'))
        except ValueError:
            print('not an integer, try again')
            continue
        else:
            break
    return number


'''

function to plot the results
'''


def plotErrorRate(errorAVG,graphicResults):
    if np.shape(graphicResults)[1] < 3:
        
        fig = plt.figure(figsize = (13,8),dpi = 300)
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)
        ax1.plot(errorAVG[:,1],errorAVG[:,0]*100)
        originalGR = graphicResults[np.argsort(graphicResults[:,1])]
        classifiedGR= graphicResults[np.argsort(graphicResults[:,0])]
        ax2.scatter(classifiedGR[0:(np.argmax(classifiedGR[:,0])-1),2],classifiedGR[0:(np.argmax(classifiedGR[:,0])-1),3],s = 2,facecolor= 'blue')
        ax2.scatter(classifiedGR[np.argmax(classifiedGR[:,0]):-1,2],classifiedGR[np.argmax(classifiedGR[:,0]):-1,3], s = 2,facecolor='red' )
        ax2.scatter(originalGR[0:(np.argmax(originalGR[:,1])-1),2],originalGR[0:(np.argmax(originalGR[:,1])-1),3],s= 0.25,facecolor = 'blue')
        ax2.scatter(originalGR[np.argmax(originalGR[:,1]):-1,2],originalGR[np.argmax(originalGR[:,1]):-1,3],s = 0.25,facecolor='red')
        ax1.set_title('plotted average error')
        ax2.set_title('classification')
        ax1.set_xlabel('k')
        ax1.set_ylabel('error rate average in %')
        ax2.set_xlabel('x-Axis')
        ax2.set_ylabel('y-Axis')
        ax2.legend(' classified blue ','original red')
        
        def zoomin(event):
            
            if event.button != 1:
                return               
            x, y = event.xdata, event.ydata
            length = ax2.get_xlim()[1]-ax2.get_xlim()[0]
            ax2.set_xlim(x - length*0.2, x + length*0.2)
            ax2.set_ylim(y - length*0.2, y + length*0.2)
            
            fig.canvas.draw()
            
            def zoomout(event):
                if event.button != 3 :
                    return
                x, y = event.xdata, event.ydata
                length = ax2.get_xlim()[1]-ax2.get_xlim()[0]
                ax2.set_xlim(x - length*0.8, x + length*0.8)
                ax2.set_ylim(y - length*0.8, y + length*0.8)
                fig.canvas.draw()        
                
                fig.canvas.mpl_connect('button_press_event', zoomin)
                fig.canvas.mpl_connect('button_press_event', zoomout)
                
                
        plt.show()
