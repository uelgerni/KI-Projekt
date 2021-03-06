import time
'''
python file for our "helper functions" so our other methods and files aren't too cluttered
'''


# calculates euclidean distance between two points, ignoring the first and last entry (flag and key
# and not taking the root since the root is monotone anyways HAHAHAH JUST KIDDING IT WONT WORK THAT WAY :))
cdef double distance(p1, p2):
    usefulP1 = p1[1:-1]
    usefulP2 = p2[1:-1]
    return np.sqrt(sum((usefulP1 - usefulP2) ** 2))

# function to read our data and add keys
def numpyTrainingData(filename):
    data = np.genfromtxt('./data/{}.train.csv'.format(filename), delimiter=',', dtype=float)
    keys = np.arange(len(data))
    t = time.time()
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
sorts matrix by last column
'''


def sortListByKey(listToSort):
    sortedList = listToSort[dataBeautifier(listToSort)[0][:, -1].argsort()]
    return sortedList


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
