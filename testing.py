import timeit
import numpy as np

cy = timeit.timeit('main',setup='import main',number=1)


print(cy)


