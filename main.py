import timeit
# suppress runtime warning
import warnings
warnings.filterwarnings("ignore")
cy = timeit.timeit('main',setup='import main',number=1)
