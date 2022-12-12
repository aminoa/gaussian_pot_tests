# setting up gaussian filters (have a mean of 0) 
# test on the previous gaussian cost matrifx
# then test on the current gaussian matrix

import ot
import ot.plot
import numpy as np
import scipy as sp
# import matplotlib.pylab as pl
from ot.datasets import make_1D_gauss as gauss
from sklearn.metrics import mean_squared_error
import math
import time

# generates symmetric 2d gaussian based on exponential approximation - taken from https://stackoverflow.com/questions/7687679/how-to-generate-2d-gaussian-with-python
def generate_square_gaussian(size, fwhm = 3, center=None):
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

def benchmark(a, b, M):
    # network simplex solver 
    start_time = time.time()
    G0 = ot.emd(a, b, M)
    end_time = time.time()
    G0_time = end_time - start_time 
    print(G0)
    print(G0_time)

    # # sinkhorn/Greenkhorn solver (bregman)
    lambd = 0.001
    start_time = time.time()
    GS = ot.sinkhorn(a, b, M, lambd, numItermax=100000)
    end_time = time.time()

    GS_mse = mean_squared_error(GS, G0)
    GS_time = end_time - start_time

    start_time = time.time()
    GG = ot.bregman.greenkhorn(a, b, M, lambd, numItermax=100000)
    end_time = time.time()

    GG_mse = mean_squared_error(GG, G0)
    GG_time = end_time - start_time

    print("GS_MSE: " + str(GS_mse))
    print("GG_MSE: " + str(GG_mse))

    print("G0_Time: " + str(G0_time))
    print("GS_Time: " + str(GS_time))
    print("GG_Time: " + str(GG_time))

n = 10
x = np.arange(n, dtype=np.float64) # bin positions
max_cost = 10000
a = np.random.randint(0, max_cost, size=(n))
b = np.random.randint(0, max_cost, size=(n))
M = generate_square_gaussian(n)

# normalization
M /= np.sum(M)
a = a / np.sum(a)
b = b / np.sum(b)

benchmark(a, b, M)