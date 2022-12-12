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

#TODO: implement constant cost matrix, importing alternate datasets for testing

# generates random square matrix 
def generate_square_random(size):
    return np.random.random((size, size))

# generates symmetric 2d gaussian based on exponential approximation - taken from https://stackoverflow.com/questions/7687679/how-to-generate-2d-gaussian-with-python
def generate_square_gaussian(size, fwhm = 3, center=None):
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    M =  np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    return M / np.sum(M)

def benchmark(a, b, M, lambd, verbose=False):
    # network simplex solver 
    start_time = time.time()
    G0 = ot.emd(a, b, M)
    end_time = time.time()
    G0_time = end_time - start_time 
    # print(G0)
    # print(G0_time)

    # # sinkhorn/Greenkhorn solver (bregman)
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

    if verbose:
        print("GS_MSE: " + str(GS_mse))
        print("GG_MSE: " + str(GG_mse))

    print("G0_Time: " + str(G0_time))
    print("GS_Time: " + str(GS_time))
    print("GG_Time: " + str(GG_time))

    return G0_time, GS_time, GG_time, GS_mse, GG_mse, 

def generate_marginals(size, max):
    a = np.random.randint(0, max, size=(size))
    b = np.random.randint(0, max, size=(size))

    # normalization
    a = a / np.sum(a)
    b = b / np.sum(b)
    return a, b

results = [] 

size = 1000
max = 10000

a, b = generate_marginals(size, max)
M_R = generate_square_random(size)
M_G = generate_square_gaussian(size)

M_R_results = [0,0,0,0,0]
M_G_results = [0,0,0,0,0]

iter = 100

for i in range(iter): 
    G0_time, GS_time, GG_time, GS_mse, GG_mse = benchmark(a, b, M_R, 0.001)
    M_R_results[0] += G0_time
    M_R_results[1] += GS_time
    M_R_results[2] += GG_time 
    M_R_results[3] += GS_mse
    M_R_results[4] += GG_mse   

    G0_time, GS_time, GG_time, GS_mse, GG_mse = benchmark(a, b, M_G, 0.001)
    M_G_results[0] += G0_time
    M_G_results[1] += GS_time
    M_G_results[2] += GG_time 
    M_G_results[3] += GS_mse
    M_G_results[4] += GG_mse

for i in range(len(M_R)):
    M_R[i] /= iter
    M_G[i] /= iter

print("Iterations: " + str(iter))
print("Size: " + str(size))
print("Max Value for Array: " + str(max))

print("Random G0_Time: " + str(M_R_results[0]))
print("Random GS_Time: " + str(M_R_results[1]))
print("Random GG_Time: " + str(M_R_results[2]))
print("Random GS_MSE: " + str(M_R_results[3]))
print("Random GG_MSE: " + str(M_R_results[4]))

print("Gaussian G0_Time: " + str(M_G_results[0]))
print("Gaussian GS_Time: " + str(M_G_results[1]))
print("Gaussian GG_Time: " + str(M_G_results[2]))
print("Gaussian GS_MSE: " + str(M_G_results[3]))
print("Gaussian GG_MSE: " + str(M_G_results[4]))


# print(M_R)
# print(M_G)