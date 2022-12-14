import matrix_scaling_modified
import ot
import ot.plot
import numpy as np
import scipy as sp
# import matplotlib.pylab as pl
from ot.datasets import make_1D_gauss as gauss
from sklearn.metrics import mean_squared_error
import math
import time
from operator import add
from PIL import Image
from matrix_generators import generate_square_gaussian, generate_square_ones, generate_square_random

# runs a benchmark for a, b, M, lambd values (dev) 
def benchmark(a, b, M, lambd, hornIterations=100000, verbose=False):
    # network simplex solver 
    start_time = time.time()
    G0 = ot.emd(a, b, M)
    end_time = time.time()
    G0_time = end_time - start_time 

    # greenkhorn solver (bregman)
    start_time = time.time()
    GG = ot.bregman.greenkhorn(a, b, M, lambd, numItermax=hornIterations)
    end_time = time.time()
    GG_mse = mean_squared_error(GG, G0)
    GG_time = end_time - start_time

    # random greenkhorn solver (bregman)
    start_time = time.time()
    GR = matrix_scaling_modified.greenkhorn_modified(a, b, M, lambd, numItermax=hornIterations)
    end_time = time.time()
    GR_time = end_time - start_time
    GR_mse = mean_squared_error(GR, G0)


    if verbose:
        print("GG_MSE: " + str(GG_mse))
        print("GR_MSE: " + str(GR_mse))
        print("G0_Time: " + str(G0_time))
        print("GG_Time: " + str(GG_time))
        print("GG_Time: " + str(GG_time))

    return [G0_time, GR_time, GG_time, GR_mse, GG_mse]

# repeats benchmark (dev)
def repeated_benchmark(a, b, M, lambd, iterations, horn_iterations=100000, verbose=False):
    results = [0, 0, 0, 0, 0]
    for i in range(iterations):
        bench = benchmark(a, b, M, lambd, horn_iterations)
        results = [sum(x) for x in zip(results, bench)]

    if verbose:
        print("G0_Time: " + str(results[0]))
        print("GR_Time: " + str(results[1]))
        print("GG_Time: " + str(results[2]))
        print("GR_MSE: " + str(results[3]))
        print("GG_MSE: " + str(results[4]))
    
    return [val / iterations for val in results]

# marginals always randomly generated 
def generate_marginals(size, max):
    a = np.random.randint(0, max, size=(size))
    b = np.random.randint(0, max, size=(size))

    # normalization
    a = a / np.sum(a)
    b = b / np.sum(b)
    return a, b

# runs benchmark on const, random, and gaussian cost matrices with inputs for marginals
def benchmark_results(a, b, size, lambd, iterations, horn_iterations, verbose=True):
    print("M_C Results")
    M_C = generate_square_ones(size)
    M_C_results = repeated_benchmark(a, b, M_C, lambd, iterations, horn_iterations, verbose=True)

    print("M_R Results")
    M_R = generate_square_random(size)
    M_R_results = repeated_benchmark(a, b, M_R, lambd, iterations, horn_iterations, verbose=True)

    print("M_G Results")
    M_G = generate_square_gaussian(size)
    M_G_results = repeated_benchmark(a, b, M_G, lambd, iterations, horn_iterations, verbose=True)
    return M_C_results, M_R_results, M_G_results

size = 784 # needed for image testing
max = 10000
lambd = 100
iterations = 5
horn_iterations = 1000000

a, b = generate_marginals(size, max)
benchmark_results(a, b, size, lambd, iterations, horn_iterations)
