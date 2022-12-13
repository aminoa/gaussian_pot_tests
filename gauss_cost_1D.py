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
from operator import add
from PIL import Image

np.seterr(divide="ignore", invalid="ignore")

def load_image(file_path_1, file_path_2):
    img1 = np.asarray(Image.open(file_path_1))
    img2 = np.asarray(Image.open(file_path_2))
    img1 = np.reshape(img1, (len(img1) ** 2))
    img2 = np.reshape(img2, (len(img2) ** 2))
    # print(img1)
    img1 = img1 / np.sum(img1)
    img2 = img2 / np.sum(img2)
    return img1, img2
    
def generate_square_ones(size):
    M = np.ones(shape=(size, size))
    return M / np.sum(M)

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

# runs a benchmark for a, b, M, lambd values (dev) 
def benchmark(a, b, M, lambd, hornIterations=100000, verbose=False):
    # network simplex solver 
    start_time = time.time()
    G0 = ot.emd(a, b, M)
    end_time = time.time()
    G0_time = end_time - start_time 

    # sinkhorn solver (bregman)
    start_time = time.time()
    GS = ot.sinkhorn(a, b, M, lambd, numItermax=hornIterations)
    end_time = time.time()
    GS_time = end_time - start_time
    GS_mse = mean_squared_error(GS, G0)

    # greenkhorn solver (bregman)
    start_time = time.time()
    GG = ot.bregman.greenkhorn(a, b, M, lambd, numItermax=hornIterations)
    end_time = time.time()
    GG_mse = mean_squared_error(GG, G0)
    GG_time = end_time - start_time

    if verbose:
        print("GS_MSE: " + str(GS_mse))
        print("GG_MSE: " + str(GG_mse))
        print("G0_Time: " + str(G0_time))
        print("GS_Time: " + str(GS_time))
        print("GG_Time: " + str(GG_time))

    return [G0_time, GS_time, GG_time, GS_mse, GG_mse]

# repeats benchmark (dev)
def repeated_benchmark(a, b, M, lambd, iterations, horn_iterations=100000, verbose=False):
    results = [0, 0, 0, 0, 0]
    for i in range(iterations):
        bench = benchmark(a, b, M, lambd, horn_iterations)
        results = [sum(x) for x in zip(results, bench)]

    if verbose:
        print("G0_Time: " + str(results[0]))
        print("GS_Time: " + str(results[1]))
        print("GG_Time: " + str(results[2]))
        print("GS_MSE: " + str(results[3]))
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
lambd = 0.001
iterations = 5
horn_iterations = 1000000

# a, b = generate_marginals(size, max)
# benchmark_results(a, b, size, lambd, iterations, horn_iterations)

print("Zero Results")
zero_1, zero_2 = load_image("MNIST/0/1.jpg", "MNIST/0/21.jpg")
benchmark_results(zero_1, zero_2, size, lambd, iterations, horn_iterations)

print("One Results")
one_1, one_2 = load_image("MNIST/1/3.jpg", "MNIST/1/6.jpg")
benchmark_results(one_1, one_2, size, lambd, iterations, horn_iterations)

print("Two Results")
two_1, two_2 = load_image("MNIST/3/7.jpg", "MNIST/3/10.jpg")
benchmark_results(two_1, two_2, size, lambd, iterations, horn_iterations)

print("Three Results")
three_1, three_2 = load_image("MNIST/3/7.jpg", "MNIST/3/10.jpg")
benchmark_results(three_1, three_2, size, lambd, iterations, horn_iterations)

print("Four Results")
four_1, four_2 = load_image("MNIST/4/2.jpg", "MNIST/4/9.jpg")
benchmark_results(four_1, four_2, size, lambd, iterations, horn_iterations)

print("Five Results")
five_1, five_2 = load_image("MNIST/5/0.jpg", "MNIST/5/11.jpg")
benchmark_results(five_1, five_2, size, lambd, iterations, horn_iterations)

print("Six Results")
six_1, six_2 = load_image("MNIST/6/13.jpg", "MNIST/6/18.jpg")
benchmark_results(six_1, six_2, size, lambd, iterations, horn_iterations)

print("Seven Results")
seven_1, seven_2 = load_image("MNIST/7/15.jpg", "MNIST/7/29.jpg")
benchmark_results(seven_1, seven_2, size, lambd, iterations, horn_iterations)

print("Eight Results")
eight_1, eight_2 = load_image("MNIST/8/17.jpg", "MNIST/8/31.jpg")
benchmark_results(eight_1, eight_2, size, lambd, iterations, horn_iterations)

print("Nine Results")
nine_1, nine_2 = load_image("MNIST/9/4.jpg", "MNIST/9/19.jpg")
benchmark_results(nine_1, nine_2, size, lambd, iterations, horn_iterations)

