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
from matrix_generators import generate_square_gaussian, generate_square_ones, generate_square_random
from matrix_scaling_stock import sinkhorn_log, greenkhorn
from matrix_scaling_modified import greenkhorn_inverse_gauss, greenkhorn_random

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
    
# runs a benchmark for a, b, M, lambd values (dev) 
def benchmark(a, b, M, lambd, iterations, hornIterations, error_allowed, verbose=False):
    results = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for __ in range(iterations):
        # network simplex solver 
        start_time = time.time()
        G0 = ot.emd(a, b, M)
        end_time = time.time()
        G0_time = end_time - start_time 

        # sinkhorn solver (bregman)
        start_time = time.time()
        GS = sinkhorn_log(a, b, M, lambd, numItermax=hornIterations, method="sinkhorn_log")
        end_time = time.time()
        GS_time = end_time - start_time
        GS_mse = mean_squared_error(GS, G0)

        # greenkhorn solver (bregman)
        start_time = time.time()
        GG = greenkhorn(a, b, M, lambd, hornIterations, error_allowed)
        end_time = time.time()
        GG_mse = mean_squared_error(GG, G0)
        GG_time = end_time - start_time

        # greenkhorn random
        start_time = time.time()
        GR = greenkhorn_random(a, b, M, lambd, hornIterations, error_allowed)
        end_time = time.time()
        GR_mse = mean_squared_error(GR, G0)
        GR_time = end_time - start_time

        # greenkhorn inverse gauss solver (custom)
        start_time = time.time()
        GIG = greenkhorn_inverse_gauss(a, b, M, lambd, hornIterations, error_allowed)
        end_time = time.time()
        GIG_mse = mean_squared_error(GIG, G0)
        GIG_time = end_time - start_time

        bench = [G0_time, GS_time, GG_time, GIG_time, GR_time, GS_mse, GG_mse, GIG_mse, GR_mse]
        # bench = [G0_time, GG_time, GR_time, GG_mse, GR_mse]
        results = [sum(x) for x in zip(results, bench)]     

    if verbose:
        print("G0_Time: " + str(results[0] / iterations))
        print("GS_Time: " + str(results[1] / iterations))
        print("GG_Time: " + str(results[2] / iterations))
        print("GIG_Time: " + str(results[3] / iterations))
        print("GR_Time: " + str(results[4] / iterations))
        print("GS_MSE: " + str(results[5] / iterations))
        print("GG_MSE: " + str(results[6] / iterations))
        print("GIG_MSE: " + str(results[7] / iterations))
        print("GR_MSE: " + str(results[8] / iterations))


    return [val / iterations for val in results]


# marginals always randomly generated 
def generate_marginals(size, max):
    a = np.random.randint(0, max, size=(size))
    b = np.random.randint(0, max, size=(size))

    # normalization
    a = a / np.sum(a)
    b = b / np.sum(b)
    return a, b

    
# generate cost matrix based on l2 distance between two marginals
def generate_l2_square_gaussian(a, b, size):
    # normalization
    a = a / np.sum(a)
    b = b / np.sum(b)

    # generate cost matrix
    M = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            M[i][j] = np.linalg.norm(a[i] - b[j])
    return M


# runs benchmark on const, random, and gaussian cost matrices with inputs for marginals
def benchmark_results(a, b, size, lambd, iterations, horn_iterations, error_allowed, verbose=True):
    print("M_C Results")
    M_C = generate_square_ones(size)
    M_C_results = benchmark(a, b, M_C, lambd, iterations, horn_iterations, error_allowed, verbose=True)

    print("M_R Results")
    M_R = generate_square_random(size)
    M_R_results = benchmark(a, b, M_R, lambd, iterations, horn_iterations, error_allowed, verbose=True)

    print("M_G Results")
    M_G = generate_square_gaussian(size)
    M_G_results = benchmark(a, b, M_G, lambd, iterations, horn_iterations, error_allowed, verbose=True)

    print("M_L2 Results")
    M_L2 = generate_l2_square_gaussian(a, b, size)
    M_L2_results = benchmark(a, b, M_L2, lambd, iterations, horn_iterations, error_allowed, verbose=True)

    return M_C_results, M_R_results, M_G_results, M_L2_results



size = 784 # 784 needed for image testing
max = 10000
lambd = 1
iterations = 5
horn_iterations = 10000
error_allowed = 0.1

a, b = generate_marginals(size, max)
benchmark_results(a, b, size, lambd, iterations, horn_iterations, error_allowed, verbose = False)

# print("Zero Results")
# zero_1, zero_2 = load_image("MNIST/0/1.jpg", "MNIST/0/21.jpg")
# benchmark_results(zero_1, zero_2, size, lambd, iterations, horn_iterations, error_allowed)

# print("One Results")
# one_1, one_2 = load_image("MNIST/1/3.jpg", "MNIST/1/6.jpg")
# benchmark_results(one_1, one_2, size, lambd, iterations, horn_iterations, error_allowed)

# print("Two Results")
# two_1, two_2 = load_image("MNIST/3/7.jpg", "MNIST/3/10.jpg")
# benchmark_results(two_1, two_2, size, lambd, iterations, horn_iterations, error_allowed)

# print("Three Results")
# three_1, three_2 = load_image("MNIST/3/7.jpg", "MNIST/3/10.jpg")
# benchmark_results(three_1, three_2, size, lambd, iterations, horn_iterations, error_allowed)

# print("Four Results")
# four_1, four_2 = load_image("MNIST/4/2.jpg", "MNIST/4/9.jpg")
# benchmark_results(four_1, four_2, size, lambd, iterations, horn_iterations, error_allowed)

# print("Five Results")
# five_1, five_2 = load_image("MNIST/5/0.jpg", "MNIST/5/11.jpg")
# benchmark_results(five_1, five_2, size, lambd, iterations, horn_iterations, error_allowed)

# print("Six Results")
# six_1, six_2 = load_image("MNIST/6/13.jpg", "MNIST/6/18.jpg")
# benchmark_results(six_1, six_2, size, lambd, iterations, horn_iterations, error_allowed)

# print("Seven Results")
# seven_1, seven_2 = load_image("MNIST/7/15.jpg", "MNIST/7/29.jpg")
# benchmark_results(seven_1, seven_2, size, lambd, iterations, horn_iterations, error_allowed)

# print("Eight Results")
# eight_1, eight_2 = load_image("MNIST/8/17.jpg", "MNIST/8/31.jpg")
# benchmark_results(eight_1, eight_2, size, lambd, iterations, horn_iterations, error_allowed)

# print("Nine Results")
# nine_1, nine_2 = load_image("MNIST/9/4.jpg", "MNIST/9/19.jpg")
# benchmark_results(nine_1, nine_2, size, lambd, iterations, horn_iterations, error_allowed)

