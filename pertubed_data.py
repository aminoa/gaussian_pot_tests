# from gauss_cost_1D import benchmark, generate_marginals, benchmark_results
import numpy as np
import random
import ot
import ot.plot
import time
from matrix_scaling_stock import sinkhorn_log, greenkhorn
from sklearn.metrics import mean_squared_error
from matrix_generators import generate_square_gaussian, generate_square_ones, generate_square_random

# runs a benchmark for a, b, M, lambd values (dev) 
def benchmark(a, b, M, lambd, iterations, hornIterations, error_allowed, verbose=False):
    results = [0, 0, 0, 0, 0]
    for __ in range(iterations):
        # network simplex solver 
        start_time = time.time()
        G0 = ot.emd(a, b, M)
        end_time = time.time()
        G0_time = end_time - start_time 

        # # sinkhorn solver (bregman)
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

        # grankhorn solver (custom)
        # start_time = time.time()
        # GR = greenkhorn_basic_rng(a, b, M, lambd, hornIterations, error_allowed)
        # end_time = time.time()
        # GR_mse = mean_squared_error(GR, G0)
        # GR_time = end_time - start_time

        bench = [G0_time, GS_time, GG_time, GS_mse, GG_mse]
        # bench = [G0_time, GG_time, GR_time, GG_mse, GR_mse]
        results = [sum(x) for x in zip(results, bench)]

        

    if verbose:
        print("G0_Time: " + str(results[0]))
        print("GS_Time: " + str(results[1]))
        print("GG_Time: " + str(results[2]))
        print("GS_MSE: " + str(results[3]))
        print("GG_MSE: " + str(results[4]))

        # print("G0_Time: " + str(results[0]))
        # print("GG_Time: " + str(results[1]))
        # print("GR_Time: " + str(results[2]))
        # print("GG_MSE: " + str(results[3]))
        # print("GR_MSE: " + str(results[4]))

    return [val / iterations for val in results]

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
    return M_C_results, M_R_results, M_G_results

size = 1000 #10000 takes a bit more time
max = 1000
lambd = 1
iterations = 5
horn_iterations = 10000
error_allowed = 0.1


# a needs to remain consistent, can just be 1/n for all entries

# b - 1/n for all entries but you pertube by a  * 1/n?
# [0.2, 0.2, 0.2, 0.2, 0.2]
# [x * random(0, x * 0.2)]

for i in range(1, 11):
    perc_factor = i / 10

    print("Percentage Factor: ", perc_factor)

    a = np.full((size), 1/size)
    a = a / np.sum(a) 

    b = np.copy(a)
    for i in range(len(b)):
        b[i] += b[i] * random.randrange(int(-1000 * perc_factor), int(1000 * perc_factor))/1000
    b = b / np.sum(b)

    benchmark_results(a, b, size, lambd, iterations, horn_iterations, error_allowed, verbose = False)
