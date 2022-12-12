# Gaussian 1D
# compare network simplex (exact solver)
# to Sinkhorn, Stochastic Sinkhorn, Greenkhorn

import ot
import ot.plot
import numpy as np
import matplotlib.pylab as pl
from ot.datasets import make_1D_gauss as gauss
from sklearn.metrics import mean_squared_error
import time

# want to test on more variations for the two distributions (probably have 4 tests based on differeing means/std)

# two distribtuions + cost matrix setup
n = 4
x = np.arange(n, dtype=np.float64) # bin positions

a = gauss(n, m=20, s=5)
b = gauss(n, m=60, s=10)

M = ot.dist(x.reshape((n, 1)), x.reshape((n, 1))) # squared euclidian distance metric
# M /= M.max()

# using network simplex solver for OT
start_time = time.time()
G0 = ot.emd(a, b, M)
end_time = time.time()
G0_time = end_time - start_time 

# sinkhorn/Greenkhorn solver (bregman)
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

print("G0_Time: " + str(G0_time))
print("GS_MSE: " + str(GS_mse))
print("GS_Time: " + str(GS_time))
print("GG_MSE: " + str(GG_mse))
print("GG_Time: " + str(GG_time))


# a = gauss(n, m=20, s=5)
# b = gauss(n, m=60, s=10)
# M = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)))

# G0_Time: 0.015637874603271484
# GS_MSE: 1.2099206105724785e-06
# GS_Time: 0.0628960132598877
# GG_MSE: 1.209920610451525e-06
# GG_Time: 0.742847204208374
