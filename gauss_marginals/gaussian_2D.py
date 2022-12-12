# Gaussian 2D

import numpy as np
import matplotlib.pylab as pl
import ot
import ot.plot
import time
from sklearn.metrics import mean_squared_error

n = 100

mu_s = np.array([0, 0])
cov_s = np.array([[1, 0], [0, 1]])

mu_t = np.array([4, 4])
cov_t = np.array([[1, -.8], [-.8, 1]])

xs = ot.datasets.make_2D_samples_gauss(n, mu_s, cov_s)
xt = ot.datasets.make_2D_samples_gauss(n, mu_t, cov_t)

a, b = np.ones((n,)) / n, np.ones((n,)) / n  # uniform distribution on samples
M = ot.dist(xs, xt)

lambd = 0.001

start_time = time.time()
G0 = ot.emd(a, b, M)
end_time = time.time()
G0_time = end_time - start_time

start_time = time.time()
GS = ot.sinkhorn(a, b, M, lambd)
end_time = time.time()
GS_mse = mean_squared_error(GS, G0)
GS_time = end_time - start_time

start_time = time.time()
GG = ot.bregman.greenkhorn(a, b, M, lambd)
end_time = time.time()
GG_mse = mean_squared_error(GG, G0)
GG_time = end_time - start_time

print("G0_Time: " + str(G0_time))
print("GS_MSE: " + str(GS_mse))
print("GS_Time: " + str(GS_time))
print("GG_MSE: " + str(GG_mse))
print("GG_Time: " + str(GG_time))
