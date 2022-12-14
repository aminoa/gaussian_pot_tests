import numpy as np
import random

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