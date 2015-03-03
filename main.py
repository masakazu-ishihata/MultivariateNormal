#! /usr/bin/env python

################################################################################
# import
################################################################################
import numpy as np
from matplotlib import pyplot as pl
from MultivariateNormal import MultivariateNormal

################################################################################
# function
################################################################################
def isPSD(A):
    E, V = np.linalg.eigh(A)
    return np.all(E > 0)

def makePSD(A, n=1e-5):
    if isPSD(A):
        return A
    else:
        return makePSD(A + n * np.identity(len(A)), n=n*10)

################################################################################
# main
################################################################################
# dimension
d = 5

# mean
m = np.zeros(d)

# covariance matrix
c = np.random.randn(d)
C = np.array( [[c[i] * c[j] for i in xrange(d) ] for j in xrange(d)])
C = makePSD(C)

# Multivariate Normal Distribution
f = MultivariateNormal(m, C)

# Samples 
n_samples = 10000
X = f.sample(n_samples)

# x ~ N_d(m, C) => dot(a, x) ~ N(0, _) for any a \in R^d
n_bins    = 50
a = np.random.randn(d)
pl.hist(np.dot(a, X.T), bins=n_bins)
pl.show()
