#! /usr/bin/env python

################################################################################
# import
################################################################################
import numpy as np

################################################################################
# Multivariate Normal Distribution
################################################################################
class MultivariateNormal:
    ########################################
    # new
    # N_d(m, C)
    ########################################
    def __init__(self, m, C):
        self.d = len(m)  # d-variate normal
        self.m = m       # mean d-dim ndarray
        self.C = C       # covariance (d,d)-dim ndarray

        I = np.identity(self.d)
        self.C_inv = np.linalg.solve(self.C, I)


    ########################################
    # Compute the conditional distribution given x[idx] = v
    # m, C = self.condition(idx, v)
    # s.t. x[-idx] | x[idx] = v ~ N_q(m, C) (q = self.d - len(idx))
    # idx = indices of given variables
    # v   = values of given variables
    ########################################
    def condition(self, idx, v):
        # masks
        a = np.arange(self.d)
        mask1 = np.in1d(a, idx, invert=True) # unseen variables
        mask2 = np.in1d(a, idx)              # given variables

        # m1 & m2
        m1 = self.m[mask1]
        m2 = self.m[mask2]

        # C11, C12, C21, C22
        C11 = self.C[mask1,:][:,mask1]
        C12 = self.C[mask1,:][:,mask2]
        C21 = C12.T
        C22 = self.C[mask2,:][:,mask2]

        # C22^-1
        I = np.identity( len(C22) )
        C22_inv = np.linalg.solve(C22, I)

        # m' = m1 + C12 C22_inv (v - m2)
        # C' = C11 - C12 C22_inv C21
        m = m1 + np.dot(C12, np.dot(C22_inv, v-m2))
        C = C11 - np.dot(C12, np.dot(C22_inv, C21))

        return m, C


    ########################################
    # Gibbs Sampling
    # X = self.gibbs(n)
    # X = (n, self.d)-dim ndarray
    # s.t. X[i] ~ N_d(m, C)
    ########################################
    def gibbs(self, n=1, burnout=1000, interval=10):
        X = np.zeros((n, self.d))

        # initialize & burnout
        x = np.array([np.random.normal(self.m[i], self.C[i,i]) for i in xrange(self.d)])
        self.gibbs_move(x, burnout)

        # sample n sample(s)
        for i in xrange(n):
            X[i] = self.gibbs_move(x, interval)

        return X


    ########################################
    # Move from x by gibbs update
    # x' = self.gibbs_move(x, n)
    # x' = x after n times move of Gibbs update
    ########################################
    def gibbs_move(self, x, n=1):
        for _ in xrange(n):
            for i in xrange(self.d):
                idx = np.where( np.arange(self.d) != i )
                m, C = self.condition(idx, x[idx])
                try:
                    x[i] = np.random.normal(m, C)
                except:
                    print m, C
                    raise

        return x

    ########################################
    # Smart Sampling method
    # X = self.sample(n)
    # X = (n, self.d)-dim ndarry
    # s.t. X[i] ~ N_d(m, C)
    ########################################
    def sample(self, n=None):
        L = np.linalg.cholesky(self.C)

        # single sample
        if n is None:
            X = np.random.randn(self.d)

        # multiple samples
        else:
            X = np.random.randn(self.d, n)

        return self.m + np.dot(L, X).T


    ########################################
    # Log likelihood
    # ll = self.loglikelihood(x)
    # ll = log p(x ; m, C)
    ########################################
    def loglikelihood(self, x):
        t1 = np.log( np.linalg.det(self.C) ) / 2
        t2 = np.dot( (x-self.m).T, np.dot(self.C_inv, (x-self.m))) / 2
        t3 = self.d * np.log(2 * np.pi) / 2
        return -(t1 + t2 + t3)
