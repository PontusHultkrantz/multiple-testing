import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from matplotlib import pyplot as plt
import pandas as pd
from collections import Counter

factor_mu = 0.0 * np.arange(7)
factor_vol = 0.02 * np.eye(7)
noise_mu = 0.0
noise_vol = 0.005

nsamples = 36
nfactors = 7

factors = stats.multivariate_normal(mean=factor_mu, cov=factor_vol**2)
noise = stats.multivariate_normal(mean=noise_mu, cov=noise_vol**2)

weights = np.array([1.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0])
asset = lambda factor_innov: factor_innov @ weights # 1.0 * factors[:,0] + 0.5 * factors[:,1] + 0.30 * factors[:,2]
obs = lambda factor_innov, noise_innov: asset(factor_innov) + noise_innov

np.random.seed(0)
nsim = 2000
data = np.zeros((nsim, 8))
data2 = np.zeros(nsim)
alpha = 0.05
for i in range(nsim):
    print('{}%'.format(int(100*i/nsim))) if i % (nsim/10) == 0 else None
    factor_innov = factors.rvs(nsamples)
    noise_innov = noise.rvs(nsamples)
    obs_innov = obs(factor_innov, noise_innov)
    X = sm.add_constant(factor_innov)
    Y = obs_innov
    mdl = sm.OLS(Y, X)
    result = mdl.fit()
    # hypotest(alpha, result.pvalues)
    data[i] = result.pvalues


def testinference(alpha, pvals, true_H0, procedure=None):
    nsamples = pvals.shape[0]
    dim = pvals.shape[1]
    if not procedure:
        reject = (pvals < alpha)
    else:
        reject = np.empty_like(pvals, dtype=bool)
        for itest in range(nsamples):
            reject[itest], _, _, _ = sm.stats.multipletests(pvals[itest], alpha=alpha, method=procedure, is_sorted=False, returnsorted=False)
        
    # Notation from
    # https://en.wikipedia.org/wiki/Multiple_comparisons_problem#Classification_of_multiple_hypothesis_tests
    # https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values    
    _m = dim
    _m0 = np.sum(true_H0)
    _FP = np.sum(reject & true_H0, axis=1)
    _FN = np.sum(~reject & ~true_H0, axis=1)
    _TN = _m0 - _FP
    _R = _TN + _FN

    keys = {}
    keys['FWER'] = np.mean(_FP > 0)
    keys['FNR'] = np.mean(_FN / (_m - _m0)) # False negative rate (FNR), Miss rate.
    keys['FPR'] = np.mean(_FP / _m0) # False positive rate (FPR), Fall-out, probability of false alarm
    keys['FDR'] = np.mean(_FP / (_m - _R)) # False discovery rate (FDR).
    return keys

H0 = (weights == 0.0) #np.array([False, False, False, True, True, True, True])
print( testinference(alpha, data[:,1::], H0, procedure=None) )
print( testinference(alpha, data[:,1::], H0, procedure='bonferroni') )
print( testinference(alpha, data[:,1::], H0, procedure='sidak') )
# print( testerrors(alpha, data[:,1::], np.array([1, 1, 1, 0, 0, 0, 0]), procedure='holm') )
# print( testerrors(alpha, data[:,1::], np.array([1, 1, 1, 0, 0, 0, 0]), procedure='holm-sidak') )

print('h')