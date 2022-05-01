import statsmodels.api as sm
import numpy as np

def multitesting(alpha, pvals, true_H0, procedure=None):
    nsamples = pvals.shape[0]
    dim = pvals.shape[1]
    if not procedure or str(procedure).upper() in ['RAW', 'NAIVE']:
        reject = (pvals < alpha)
    else:
        reject = np.empty_like(pvals, dtype=bool)
        for itest in range(nsamples):
            reject[itest], _, _, _ = sm.stats.multipletests(pvals[itest], alpha=alpha, method=procedure, is_sorted=False, returnsorted=False)
        
    # Notation from
    # https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values    
    _m = dim
    _m0 = np.sum(true_H0)
    _m1 = _m - _m0
    _FP = np.sum(reject & true_H0, axis=1)
    _FN = np.sum(~reject & ~true_H0, axis=1)
    _TN = _m0 - _FP
    _TP = _m1 - _FN
    _R = _TN + _FN

    keys = {'nCond':_m, 'nCondNeg':_m0, 'nCondPos':_m1}
    keys['FWER'] = np.mean(_FP > 0)
    keys['TPR'] = np.mean(_TP / _m1) # True positive rate (TPR), Recall, Sensitivity, probability of detection, Power.
    keys['FNR'] = np.mean(_FN / _m1) # False negative rate (FNR), Miss rate.
    keys['FPR'] = np.mean(_FP / _m0) # False positive rate (FPR), Fall-out, probability of false alarm
    keys['TNR'] = np.mean(_TN / _m0) # Specificity (SPC), Selectivity, True negative rate (TNR).
    keys['Prevalence'] = _m1 / _m
    keys['ACC'] = np.mean((_TP + _TN) / _m)
    keys['PPV'] = np.mean(_TP / (_m - _R)) # Positive predictive value (PPV), Precision.
    keys['FDR'] = np.mean(_FP / (_m - _R)) # False discovery rate (FDR).
    keys['FOR'] = np.mean(_FN / _R) # False omission rate (FOR).
    keys['NPV'] = np.mean(_TN / _R)  # Negative predictive value (NPV).
    keys['F1'] = 2 * keys['TPR'] * keys['PPV'] / (keys['TPR'] + keys['PPV'])
    return keys, reject