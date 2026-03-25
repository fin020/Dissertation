from pandas import Series
from scipy.stats import chi2
from scipy.optimize import minimize
import numpy as np
from typing import cast

def backtest_var(
    returns: Series[float], 
    var: Series[float]
    ):
    """
    Creates an Indicator Series for Value-at-Risk exceedance. 
    Value-at-Risk is a positive value. 
    
    Evaluated by: 
    I = 1 | r_t < - VaR_t
    I = 0 | r_t < -VaR_t 
    
    Parameters
    ----------
    returns: Series of float
        A Series of returns.
    var: Series of float
        A series of VaR thresholds.
        
    Returns
    -------
    hits
        Indicator Series of var exceedances. 
    """
    
    if len(returns) != len(var): 
        raise ValueError(f"Check length of returns and var are the same. var is length:{len(var)} ")
    if (var < 0).any():
        raise ValueError(f"Value-at-risk must be positive. your var entered is {var}")
    
    hits = returns < -var 
    return hits


def kupiec_test(I:Series, alpha:float) -> dict[str, float]:
    """
    Computes the Kupiec test (Kupiec, 1995) for unconditional coverage. This evaluates if the
    proportion of var exceedance matches the theoretical var confidence level. 
    
    Parameters
    ----------
    I : Series
        Indicator Series created from the backtest_var() function.
    alpha : float
        theoretical rate of VaR exceedance.
    
    Returns
    -------
    dict: 
        Likelihood ratio for the kupiec test.
        1% significance level.
        5% significance level.
        p_value.
    
    """
    T = len(I) #totals 
    x = I.sum() #number of exceedance
    p_hat = x / T # proportion of exceedance
    
    ll_H0 = (T - x) * np.log(1- alpha) + x * np.log(alpha)
    ll_H1 = (T - x) * np.log(1 - p_hat) + x * np.log(p_hat)

    lr_POF = -2*(ll_H0 - ll_H1)
    
    chi_result: np.ndarray = cast(np.ndarray,chi2.ppf([0.99, 0.95], df=1))
    
    p_value: float = cast(float, 1 - chi2.cdf(lr_POF, df=1))
    
    result: dict[str,float] = {"LR_test": lr_POF,
              r"1% significance:": chi_result[0],
              r"5% significance:": chi_result[1],
              "p-value": p_value}
    
    return result

def christofferssen_test(
    I: Series,
    )-> dict[str, float]:
    """
    Computes the Christofferssen test (Christofferssen, 1998) of independence.
    This evaluates if VaR exceedance clusters and if hits are independent. 
    
    Parameters
    ----------
    I : Series
        Hit sequence of VaR exceedance.
    
    Returns
    -------
    dict:  
        Likelihood ratio test statistic
        1% signficance level
        5% significance level
        p_value    
    """
    n_00 = 0
    n_01 = 0
    n_10 = 0
    n_11 = 0
    
    l = I.tolist()
    T = len(l) - 1
    for i in range(T): 
        if l[i] == 0 and l[i+1] == 1:
            n_01 += 1 
        elif l[i] == 1 and l[i+1] == 0:
            n_10 += 1
        elif l[i] == 0 and l[i+1] == 0:
            n_00 += 1
        elif l[i] == 1 and l[i+1] == 1:
            n_11 +=1
        
    
    if (n_10 + n_11) == 0 or (n_01 + n_00) == 0:
        raise ValueError('Insifficent violation clsuters for independence test')
    
    prob_failure_given_no_failure = n_01 / (n_00 + n_01)
    prob_failure_given_failure = n_11 / (n_10 + n_11)
    prob_failure = (n_01 + n_11) / (n_01 + n_11 + n_00 + n_10)
    
        
    ll_H0 = (n_00 + n_10) * np.log(1 - prob_failure) + (n_01 + n_11) * np.log(prob_failure)
    ll_H1 = n_00 * np.log(1- prob_failure_given_no_failure) + n_01 * np.log(prob_failure_given_no_failure) + n_10 * np.log(1-prob_failure_given_failure) + n_11 * np.log(prob_failure_given_failure)
        
    LR_CCI = -2 * (ll_H0 - ll_H1)
    
    chi_result: np.ndarray = cast(np.ndarray, chi2.ppf([0.99, 0.95], df=1))
    
    p_value: float = cast(float, 1 - chi2.cdf(LR_CCI, df=1))
    
    result: dict[str,float] = {"Christofferssen LR test": LR_CCI,
              r"1% significance:": chi_result[0],
              r"5% significance:": chi_result[1],
              "p-value": p_value}
    
    return result
    
    
def christofferssen_conditional_test(
    lr_pof: float,
    lr_ind: float
    ) -> dict[str, float]:
    """
    Computes the Christoffersen test of conditional coverage (Christofferssen, 1998).
    This is the sum of the LR ratios for kupiec test and independence test.
    This test statistic follows the Chi-squared distribution (2 degrees of freedom). 
    
    Parameters
    ---------
    lr_pof: float
        The likelihood ratio of proportion of failures or Kupiec test.
    lr_ind: float
        The likelihood ratio of the christofferssen independence test. 
    
    Returns
    -------
    result : dict
        Likelihood ratio of conditional coverage
        1% critical value
        5% critical value
        p_value
    """
    

    lr_cc = lr_pof + lr_ind
    p_value = 1 - chi2.cdf(lr_cc, df=2)
    
    result: dict[str,float] = {"LR_CC": lr_cc,
              r"1% critical": chi2.ppf(0.99, df=2),
              r"5% critical": chi2.ppf(0.95, df=2),
              "p-value": p_value} # type: ignore
    
    return result
    
def _compute_durations(I:Series):
    """
    Computes the duration between VaR exceedances. 
    Parameters
    ----------
    I: Series
        hit sequence of var exceedances
        
    Returns
    -------
    durations: Series
        Series of durations between var exceedances.
    """
    
    indicator = np.array(I)
    hit_indicies = np.where(indicator == 1)[0]
    
    if len(hit_indicies) < 2:
        return np.array([])
    durations = np.diff(hit_indicies)
    return durations

def duration_test_unconditional(I: Series, alpha: float):
    """
    Parameters
    ----------
    I : Series
        hit sequence series,
    alpha : float
        theoretical VaR proportion of exceedances. 
    
    Returns
    -------
    result : dict
        Likelihood ratio of durations
        p_value
        lambda hat
    """
    
    durations = _compute_durations(I)
    
    if len(durations) == 0:
        raise ValueError("Error. Not enough violations")
    
    lambda_hat: float = cast(float, 1 / np.mean(durations))
    
    ll_H0 = np.sum(np.log(alpha) - alpha * durations)
    ll_H1 = np.sum(np.log(lambda_hat) - lambda_hat * durations)
    
    LR = -2 * (ll_H0 - ll_H1)
    
    p_value: float = cast(float, 1 - chi2.cdf(LR,df=1))
    
    result: dict[str,float] = {  
        "LR_duration": LR,
        "p-value":p_value,
        "lambda_hat": lambda_hat
    }
    
    return result



def duration_test_conditional(I: Series):
    """
    
    Parameters
    ----------
    I : Series
        hit sequence. 
    
    Returns
    -------
    result : dict
        Likelihood ratio for the weibull distribution.
        p_value
        k_hat
        
    """
    
    durations = _compute_durations(I)
    
    if len(durations) == 0:
        raise ValueError("Not enough violations")
    
    def neg_ll(params: np.ndarray) -> float:
        k, lam = params
        
        if k <= 0 or lam <= 0:
            return 1e10
        
        ll = np.sum(
            np.log(k) - k*np.log(lam) +
            (k-1)*np.log(durations) -
            (durations / lam)**k
        )
        return -ll
    
    res = minimize(neg_ll, x0=[1.0, np.mean(durations)], method='L-BFGS-B')
    
    k_hat, lam_hat = res.x
    
    ll_H1 = -res.fun
    
    lam_exp = np.mean(durations)
    ll_H0 = np.sum(
        np.log(1/lam_exp) - durations / lam_hat
    )
    
    LR = -2 * (ll_H0 - ll_H1)
    
    p_value: float = cast(float, 1 - chi2.cdf(LR,df=1))
    
    result: dict[str,float] = {
        "LR_weibull": LR,
        "p-value":p_value,
        "k_hat": k_hat
    }
    return result
    
    
#model comps table