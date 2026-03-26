import numpy as np 
from numpy.linalg import solve
from pandas import Series
from scipy.stats import norm, t as t_dist
from datetime import datetime 
from tqdm import tqdm
from typing import Literal, Optional
from arch import arch_model

windowsize = 1000
alpha = 0.01
horizon = 1 

def _garch_process(
    ti: int,
    returns: Series,
    window: int,
    alpha: float,
    horizon: int,
    dist: Literal['normal', 't']):
    train = returns[ti-window:ti]
    model = arch_model(train, vol='GARCH', p=1, q=1, dist=dist)
    res = model.fit(disp='off')
    
    forecast = res.forecast(horizon=horizon)
    
    sigma = np.sqrt(forecast.variance.values[-1,0])
    mu = forecast.mean
    
    if dist == 't':
        nu = res.params['nu']
        scaling = np.sqrt((nu -2) / nu)
        VaR = mu + sigma * t_dist.ppf(alpha, df=nu) * scaling
    else: 
        VaR = mu + sigma * norm.ppf(alpha)
    
    return returns.index[ti], VaR, returns[ti]
    
    
def rolling_garch_var(
    returns:Series, window:int =1000,
    alpha: float=0.01, horizon:int=1,
    dist:Literal['normal', "t"]='t',
    n_jobs: int = -1
    ):
    """
    
    Parameters
    ----------
    returns : Series
        time series of returns.
    window : int = 1000
        Size of previous return values to include in the rolling model.
    alpha: float = 0.01
        theoretical proportion of VaR exceedances.
    horizon : int  = 1
        t (days) forecast of value-at-risk
    dist : Literal['normal', 't']  = 't'
        distribution used for error calculation.
    
    Returns
    -------
    tuple
        Array of var forecasts
        Array of actual returns
    
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed 
    indicies = list(range(window, len(returns) - horizon + 1))
    
    
    max_workers = None if n_jobs == -1 else n_jobs
    with ProcessPoolExecutor(max_workers=max_workers) as executor: # type: ignore
        futures = [executor.submit(
            _garch_process,
            t, returns,
            window, alpha,
            horizon, dist)
                   for t in indicies]  # type: ignore
        
        results = {}
        pbar = tqdm(total=len(futures), desc="Calculating rolling GARCH VaR")
        
        for future in as_completed(futures): # type: ignore
            t, VaR, actual = future.result() # type: ignore
            results[t] = (VaR, actual)
            pbar.update(1)
        pbar.close()
        
    sorted_indicies = sorted(results.keys())
    var_forecasts = [results[t][0] for t in sorted_indicies]
    actual = [results[t][1] for t in sorted_indicies]  
    
    return sorted_indicies, np.array(var_forecasts), np.array(actual)



def _ms_process(
    ti: int, 
    returns: Series,
    k_regimes: int=2,
    window: int= 500,
    alpha: float = 0.01,
    horizon: int = 1,
    min_variance: float = 1e-8,
    ):
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    from scipy.optimize import brentq
    from scipy.stats import norm
    train = returns[ti-window:ti]
        
    model = MarkovRegression(
        train,
        k_regimes=k_regimes,
        trend='c', 
        switching_variance=True
        )
    res = model.fit(maxiter=1000)
    
    param_names = model.param_names
    params = res.params
    
    means = np.array([
        params[param_names.index(f'const[{k}]')]
        for k in range(k_regimes)
    ])
    variances = np.array([
        params[param_names.index(f'sigma2[{k}]')]
        for k in range(k_regimes)
    ])
    
    if np.any(variances < 0):
        print(f"Variance IS NEGATIVE: {variances}")
    
    variances = np.maximum(variances, min_variance)
    sigmas = np.sqrt(variances)
    
    if np.any(sigmas <= 0) or np.any(np.isnan(means)) or np.any(np.isnan(sigmas)):
            print("sigma is zero or negative or paramters are NaN")
            return ti, np.nan, returns[ti]
        
    filtered_probs = res.filtered_marginal_probabilities.values[-1]
    
    trans_mat = res.regime_transition
    
    pi_next = filtered_probs @ trans_mat
    
    def cdf(x):
        return np.sum(pi_next * norm.cdf((x - means) / sigmas)) - alpha
    
    low = np.min(means) - 5 * np.max(sigmas)
    high = np.max(means) + 5 * np.max(sigmas)
    
    try:
        q = brentq(cdf, low, high, xtol=1e-12)
        var_forecast = -q
    except ValueError:
        print("ERROR WITH BRENTQ")
        
    return ti, var_forecast, returns[ti]
    
    
def rolling_ms_var(returns: Series,
    k_regimes: int=2,
    window: int= 500,
    alpha: float = 0.01,
    horizon: int = 1,
    min_variance: float = 1e-8,
    n_jobs: int = -1
    ):
    
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm
    
    indicies = list(range(window, len(returns) - horizon))
    
    max_workers = None if n_jobs == -1 else n_jobs
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(
            _ms_process,
            t, returns,
            k_regimes, window,
            alpha, horizon,
            min_variance)
                   for t in indicies]
        
        
        results = {}
        
        pbar = tqdm(total=len(futures), desc="Rolling Markov Switching model loading")
        
        for future in as_completed(futures):
            t, VaR, actuals = future.result()
            results[t] = (VaR, actuals)
            pbar.update(1)
        pbar.close()
        
    sorted_indicies = sorted(results.keys())
    var_results = [results[t][0] for t in sorted_indicies]
    actual = [results[t][1] for t in sorted_indicies]
    
    return sorted_indicies, np.array(var_results), np.array(actual)



class HaasMSGarch:
    """
    Haas (2004) Markov-Switchin Garch(1,1). with parallel processes.
    """
    
    def __init__(self, k_regimes: int=2, dist:str ='normal'):
        if k_regimes < 2:
            raise ValueError("k_regimes must be >= 2.")
        if dist not in ('normal', 't'):
            raise ValueError("dist must be 'normal' or 't'.")
        
        self.k_regimes = k_regimes
        self.dist = dist
        self.params_ = None
        self.filtered_probs_ = None
        self.h_ = None
        self.loglik_: Optional[float] = None
        self.aic_: Optional[float] = None 
        self.bic_: Optional[float] = None
        self.regime_labels: dict[str, int] = {}
        self._returns: Optional[Series] = None
        self.is_fitted: bool = False
    
    @property    
    def _n_garch(self) -> int:
        return 4
    
    @property
    def _n_regime_params(self) -> int:
        return self._n_garch + (1 if self.dist == 't' else 0)
    
    @property
    def n_params(self) -> int:
        return self.k_regimes + self.k_regimes * self._n_regime_params
    
    def _pack(self, p_diag: np.ndarray, garch: list[dict[str,float]]):
        parts = list(p_diag)
        for gp in garch:
            parts += [gp['mu'], gp['omega'], gp['alpha'], gp['beta']]
            if self.dist == 't':
                parts.append(gp['nu'])
        return np.array(parts, dtype=float)
   
    def _build_P(self, p_diag: np.ndarray) -> np.ndarray:
        K = self.k_regimes
        P = np.zeros((K,K))
        
        for i in range(K):
            P[i,i] = p_diag[i]
            off = (1.0-p_diag[i]) / (K-1)
            for j in range(K):
                if j != i:
                    P[i,j] = off
        return P
    
    def _unpack(self, params: np.ndarray
                ) -> tuple[np.ndarray, list[dict[str,float]], np.ndarray]:
        
        K = self.k_regimes
        p_diag = params[:K]
        P = self._build_P(p_diag)
        
        garch: list[dict[str,float]] = []
        idx = K
        
        for _ in range(K):
            gp: dict[str,float] = {
                'mu': params[idx],
                'omega': params[idx+1],
                'alpha': params[idx+2],
                'beta': params[idx+3]
            }
            idx += 4
            
            if self.dist == 't':
                gp['nu'] = params[idx]
                idx += 1
            garch.append(gp) 
                
        
        return p_diag, garch, P 
            
            
    def _stationary(self, P:np.ndarray) -> np.ndarray:
        K = P.shape[0]
        
        A = P.T - np.eye(K)
        A[-1,:] = 1.0
        
        b = np.zeros(K)
        b[-1] = 1.0
        try:
            pi = solve(A,b)
        except Exception:
            pi = np.ones(K) / K
        
        pi = np.maximum(pi,1e-12)
        pi /= pi.sum()
        
        return pi 
        
    def _density(self,
                 r: float,
                 mu: float,
                 h: float,
                 nu: Optional[float] = None) -> float:
        
        sigma = np.sqrt(max(h, 1e-12))
        z = (r-mu) / sigma
        
        if self.dist == 'normal' or nu is None:
            return float(norm.pdf(z) / sigma)
        else:
            scale = np.sqrt(nu / (nu-2.0))
            return float(t_dist.pdf(z * scale, df=nu) * scale / sigma)
        

    