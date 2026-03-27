import numpy as np 
from numpy.linalg import solve
import pandas as pd
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
        self._arr: Optional[np.ndarray] = None
    
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
        
    def _filter(self,
                params: np.ndarray,
                returns: np.ndarray):
        
        k, garch, P = self._unpack(params)
        K = self.k_regimes
        T = len(returns)
        
        mu = np.array([gp['mu'] for gp in garch])
        omega = np.array([gp['omega'] for gp in garch])
        alpha = np.array([gp['alpha'] for gp in garch])
        beta = np.array([gp['beta'] for gp in garch])
       
        if self.dist == 't':
           nu = np.array([gp['nu'] for gp in garch])
        else:
            nu = None
            
        xi = self._stationary(P)
        denom = 1 - alpha - beta
        denom = np.maximum(denom, 1e-10)
        h = omega / denom
        h = np.maximum(h, 1e-12)
        
        xi_filtered = np.zeros((T,K), dtype=float)
        h_all = np.zeros((T,K), dtype=float)
        loglik = 0.0
        
        for t in range(T):
            r_t = returns[t]
            
            h_all[t] = h
            sqrt_h = np.sqrt(h)
            z = (r_t-mu) / sqrt_h
            
            if self.dist == 'normal':
                eta = norm.pdf(z) / sqrt_h
            else:
                 eta = t_dist(z, df=nu) / sqrt_h
                
            eta = np.maximum(eta, 1e-300)
            
            numerator = xi * eta
            total = numerator.sum()
            
            if total <= 0 or not np.isinf(total):
                return -1e10, xi_filtered, h_all
            
            loglik += np.log(total)
            xi_filtered[t] = numerator / total
            
            xi = np.transpose(P) @ xi_filtered[t]
            xi = np.maximum(xi, 1e-10)
            xi /= xi.sum()
            
            
            e2 = (r_t - mu) **2
            h = omega + alpha * e2 + beta * h
            h = np.maximum(h, 1e-10)
            
        return loglik, xi_filtered, h_all
    
    def _neg_loglik(self, params: np.ndarray):
        
        if not np.all(np.isfinite(params)):
            print("WARNING: Non-finite parameters received.")
            return 1e10
        
        _, garch, _ = self._unpack(params)
        K = self.k_regimes
        
        for _, p in enumerate(params[:K]):
            if not (1e-4 < p < 1-1e-4):
                return 1e10
        
        for gp in garch:
            if gp['omega'] < 0 or gp['alpha'] < 0 or gp['beta'] < 0:
                return 1e10
            if gp['alpha'] + gp['beta'] >= 0.99999:
                return 1e10
            if self.dist == 't' and gp.get('nu',10) <= 2.0:
                return 1e10
            
        ll, _, _ = self._filter(params, self._arr)
        return -ll
    
    def _bounds(self):
        K = self.k_regimes
        bounds = [(0.5, 0.99999)] * K
        
        for _ in range(K):
            bounds += [
                (-5.0, 5.0),  #mu
                (1e-8, 10.0),  # omega
                (1e-8, 0.5),  # alpha
                (1e-8, 0.99999),  #beta
            ]
            if self.dist == 't':
                bounds.append((2.01, 100))  # nu
        
        return bounds

    
    def _starting_values(self, returns: np.ndarray) -> np.ndarray:
        
        K = self.k_regimes
        abs_ret = np.abs(returns)
        quantiles = np.linspace(0, 100, K+1)[1:-1]
        thresholds = np.percentile(abs_ret, quantiles) if K > 2 else [np.percentile(abs_ret, 70)]
        
        
        cluster = [[] for _ in range(K)]
        for r in returns:
            placed = False
            for i, thr in enumerate(thresholds):
                if abs(r) <= thr:
                    cluster[i].append(r)
                    placed = True
                    break
            if not placed:
                cluster[-1].append(r)
        
        p_diag = np.full(K,0.95)
        
        garch_x0 = []
        for k in range(K):
            rd = np.array(cluster[k] if len(cluster[k]) > 30 else returns)
            mu_k = float(np.mean(rd))
            var_k = max(float(np.var(rd)), 1e-5)
            omega_k = var_k * 0.05
            garch_x0 += [mu_k, omega_k, 0.05, 0.9]
            if self.dist == 't':
                garch_x0.append(8.0)
                
        return np.concatenate(p_diag, garch_x0)
    
    def fit(self,
            returns :Series,
            verbose: bool = True,
            start_params: Optional[np.ndarray] = None
            ) -> "HaasMSGarch":
        
        from scipy.optimize import minimize
        from tqdm import tqdm
        
        self._returns = returns
        self._arr = np.array(returns)
        T = len(self._arr)
        
        if verbose:
            print(f'Fitting Haas Markov_Switching GARCH(1,1) Model:'
                  f' K = {self.k_regimes}, dist={self.dist}'
                  f'\nParameters: {self.n_params} | Observations: {T}')
            
        best_ll = -np.inf
        best_result = None
        
        if start_params is None:
            x0_base = self._starting_values(self._arr)
        else:
            x0_base = start_params.copy()
            if len(x0_base) != self.n_params:
                raise ValueError(f'start param length {len(x0_base)} != {self.n_params}')
            
        total_evals = 0 
        
        pbar = tqdm(desc='Fitting MS_GARCH.', total=1000)
        
        def callback(xk: np.ndarray):
            pbar.update(1)
        
        results = minimize(
            self._neg_loglik,
            x0=x0_base,
            method='L-BFGS-B',
            bounds=self._bounds(),
            callback=callback,
            options={'maxiter': 1000}
        )
        total_evals += results.nfev
        
        if verbose:
            status = "Y" if results.success else "N"
            print(f'Status: {status}'
                f"LL = {results.fun:.4f}")
            
        if -results.fun > best_ll:
            best_ll = -results.fun
            best_result = results
        
        if verbose:
            print(f"Total function evaluations: {total_evals}")
        
        if best_result is None or not np.isinf(best_ll):
            raise RuntimeError("Optimisation failed to converge.")
        
        self.params_ = best_result.x
        self.loglik_ = best_ll
        
        def AIC(ll: float, n_params: int):    
            
            AIC = 2.0 * (n_params - ll)
            return  AIC
        
        self.aic_ = AIC(self.loglik_, self.n_params)
        
        def BIC(ll: float, n_params: int, T: int):
            
            BIC = n_params*np.log(T) - 2.0*ll
            return BIC
        
        self.bic_ = BIC(self.loglik_, self.n_params, T)
        
        _, xi_filtered, h_all = self._filter(self.params_, self._arr)
        
        self.filtered_probs_ = pd.DataFrame(
            xi_filtered,
            index = returns.index,
            columns=[f"P(S={k})" for k in range(self.k_regimes)]
        )
        
        self.h_ = pd.DataFrame(
            h_all, 
            index=returns.index,
            columns=[f"h_{k}" for k in range(self.k_regimes)]
        )
        
        _, garch, _ = self._unpack(self.params_)
        unc_var = np.array([
            gp['omega'] / max((1.0-gp['alpha']-gp['beta'], 1e-5))
            for gp in garch
        ])
        order = np.argsort(unc_var)
        self.regime_labels = {
            'low_vol': int(order[0]),
            'high_vol': int(order[1])
        }
        
        self.is_fitted = True
        return self
    
    def summary(self):
        if not self.is_fitted:
            raise RuntimeError("Call .fit() first")
        
        _, garch, P = self._unpack(self.params_)
        K = self.k_regimes
        T = len(self._arr)
        
        print('=' * 50)
        print('Haas MS-GARCH(1,1)')
        print('=' * 50)
        print(f'\nRegimes: {K}')      
        print(f'\nDistribution: {self.dist}') 
        print(f'\nSample: {self._returns.index[0].date()}')
        print(f'\nLog-Lik: {self.loglik_}')
        print(f'\nAIC: {self.aic_}')
        print(f'\nBIC: {self.bic_}')
        
            
            
        
            
                   
         
        
            
            