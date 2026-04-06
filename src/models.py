import numpy as np 
from numpy.linalg import solve
import pandas as pd
from pandas import Series
from scipy.stats import norm, t as t_dist
from tqdm import tqdm
from typing import Literal, Optional
from arch import arch_model


def in_sample_ms_var(
    params: Series,
    smoothed_probs: pd.DataFrame,
    k_regimes: int,
    alpha: float
    ) -> np.ndarray:
    from scipy.stats import norm
    from scipy.optimize import brentq
    K = k_regimes
    means = [params[f'const[{k}]'] for k in range(K)]
    vars = [params[f'sigma2[{k}]'] for k in range(K)]
    probs = smoothed_probs.values
    
    sigmas = np.sqrt(vars)
    sigmas = np.maximum(sigmas, 1e-8)
    
    n_obs = probs.shape[0]
    var_estimates = np.zeros(n_obs)
    #print(f"means: {means}, sigmas: {sigmas}")
    
    def mix_cdf_0(x: float, row_probs:np.ndarray) -> float:
        total = 0.0
        for prob, mu, sigma in zip(row_probs, means, sigmas):
            if prob > 0:
                total += prob * norm.cdf(x, loc=mu, scale=sigma)
        return total - alpha
        
    for t in range(n_obs):
        probs_t = probs[t,:]
        #print(f"t={t}, probs_t={probs_t}, sigmas: {sigmas}, means:{means}")
        
        if probs_t.sum() == 0:
            var_estimates[t] = np.nan
            continue
        
        a = -10
        b= 10
        f_a = mix_cdf_0(a,probs_t)
        f_b = mix_cdf_0(b,probs_t)
        #print(f"a={a}, b={b}, f_a={f_a}, f_b={f_b}")
        
        while f_a * f_b >= 0:
            a -= 1
            b += 1
            f_a = mix_cdf_0(a,probs_t)
            f_b = mix_cdf_0(b,probs_t)
            #print(f"  expanded: a={a}, b={b}, f_a={f_a}, f_b={f_b}")
        
        def f(x: float):
            return mix_cdf_0(x, probs_t)
        
        q = brentq(f=f,a=a, b=b)
        var_estimates[t] = -q
    
    return var_estimates
     

def _garch_process(
    ti: int,
    returns: Series,
    window: int,
    alpha: float,
    horizon: int,
    dist: Literal['normal', 't'],
    vol: Literal['GARCH', 'EGARCH'],
    GRJ: int = 0
    ):
    train = returns[ti-window:ti]
    model = arch_model(train, vol=vol, p=1, q=1,o=GRJ, dist=dist)
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
    returns:Series, 
    window:int =1000,
    alpha: float=0.01,
    horizon:int=1,
    dist:Literal['normal', "t"]='t',
    vol:Literal['GARCH', 'EGARCH'] ='GARCH',
    GRJ: int = 0,
    n_jobs: int = -1
    ) -> tuple[list[int],np.ndarray, np.ndarray]:
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
            horizon, dist,
            vol, GRJ)
                   for t in indicies]  # type: ignore
        
        results: dict[int, tuple[float,float]] = {}
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
    
    def cdf(x:float):
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
        
        
        results:dict[int, tuple[float,float]] = {}
        
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


def _fast_std_norm_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-.5 * z * z) / np.sqrt(2 * np.pi)
    
def _fast_std_t_pdf(
    z: np.ndarray,
    nu: np.ndarray,
    const: np.ndarray
    ) -> np.ndarray:
    
    term = z*z / (nu-2)
    log_term = np.log1p(term)
    return const * np.exp(-(nu+1) / 2 * log_term)

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
        self.params_: Optional[np.ndarray] = None
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
    
    def _pack(
        self,
        p_diag: np.ndarray,
        garch: list[dict[str,float]]
        ):
        parts = list(p_diag)
        for gp in garch:
            parts += [gp['mu'], gp['omega'], gp['alpha'], gp['beta']]
            if self.dist == 't':
                parts.append(gp['nu'])
        return np.array(parts, dtype=float)
   
    def _build_P(
        self,
        p_diag: np.ndarray
        ) -> np.ndarray:
        K = self.k_regimes
        P = np.zeros((K,K))
        
        for i in range(K):
            P[i,i] = p_diag[i]
            off = (1.0-p_diag[i]) / (K-1)
            for j in range(K):
                if j != i:
                    P[i,j] = off
        return P
    
    def _unpack(
        self,
        params: np.ndarray        
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
            
            
    def _stationary(
        self,
        P:np.ndarray
        ) -> np.ndarray:
        
        K = P.shape[0]
        A = P.T - np.eye(K)
        A[-1,:] = 1.0
        
        b = np.zeros(K)
        b[-1] = 1.0
        try:
            pi = solve(A, b)
        except Exception:
            pi = np.ones(K) / K
        
        pi = np.maximum(pi,1e-12)
        pi /= pi.sum()
        
        return pi
        
    def _filter(
        self,
        params: np.ndarray,
        returns: np.ndarray
        ):

        _, garch, P = self._unpack(params)
        K = self.k_regimes
        T = len(returns)
        
        mu = np.array([gp['mu'] for gp in garch])
        omega = np.array([gp['omega'] for gp in garch])
        alpha = np.array([gp['alpha'] for gp in garch])
        beta = np.array([gp['beta'] for gp in garch])
       
        if self.dist == 't':
            nu = np.array([gp['nu'] for gp in garch])
            from scipy.special import gamma
            const = gamma((nu+1) / 2) / (np.sqrt((nu-2) * np.pi) * gamma(nu/2))
        else:
            nu = None
            const = None
            
        xi = self._stationary(P)
        denom = 1 - alpha - beta
        denom = np.maximum(denom, 1e-8)
        h = omega / denom
        h = np.maximum(h, 1e-10)
        
        xi_filtered = np.zeros((T,K), dtype=float)
        h_all = np.zeros((T,K), dtype=float)
        loglik = 0.0
        
        for t in range(T):
            r_t = returns[t]
            
            h_all[t] = h
            sqrt_h = np.sqrt(h)
            z = (r_t-mu) / sqrt_h
            
            if self.dist == 'normal':
                eta = _fast_std_norm_pdf(z) / sqrt_h
            else:
                eta = _fast_std_t_pdf(z,nu=nu, const=const)/ sqrt_h

            eta = np.maximum(eta, 1e-300)

            numerator = xi * eta
            total = numerator.sum()

            if total <= 0 or not np.isfinite(total):
                print(f'total: {total}, eta: {eta}')
                return -1e10, xi_filtered, h_all
            
            loglik += np.log(total)
            xi_filtered[t] = numerator / total
            
            xi = np.transpose(P) @ xi_filtered[t]
            xi = np.maximum(xi, 1e-12)
            xi /= xi.sum()
            
            
            e2 = (r_t - mu) **2
            h = omega + alpha * e2 + beta * h
            h = np.maximum(h, 1e-10)
            
        return loglik, xi_filtered, h_all
    
    def _neg_loglik(
        self,
        params: np.ndarray
        ):
        
        if not np.all(np.isfinite(params)):
            print("WARNING: Non-finite parameters received.")
            return 1e10
        
        _, garch, _ = self._unpack(params)
        K = self.k_regimes
        
        for _, p in enumerate(params[:K]):
            if not (1e-4 < p < 1-1e-4):
               # print(f'p is {p}')
                return 1e10
        
        for gp in garch:
            if gp['omega'] < 0 or gp['alpha'] < 0 or gp['beta'] < 0:
                #print(f' Omega: {gp['omega']} alpha:  {gp['alpha']} beta: {gp['beta']}')
                return 1e10
            if gp['alpha'] + gp['beta'] >= 0.98:
                #print(f'Persistence is: {gp['alpha'] + gp['beta']}')
                return 1e10
            if self.dist == 't' and gp.get('nu',10) <= 2.0:
                #print(f"nu is: {gp['nu']}")
                return 1e10
            
        ll, _, _ = self._filter(params, self._arr)
        return -ll
    
    def _bounds(
        self
        ):
        K = self.k_regimes
        bounds = [(0.7, 0.9999)] * K
        
        for _ in range(K):
            bounds += [
                (-5.0, 5.0),  #mu
                (1e-8, 10.0),  # omega
                (1e-8, 0.45),  # alpha
                (1e-8, 0.97),  #beta
            ]
            if self.dist == 't':
                bounds.append((2.01, 100))  # nu
        
        return bounds

    
    def _starting_values(
        self,
        returns: np.ndarray
        ) -> np.ndarray:
        
        K = self.k_regimes
        abs_ret = np.abs(returns)
        quantiles = np.linspace(0, 100, K+1)[1:-1]
        thresholds = np.percentile(abs_ret, quantiles) if K > 2 \
        else [np.percentile(abs_ret, 70)]
        
        
        cluster: list[list[float]] = [[] for _ in range(K)]
        for r in returns:
            placed = False
            for i, thr in enumerate(thresholds):
                if abs(r) <= thr:
                    cluster[i].append(float(r))
                    placed = True
                    break
            if not placed:
                cluster[-1].append(float(r))
        
        p_diag = np.full(K,0.97)
        
        garch_x0: list[float] = []
        for k in range(K):
            rd = np.array(cluster[k] if len(cluster[k]) > 30 else returns)
            mu_k = float(np.mean(rd))
            var_k = max(float(np.var(rd)), 1e-5)
            omega_k = var_k * 0.05
            garch_x0 += [mu_k, omega_k, 0.08, 0.88]
            if self.dist == 't':
                garch_x0 += [5.0]
                
        return np.concatenate([p_diag, garch_x0])
    
    def fit(
        self,
        returns: Series,
        n_restarts: int= 3, 
        verbose: bool = True,
        start_params: Optional[np.ndarray] = None
        ) -> "HaasMSGarch":
    
        from scipy.optimize import minimize
        from tqdm import tqdm
        
        self._returns = returns.copy()
        self._arr = np.array(returns)
        T = len(self._arr)
        K = self.k_regimes
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
       
        for restarts in range(n_restarts):
            if restarts == 0:
                x0 = x0_base.copy()
            else:
                rng = np.random.default_rng(restarts)
                bounds_arr = np.array(self._bounds())
                x0 = x0_base.copy()
                
                noise = rng.normal(0,0.02, size=len(x0_base))
                
                for regime in range(K):
                    omega_idx = K + regime * (4 + (1 if self.dist == 't' else 0)) + 1
                    factor = np.exp(rng.normal(0,0.5))
                    x0[omega_idx] = x0_base[omega_idx] * factor
                    
                    if self.dist == 't':
                        nu_idx = omega_idx + 3
                        factor_nu = np.exp(rng.normal(0,0.3))
                        x0[nu_idx] = x0_base[nu_idx] * factor_nu
                        
                x0 = np.clip(x0 + noise,
                bounds_arr[:, 0] + 1e-6,
                bounds_arr[:, 1] - 1e-6)
                    
            if verbose:
                print(f'\nRestart {restarts + 1} / {n_restarts}'
                    f" \ninitial LL = { -self._neg_loglik(x0):.4f}")
            
            pbar = tqdm(desc='Fitting MS_GARCH.', total=300)
            def callback(xk: np.ndarray):
                pbar.update(1)
            
            results = minimize(
                self._neg_loglik,
                x0,
                method='L-BFGS-B',
                bounds=self._bounds(),
                callback=callback,
                options={'maxiter': 300,'ftol': 1e-8, 'gtol': 1e-5}
            )
            total_evals += results.nfev
            
            if verbose:
                status = "Y" if results.success else "N"
                print(f'Status: {status}'
                    f"\nLL = {-results.fun:.4f}")
                
            if -results.fun > best_ll:
                best_ll = -results.fun
                best_result = results
            
            if verbose:
                print(f"Total function evaluations: {total_evals}")
            
        if best_result is None or np.isinf(best_ll):
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
        print(f'\nRegimes: {K}')
        print('=' * 50)
        
        print(f"\nTransition Matrix P[i,j] = P(S_t = j| S_{{t-1}}=i):")
        header = "          " + "".join(f' Regime {j}' for j in range(K))
        print(header)  
        for i in range(K):
            row = f' Reg[{i}]:' + ' '.join(f'{P[i,j]:4f} '  for j in range(K))
            print(row)
        
        print('-' * 50)
        
        print('\nRegime Classification:')
        classification = self.filtered_probs_.values.argmax(axis=1)
        print(f"Regime      Label     Obs     Share     ")
        print('-' *50)
        for k in range(K):
            label = 'low volatility' if k == self.regime_labels['low_vol'] else 'high_vol'
            n = int((classification == k).sum())
            print(f'{k}     {label}     {n}     {100 * n/T}%')
        
        print(f'\nGARCH(1,1) Parameters by Regime:')
        header = 'Regime     mu     omega      alpha     beta     persistence'
        if self.dist == 't':
            header += '     nu'
        print(header)
        print('-' * 50)
        for k in range(K):
            gp = garch[k]
            ab = gp['alpha'] + gp['beta']
            unc = gp['omega'] / max(1 - ab, 1e-8)
            row = f"{k:<8} {gp['mu']:<8.4f} {gp['omega']:<10.4f} {gp['alpha']:<8.4f} {gp['beta']:<8.4f} {ab:<12.4f}"
            if self.dist == 't':
                row += f" {gp['nu']:<8.3f}"
            print(row)

            print(f"    Unconditional variance = {unc:.4f} (sigma = {np.sqrt(unc):.4f})")
    
    def predict_var(
        self,
        confidence:float = 0.95
    ) -> float:
        
        if not self.is_fitted:
            raise RuntimeError('Call .fit() prior to forecastin .')
        
        from scipy.optimize import brentq
        from scipy.stats import norm, t
        
        _, garch, P = self._unpack(self.params_)  # type: ignore
        
        mu = np.array([gp['mu'] for gp in garch])
        omega = np.array([gp['omega'] for gp in garch])
        alpha = np.array([gp['alpha'] for gp in garch])
        beta = np.array([gp['beta'] for gp in garch])
        
        
        if self.dist == 't':
            nu = np.array([gp['nu'] for gp in garch])
        else:
            nu = None
            
        xi_T = self.filtered_probs_.values[-1]  # type: ignore
        pi_next = P.T @ xi_T
        pi_next = np.maximum(pi_next, 1e-10)
        pi_next /= pi_next.sum()
        
        h_t = self.h_.values[-1]  # type: ignore
        r_t = float(self._arr[-1])  # type: ignore
        e2 = (r_t - mu) ** 2
        h_next = omega + alpha * e2 + beta * h_t
        sigma_next = max(np.sqrt(h_next),1e-5)
        
        def mix_cdf(x: float) -> float:
            if self.dist == 'normal':
                component_cdfs = norm.cdf((x-mu) / sigma_next)  # type: ignore
            elif self.dist == 't':
                z = (x - mu) / sigma_next * np.sqrt(nu / (nu-2.0)) # type: ignore
                component_cdfs = t.cdf(z, df=nu)  # type: ignore
            
            return float(np.dot(pi_next,component_cdfs)) - (1-confidence)  # type: ignore
        
        low = float(np.min(mu) - 10.0 * np.max(sigma_next))
        high = float(np.max(mu) + 10.0 * np.max(sigma_next))
        
        expansions = 0
        while mix_cdf(low) * mix_cdf(high) < 0:
            if expansions >= 20:
                raise RuntimeError("No sign change. VaR cannot be computed.")
            else:
                low -= np.max(sigma_next)
                high += np.max(sigma_next)
                expansions += 1
        
        try:
            q = brentq(mix_cdf, low, high, maxiter=200)
        except ValueError:
            return np.nan
        
        return float(-q)
    

def _ms_garch_process(
    ti: int,
    returns: Series,
    k_regimes: int,
    window: int,
    alpha: float,
    dist: Literal['t', 'normal'],
    n_restarts: int
) -> tuple[int, float, float]:
    
    train = returns[ti - window: ti]
    model = HaasMSGarch(k_regimes=k_regimes, dist=dist)
    
    try:
        model.fit(train, n_restarts=n_restarts, verbose=False)
        var_forecast = (model.predict_var(confidence=(1-alpha)))
    except Exception:
        var_forecast = np.nan
    
    t = ti
    actual = float(returns.iloc[ti])
    return t, actual, float(var_forecast)

def rolling_ms_garch_var(
    returns: Series,
    k_regimes: int =2,
    window: int =500,
    alpha: float =0.05,
    dist: Literal['t', 'normal'] = 'normal',
    n_restarts: int = 1,
    n_jobs: int = -1
) -> tuple[list[int], np.ndarray, np.ndarray]:
    
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm
    
    indicies = list(range(window, len(returns)))
    max_workers = None if n_jobs == -1 else n_jobs
    
    results: dict[int, tuple[float, float]] = {}
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _ms_garch_process,
                t, returns, k_regimes, window, alpha, dist, n_restarts
            ): t
            for t in indicies
        }

        pbar = tqdm(total=len(futures), desc='Rolling MS-GARCH VaR')
        for future in as_completed(futures):
            t, actual, var_forecast = future.result()
            results[t] = (actual, var_forecast)
            pbar.update(1)
        pbar.close()

    sorted_indicies = sorted(results.keys())
    actuals = np.array([results[d][0] for d in sorted_indicies])
    var_forecasts= np.array([results[d][1] for d in sorted_indicies])
    return sorted_indicies, actuals, var_forecasts
