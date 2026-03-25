import numpy as np 
from pandas import Series
from scipy.stats import norm, t as t_dist
from datetime import datetime 
from tqdm import tqdm
from concurrent import futures
from typing import Literal
from arch import arch_model

windowsize = 1000
alpha = 0.01
horizon = 1 

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
    
    indicies = list(range(window, len(returns) - horizon + 1))
    
    def process(ti):
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
    
    with futures.ProcessPoolExecutor(max_workers=n_jobs) as executor: # type: ignore
        futures = [executor.submit(process,t) for t in indicies]  # type: ignore
        
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
   