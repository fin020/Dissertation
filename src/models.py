import numpy as np 
from pandas import Series
from scipy.stats import norm, t as t_dist
from datetime import datetime 
from tqdm import tqdm
from typing import Literal
from arch import arch_model

windowsize = 1000
alpha = 0.01
horizon = 1 

def rolling_garch_var(
    returns:Series, window:int =1000,
    alpha: float=0.01, horizon:int=1,
    dist:Literal['normal', "t"]='t'
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
    
    returns = np.array(returns) # type: ignore
    
    var_forecasts: list[float] = []
    actual: list[float] = []
    dates: list[datetime] = []
    
    for ti in tqdm(range(window, len(returns)-horizon)):
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
        
        var_forecasts.append(VaR)
        actual.append(returns[ti])
        dates.append(returns.index[ti])
        
    return np.array(var_forecasts), np.array(actual), np.array(dates)
   