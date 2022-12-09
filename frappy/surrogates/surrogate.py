# -*- coding: utf-8 -*-
"""
Created on Mar 17 2021

@author: SVG
"""
import numpy as np
import random
def acf1(x,t,bs):
    """
    Returns autocorrelation at lag 1 for time series with missing data
    Parameter
    ---------
    x : 1-d numpy array
        Input time series
    
    t : 1-d numpy array
        Value of time
    bs : float
        Bin size of the time series. 
    Returns
    -------
    rk: float
    	Lag-1 autocorrelation
    """
    mx=np.mean(x)
    denom=0.
    rk=0.
    dis=bs
    ctr=0
    size=[]
    x=[p-mx for p in x]
    xp=[]
    l=1
    for i in range (0,len(x)-l):
        size.append(t[i+l]-t[i])
        if(t[i+l]-t[i]<=dis+(bs/2.)):
            xp.append(x)
            rk=rk+((x[i])*(x[i+l]))
            ctr=ctr+1
        denom=denom+(x[i]*x[i])
    rk=rk/denom
    return (rk)
def ar1(x0,rk,var):
    """
    Returns next ar1 time point for given parameters.
    Parameter
    ---------
    x0 : float
        previous point
    
    rk : float
        Dependence on previous value
    var : float
        Variance of the noise 
    Returns
    -------
    rk*x0+np.random.normal(0,sd): float
    	Next point in the time series
    """
    sd=np.sqrt(var)
    return (rk*x0+np.random.normal(0,sd))
def ar1surr(x,t=-1,bs=1):
    """
    Returns a surrogate dataset preserving autocorrelation at lag 1.
    Parameter
    ---------
    x : 1-d numpy array
        Input time series
    
    t : 1-d numpy array
        Value of time
    bs : float
        Bin size of the time series. 
    Returns
    -------
    x1: 1-d numpy array
    	Surrogate time series
    """
    x=np.array(x)
    if(t==-1):
        t=np.arange(len(x))
        if(bs!=1):
            t=t*bs
    rk=acf1(x,t,bs)
    l1=len(x)
    var_s=np.var(x)
    var_n=(1-rk**2)*var_s
    x0=np.random.choice(x)
    x1=np.zeros(l1)
    x1[0]=x0
    for i in range (1,l1):
        x1[i]=(ar1(x0,rk,var_n))
    return (x1)

def random_surr(x):
    """
    Returns a shuffled surrogate.
    Parameter
    ---------
    x : 1-d numpy array
        Input time series 
    Returns
    -------
    np.random.shuffle(x): 1-d numpy array
    	Shuffled surrogate time series
    """
    return(np.random.shuffle(x))
