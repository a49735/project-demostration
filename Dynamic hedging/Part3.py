#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 00:31:11 2019

@author: zhangcheng
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq,fsolve
import matplotlib.pylab as plt
import datetime as dt
import math
import scipy.integrate as integrate 

rate=pd.read_csv('discount.csv',header=0,parse_dates=False)
today = dt.date(2013, 8, 30)
expiry = dt.date(2015, 1, 17)
T = (expiry-today).days/365.0
S=846.9
sigma=0.257
def interpo(today,expiry):
    t=(expiry-today).days
    for d in rate['Day']:
        if d>=t:
            D1=d
            break
    idx=rate.index[rate['Day']==D1]
    r1=rate['Rate (%)'][idx].values
    r2=rate['Rate (%)'][idx-1].values
    idx2=rate.index[rate['Rate (%)'].values==r2]
    D2=rate['Day'][idx2].values
    R=r2+(t-D2)*(r1-r2)/(D1-D2)
    return R/100

R=interpo(today,expiry)

def BS_euro_deriv(S,r,t,sigma):
    p1=S**3*(np.exp(3*(r+(sigma**2))*t)*10**(-8))
    p2=0.5*(np.log(S)+(r-1/2*(sigma**2)*t))+10
    return np.exp(-r*t)*(p1+p2)

p=BS_euro_deriv(S,R,T,sigma)

def Bache_euro_deriv(S,r,t,sigma):
    bnd=-1/(sigma*np.sqrt(t))
    p1=(S**3+3*S**3*sigma**2*t)*10**(-8)
    p2=0.5*np.log(S)+0.5/np.sqrt(2*math.pi)*\
    (integrate.quad(lambda x:(1+sigma*np.sqrt(t)*x)*np.exp(-x**2/2),bnd,np.inf))[0]+10
    return p1+p2

p2=Bache_euro_deriv(S,R,T,sigma)


    
