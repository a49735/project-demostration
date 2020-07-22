#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 08:50:53 2019

@author: zhangcheng
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pylab as plt


S=100
sigma=0.25
r=0.05
T=1/12
K=100
N1=21
N2=84

def simulate_stock_process(paths,steps,S,r,sigma,):
    'Simulate stock moving process in the context of\
    Black Scholes'
    
    deltaT=T/steps
    t=np.linspace(0,T,num=steps+1)
    X=np.c_[np.zeros((paths,1)),np.random.randn(paths,steps)]
    w=np.cumsum(np.sqrt(deltaT)*X,axis=1)
    return S*np.exp((r-1/2*sigma**2)*t+sigma*w)




def hedged_call(S, K, r, sigma,steps):
    'Dynamic hedging strategy, it returns fair value of \
    call, Delta and bond value that are shorting'
    
    t=np.linspace(0,T,num=steps+1)
    d1 = (np.log(S/K)+(r+sigma**2/2)*(T-t)) / (sigma*np.sqrt(T-t))
    d2 = d1 - sigma*np.sqrt(T-t)
    return S*norm.cdf(d1)- K*np.exp(-r*(T-t))*norm.cdf(d2), norm.cdf(d1),K*np.exp(-r*(T-t))*norm.cdf(d2)

def hedging_error(N):
    np.random.seed(0)
    deltaT=T/N
    St=simulate_stock_process(50000,N,S,r,sigma)
#    return call,delta and bond ndarray with Shape (50000,N+1)
    pcall,delta,bond=hedged_call(St,K,r,sigma,N)
#    The portfolio rebalances untill time T
    delta=delta[:,:-1]
    pcall=pcall[:,:-1]
#    the bond will grow at rf for each t
    fbond=bond*np.exp(r*deltaT)
#    profit and loss if no rebalancing made
    pnl=delta*St[:,1:]-delta*St[:,:-1]-(fbond-bond)[:,:-1]
#    the costs from implementing the rebalance to ensure delta hedged 
    cost=delta[:,1:]*St[:,1:-1]-delta[:,:-1]*St[:,1:-1]-(bond[:,1:]-fbond[:,:-1])[:,:-1]
#    cash balance after deduct costs from P&L
    cashbalance=pnl[:,:-1]-cost
#    the inital call premium should be included into cash balance
#    time 0, it will just grow at risk free rate untill T. 
    cashbalance[:,0]=cashbalance[:,0]+pcall[0,0]
#    All cash postions are expected to grow at risk free rate
    interval=len(cashbalance[1,:])
    futurecash=[]
    for i in range(interval):
        futurecash.append(cashbalance[:,i]*np.exp(r*deltaT*(interval-1-i)))
    futurecash=np.array(futurecash).T
#    the P&L at time T
    finalpnl=np.sum(futurecash,axis=1)
#    the hedging error
    return finalpnl-np.maximum(St[:,-1]-K,0)

er21=hedging_error(N1)
er84=hedging_error(N2)

plt.rcParams['figure.figsize']=[15,5] 
fig,ax=plt.subplots(1,2,sharey=True)
plt.suptitle('Black Scholes dynamic hedging error')
ax[0].grid()
ax[1].grid()
ax[0].hist(er21,range=(-1.5,1.5),density=True,color='r')
ax[0].set_xlabel('Final profit/loss')
ax[0].set_ylabel('Probability Density')
ax[0].set_title('21 rebalancing trades')
ax[1].hist(er84,range=(-1.5,1.5),density=True,color='r')
ax[1].set_title('84 rebalancing trades')
ax[1].set_xlabel('Final profit/loss')
fig.tight_layout()
plt.show()
fig.savefig('Hedging_error',bbox_inches = 'tight')

mean1,std1=np.mean(er21), np.std(er21)
mean2,std2=np.mean(er84), np.std(er84)

