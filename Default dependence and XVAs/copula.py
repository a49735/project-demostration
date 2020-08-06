# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from scipy import stats


def default_prob_norm(n,y,rho, default=True):
    count = 0
    if rho==0:
        X = np.random.normal(size = (assets,simulation))
        U = stats.norm.cdf(X)
    else:
        C_ = np.identity(assets)
        C = np.where(C_==0, rho,1)
        L = np.linalg.cholesky(C)
        X=np.zeros((assets,simulation))
        for i in range(assets):
            X[i] = np.random.normal(size =simulation)
        LX = L@X
        U = stats.norm.cdf(LX)
    tau = -1/lbd * np.log(U)    
    if default == True:
        tau_p = np.where(tau<=y,1,0)
        p = np.count_nonzero(tau_p, axis=0)
        for i in p:
            if i>=n:
                 count+=1
        prob = round((count/simulation),4)
    else:
        tau_p = np.where(tau>y,1,0)
        p = np.count_nonzero(tau_p, axis =0)
        for i in p:
            if i==assets:
                count+=1
        prob = round((count/simulation),4)
    
    return prob

def default_prob_t(n,y,rho, default=True):
    count = 0
    if rho==0:
        X = np.random.standard_t(4,size = (assets,simulation))
        U = stats.t.cdf(X,4)
    else:
        C_ = np.identity(assets)
        C = np.where(C_==0, rho,1)
        L = np.linalg.cholesky(C)
        X=np.zeros((assets,simulation))
        for i in range(assets):
            X[i] = np.random.standard_t(4,size =simulation)
        LX = L@X
        U = stats.t.cdf(LX,4)
    tau = -1/lbd * np.log(U)    
    if default == True:
        tau_p = np.where(tau<=y,1,0)
        p = np.count_nonzero(tau_p, axis=0)
        for i in p:
            if i>=n:
                 count+=1
        prob = round((count/simulation),4)
    else:
        tau_p = np.where(tau>y,1,0)
        p = np.count_nonzero(tau_p, axis =0)
        for i in p:
            if i==assets:
                count+=1
        prob = round((count/simulation),4)
    
    return prob


if '__name__'=='__main__':
    np.random.seed(0)
    lbd = 0.1
    simulation = 25000
    assets =30
#    Gaussian
    Prob_9_1y_0 = default_prob_norm(9,1,0)
    Prob_9_1y_20 = default_prob_norm(9,1,0.2)
    Prob_9_1y_40 = default_prob_norm(9,1,0.4)
    Prob_11_1y_0 = default_prob_norm(11,1,0)
    Prob_11_1y_20 = default_prob_norm(11,1,0.2)
    Prob_11_1y_40 = default_prob_norm(11,1,0.4)
    Prob_14_1y_0 = default_prob_norm(14,1,0)
    Prob_14_1y_20 = default_prob_norm(14,1,0.2)
    Prob_14_1y_40 = default_prob_norm(14,1,0.4)
    
    Prob_0_15_0 = default_prob_norm(0,1.5,0,default=False)
    Prob_0_15_20 = default_prob_norm(0,1.5,0.2,default=False)
    Prob_0_15_40 = default_prob_norm(0,1.5,0.4,default=False)
    Prob_0_25_0 = default_prob_norm(0,2.5,0,default=False)
    Prob_0_25_20 = default_prob_norm(0,2.5,0.2,default=False)
    Prob_0_25_40 = default_prob_norm(0,2.5,0.4,default=False)
    Prob_0_35_0 = default_prob_norm(0,3.5,0,default=False)
    Prob_0_35_20 = default_prob_norm(0,3.5,0.2,default=False)
    Prob_0_35_40 = default_prob_norm(0,3.5,0.4,default=False)
#    Student t    
    tProb_9_1y_0 = default_prob_t(9,1,0)
    tProb_9_1y_20 = default_prob_t(9,1,0.2)
    tProb_9_1y_40 = default_prob_t(9,1,0.4)
    tProb_11_1y_0 = default_prob_t(11,1,0)
    tProb_11_1y_20 = default_prob_t(11,1,0.2)
    tProb_11_1y_40 = default_prob_t(11,1,0.4)
    tProb_14_1y_0 = default_prob_t(14,1,0)
    tProb_14_1y_20 = default_prob_t(14,1,0.2)
    tProb_14_1y_40 = default_prob_t(14,1,0.4)
    
    tProb_0_15_0 = default_prob_t(0,1.5,0,default=False)
    tProb_0_15_20 = default_prob_t(0,1.5,0.2,default=False)
    tProb_0_15_40 = default_prob_t(0,1.5,0.4,default=False)
    tProb_0_25_0 = default_prob_t(0,2.5,0,default=False)
    tProb_0_25_20 = default_prob_t(0,2.5,0.2,default=False)
    tProb_0_25_40 = default_prob_t(0,2.5,0.4,default=False)
    tProb_0_35_0 = default_prob_t(0,3.5,0,default=False)
    tProb_0_35_20 = default_prob_t(0,3.5,0.2,default=False)
    tProb_0_35_40 = default_prob_t(0,3.5,0.4,default=False)
    
    
    
        
        
        


        
        

    
    
