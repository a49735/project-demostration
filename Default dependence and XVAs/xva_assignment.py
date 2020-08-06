"""
xva SMU assignment
"""

import numpy as np
import pandas as pd
from scipy import stats



###############################################################################
# parameter values
###############################################################################

mu = 0.08
r = 0.04
sigma = 0.16
q = 0.02
K = 0.04
notional =100
notional_i = 20
sigma2 = sigma*sigma

num_years = 5

###############################################################################
# coupon values
###############################################################################

# statistical measure
EpR_i = mu - 0.5 * sigma2 - K
Rp = EpR_i * np.ones(num_years)

# risk-neutral measure
EqR_i = r - 0.5 * sigma2 - K
Rq = EqR_i * np.ones(num_years)


#Q1
###############################################################################
# swap values
###############################################################################

# discount curve
Z = np.exp(-r * np.arange(0, num_years + 1))

# value
discounted_cf = Z[1:] * Rq
# values discounted to today
V = np.cumsum(discounted_cf[::-1])[::-1]
V_0 = V[0]
print(V_0)
# value of swap at each point in time [V(0), V(1), ..., V(num_years-1)]
V_t = V / Z[:-1]

###############################################################################
# expected exposures
###############################################################################

# expected exposure at end of period

EE_e = np.round(((EpR_i + V_t[1:])) * notional,2)
#print(EE_e)
EE_s = np.round(V_t * notional,2)

###############################################################################
# expected postive exposures
###############################################################################
#X = log(St)/log(St-1) = mu - 0.5*sigma2 + sigma*epsilon
# Discount cruve under p measure
EE_plusi = (mu - 0.5 * sigma2 - K)*(stats.norm.cdf((mu - 0.5 * sigma2-K)/sigma)) +\
sigma*(stats.norm.pdf((mu - 0.5 * sigma2-K)/sigma))
EE_plus = EE_plusi * np.ones(num_years)
dis_EE_plus = EE_plus * Z[1:]
V_EE_plus = np.cumsum(dis_EE_plus[::-1])[::-1]
#EPE at time 0
V_EE_plus_0 = np.round(V_EE_plus[0] * notional, 2)
V_EE_plus_t = V_EE_plus / Z[1:]
#EPE at time 1 to 5
EE_plus_e = np.round(V_EE_plus_t * notional,2)


###############################################################################
# expected negative exposures
###############################################################################
EE_minus_i = (K-mu + 0.5 * sigma2)*(stats.norm.cdf((K-mu +0.5 * sigma2)/sigma)) +\
sigma*(stats.norm.pdf((mu - 0.5 * sigma2-K)/sigma))
EE_minus = EE_minus_i * np.ones(num_years)
dis_EE_minus = EE_minus * Z[1:]
V_EE_minus = np.cumsum(dis_EE_minus[::-1])[::-1]
#ENE at time 0
V_EE_minus_0 = np.round(V_EE_minus[0] * notional, 2)
V_EE_minus_t = V_EE_minus / Z[1:]
# ENE at time 1 to 5
EE_minus_e = np.round( V_EE_minus_t * notional,2)

# Potential future exposure
###############################################################################
#statistical measure
PFE_i = mu - 0.5 * sigma2 - K + sigma*1.645
PFE = PFE_i * np.ones(num_years)
dis_PFE = PFE*Z[1:]
V_PFE = np.cumsum(dis_PFE[::-1])[::-1]
#PFE at time 0
V_PFE_0 = np.round(V_PFE[0] * notional, 2)
print(V_PFE_0)
V_PFE_t = V_PFE/Z[1:]
#PFE at time 1 to 5
PFE_a = np.round( V_PFE_t * notional,2)

###############################################################################
# CVA
###############################################################################
#Expected postive exposure at year end under Q measure
EE_plus_q = ((r - 0.5 * sigma2 - K)*stats.norm.cdf((r - 0.5 * sigma2-K)/sigma) +\
sigma*stats.norm.pdf((r - 0.5 * sigma2-K)/sigma)) * np.ones(num_years)
dis_EE_plus_q = EE_plus_q * Z[1:]
V_EE_plus_q = np.cumsum(dis_EE_plus_q[::-1])[::-1]/Z[1:]
#Default probability
Q = [np.exp(-q*t)-np.exp(-q*(t+1)) for t in range(num_years)]
CVA = (1-0.4)*sum((Q*V_EE_plus_q*notional))
print(CVA)
###############################################################################
# DVA
###############################################################################
EE_minus_q = (K-r + 0.5 * sigma2)*stats.norm.cdf((K-r +0.5 * sigma2)/sigma) +\
sigma*stats.norm.pdf((r - 0.5 * sigma2-K)/sigma) * np.ones(num_years)
dis_EE_minus_q = EE_minus_q * Z[1:]
V_EE_minus_q = np.cumsum(dis_EE_minus_q[::-1])[::-1]/Z[1:]
DVA = (1-0.4)*sum((Q*V_EE_minus_q*notional))
print(DVA)
###############################################################################
#Q2
###############################################################################
#Five independent swaps will have the return process 5mu-5*0.5sigma+sigma*sum(epsilon1+epsilon2+..)
#Expected postive exposure
EE_plus5i = (5*mu - 5*0.5 * sigma2 - 5*K)*\
stats.norm.cdf((5*mu - 5*0.5 * sigma2-5*K)/np.sqrt(5)*sigma) +\
np.sqrt(5) * sigma * stats.norm.pdf((5*mu -5* 0.5 * sigma2-5*K)/np.sqrt(5)*sigma)
EE_plus5 = EE_plus5i * np.ones(num_years)
dis_EE_plus5 = EE_plus5 * Z[1:]
V_EE_plus5 = np.cumsum(dis_EE_plus5[::-1])[::-1]
# EPE at time 0
V_EE_plus5_0 = V_EE_plus5[0]*notional_i
print(V_EE_plus5_0)
V_EE_plus5_t = V_EE_plus5 / Z[1:]
# EPE at time 1 to 5
EE_plus5_e = np.round(V_EE_plus5_t * notional_i,2)

#Expected negative exposure
EE_minus5_i = (5*K-5*mu + 5*0.5 * sigma2)*\
stats.norm.cdf((5*K-5*mu +5*0.5 * sigma2)/np.sqrt(5)*sigma) +\
np.sqrt(5)*sigma*stats.norm.pdf((5*mu - 5*0.5 * sigma2-5*K)/np.sqrt(5)*sigma)
EE_minus5 = EE_minus5_i * np.ones(num_years)
dis_EE_minus5 = EE_minus5 * Z[1:]
V_EE_minus5 = np.cumsum(dis_EE_minus5[::-1])[::-1]
#ENE at time 0 
V_EE_minus5_0 = V_EE_minus5[0]*notional_i
print(V_EE_minus5_0)
V_EE_minus5_t = V_EE_minus5 / Z[1:]
# ENE at time 1 to 5
EE_minus5_e = np.round(V_EE_minus5_t * notional_i,2)

#Potential future exposure
PFE5_i = 5*mu - 5*0.5 * sigma2 - 5*K + np.sqrt(5)*sigma*1.645
PFE5 = PFE5_i * np.ones(num_years)
dis_PFE5 = PFE5*Z[1:]
V_PFE5 = np.cumsum(dis_PFE5[::-1])[::-1]
# PFE at time 0
V_PFE5_0 = V_PFE5[0]*notional_i
print(V_PFE5_0)
V_PFE5_t = V_PFE5/Z[1:]
# PFE at time 1 to 5
PFE5_a = np.round(V_PFE5_t * notional_i,2)

#CVA
EE_plus5_q = (5*r - 5*0.5 * sigma2 - 5*K)*\
stats.norm.cdf((5*r - 5*0.5 * sigma2-5*K)/np.sqrt(5)*sigma) +\
np.sqrt(5) * sigma * stats.norm.pdf((5*r -5* 0.5 * sigma2-5*K)/np.sqrt(5)*sigma) * np.ones(num_years)
dis_EE_plus5_q = EE_plus5_q * Z[1:]
V_EE_plus5_q = np.cumsum(dis_EE_plus5_q[::-1])[::-1]/Z[1:]
CVA5 = (1-0.4)*sum((Q*V_EE_plus5_q*notional_i))

#DVA
EE_minus5_q = (5*K-5*r + 5*0.5 * sigma2)*\
stats.norm.cdf((5*K-5*r +5*0.5 * sigma2)/np.sqrt(5)*sigma) +\
np.sqrt(5)*sigma*stats.norm.pdf((5*r - 5*0.5 * sigma2-5*K)/np.sqrt(5)*sigma) * np.ones(num_years)
dis_EE_minus5_q = EE_minus5_q * Z[1:]
V_EE_minus5_q = np.cumsum(dis_EE_minus5_q[::-1])[::-1]/Z[1:]
DVA5 = (1-0.4)*sum((Q*V_EE_minus5_q*notional_i))
print(CVA5)
print(DVA5)


#Q3
###############################################################################
#Monte Carlo simulation
EEQ_list, EEP_list, EPE_Q_list, EPE_P_list, PFE_Q_list, PFE_P_list, ENE_Q_list,\
ENE_P_list = [], [],[],[],[],[], [], []
steps = 2**17
assets = 5
meanQ = r-0.5*sigma2-K
meanP = mu-0.5*sigma2-K
np.random.seed(0)
for n in range(num_years):
    epsilonQ = np.random.normal(meanQ,sigma,size=(steps,assets))
    EEQ = np.mean(np.sum(epsilonQ, axis=1))
    epsilonP = np.random.normal(meanP,sigma,size=(steps,assets))
    EEP = np.mean(np.sum(epsilonP, axis=1))
    pos_P = [x for x in np.sum(epsilonP, axis=1) if x>0]
    neg_P = [x for x in np.sum(epsilonP, axis=1) if x<=0]
    pos_Q = [x for x in np.sum(epsilonQ, axis=1) if x>0]
    neg_Q = [x for x in np.sum(epsilonQ, axis=1) if x<=0]
    EEP_pos = np.mean(pos_P)
    EEP_neg = np.mean(neg_P)
    EEQ_pos = np.mean(pos_Q)
    EEQ_neg = np.mean(neg_Q)
    PFE_P = stats.norm.ppf(0.95,np.mean(np.sum(epsilonP, axis=1)),np.std(np.sum(epsilonP, axis=1)))
    EEQ_list.append(EEQ)
    EEP_list.append(EEP)
    EPE_P_list.append(EEP_pos)
    ENE_P_list.append(EEP_neg)
    EPE_Q_list.append(EEQ_pos)
    ENE_Q_list.append(EEQ_neg)
    PFE_P_list.append(PFE_P)


dis_EE_mc = Z[1:]*EEQ_list
V_EE_mc = np.cumsum(dis_EE_mc[::-1])[::-1]
#Swap value
V_mc = V_EE_mc[0]*notional_i
V_EE_mc_t_e = V_EE_mc/Z[:-1]
#Expected exposure
EE_mc_e = np.round((np.array(EEP_list) + np.append(np.array(V_EE_mc_t_e[1:]),0))*notional_i,2)

#Expected postive exposure
dis_EPE_mc = Z[1:]*EPE_P_list
V_EPE_mc = np.cumsum(dis_EPE_mc[::-1])[::-1]
#EPE at time 0
V_EPE_mc_0 = V_EPE_mc[0]*notional_i
print(V_EPE_mc_0)
V_EPE_mc_t = V_EPE_mc/Z[1:]
#EPE at time 1 to 5
EPE_mc_e = np.round( V_EPE_mc_t * notional_i,2)

#Expected negative exposure
dis_ENE_mc = Z[1:]*ENE_P_list
V_ENE_mc = np.cumsum(dis_ENE_mc[::-1])[::-1]
#ENE at time 0
V_ENE_mc_0 = V_ENE_mc[0]*notional_i
print(V_ENE_mc_0)
#EPE at time 1 to 5
V_ENE_mc_t = V_ENE_mc/Z[1:]
ENE_mc_e = (np.round(V_ENE_mc_t * notional_i,2))
#Potential future exposure
dis_PFE_mc = Z[1:]*PFE_P_list
V_PFE_mc = np.cumsum(dis_PFE_mc[::-1])[::-1]
#PFE at time 0
V_PFE_MC_0 = V_PFE_mc[0]*notional_i
print(V_PFE_MC_0)
V_PFE_mc_t = V_PFE_mc/Z[1:]
#PFE at time 1 to 5
PFE_mc_e = np.round(V_PFE_mc_t * notional_i,2)
#CVA and DVA
dis_EPE_mc_q = Z[1:]*EPE_Q_list
V_EPE_mc_q = np.cumsum(dis_EPE_mc_q[::-1])[::-1]/Z[1:]
CVA_mc = (1-0.4)*sum(Q*V_EPE_mc_q*notional_i)

dis_ENE_mc_q = Z[1:]*ENE_Q_list
V_ENE_mc_q = np.cumsum(dis_ENE_mc_q[::-1])[::-1]/Z[1:]
DVA_mc = (1-0.4)*sum(Q*V_ENE_mc_q*notional_i)
print(CVA_mc)
print(DVA_mc)  
###############################################################################    
#Q4
#Monte Carlo simulation with monthly grid
 
m_EEQ_list, m_EEP_list, m_EPE_Q_list, m_EPE_P_list, m_PFE_Q_list, m_PFE_P_list, m_ENE_Q_list,\
m_ENE_P_list = [], [],[],[],[],[], [], []

meanQ_m = meanQ/12
meanP_m = meanP/12
for n in range(num_years*12):
    epsilonQ = np.random.normal(meanQ_m,sigma/np.sqrt(12),size=(steps,assets))
    EEQ = np.mean(np.sum(epsilonQ, axis=1))
    epsilonP = np.random.normal(meanP_m,sigma/np.sqrt(12),size=(steps,assets))
    EEP = np.mean(np.sum(epsilonP, axis=1))
    pos_P = [x for x in np.sum(epsilonP, axis=1) if x>0]
    neg_P = [x for x in np.sum(epsilonP, axis=1) if x<=0]
    pos_Q = [x for x in np.sum(epsilonQ, axis=1) if x>0]
    neg_Q = [x for x in np.sum(epsilonQ, axis=1) if x<=0]
    EEP_pos = np.mean(pos_P)
    EEP_neg = np.mean(neg_P)
    EEQ_pos = np.mean(pos_Q)
    EEQ_neg = np.mean(neg_Q)
    PFE_P = stats.norm.ppf(0.95,np.mean(np.sum(epsilonP, axis=1)),np.std(np.sum(epsilonP, axis=1)))
    m_EEQ_list.append(EEQ)
    m_EEP_list.append(EEP)   
    m_EPE_P_list.append(EEP_pos)  
    m_ENE_P_list.append(EEP_neg) 
    m_EPE_Q_list.append(EEQ_pos)
    m_ENE_Q_list.append(EEQ_neg)
    m_PFE_P_list.append(PFE_P)


Zm = np.exp(-r/12 * np.arange(0, num_years*12 + 1))

m_dis_EE_mc = Zm[1:]*m_EEQ_list
m_V_EE_mc = np.cumsum(m_dis_EE_mc[::-1])[::-1]
#Swap value
m_V_mc = m_V_EE_mc[0]*notional_i
m_V_EE_mc_t = m_V_EE_mc/Zm[:-1]
#Expected exposure
m_EE_mc_e = np.round((np.array(m_EEP_list) + np.append(np.array(m_V_EE_mc_t[1:]),0))*notional_i,2)

#Expected postive exposure
m_dis_EPE_mc = Zm[1:]*m_EPE_P_list
m_V_EPE_mc = np.cumsum(m_dis_EPE_mc[::-1])[::-1]
#EPE at time 0
m_V_EPE_mc_0 = m_V_EPE_mc[0]*notional_i
m_V_EPE_mc_t = m_V_EPE_mc/Zm[1:]
#EPE at time 1 to 5
m_EPE_mc_e = np.round(m_V_EPE_mc_t * notional_i,2)

#Expected negative exposure
m_dis_ENE_mc = Zm[1:]*m_ENE_P_list
m_V_ENE_mc = np.cumsum(m_dis_ENE_mc[::-1])[::-1]
#ENE at time 0
m_V_ENE_mc_0 = m_V_ENE_mc[0]*notional_i
m_V_ENE_mc_t = m_V_ENE_mc/Zm[1:]
#ENE at time 1 to 5
m_ENE_mc_e = abs(np.round(m_V_ENE_mc_t * notional_i,2))

#Potential future exposure
m_dis_PFE_mc = Zm[1:]*m_PFE_P_list
m_V_PFE_mc = np.cumsum(m_dis_PFE_mc[::-1])[::-1]
#PFE at time 0
m_V_PFE_mc_0 = m_V_PFE_mc[0]*notional_i
m_V_PFE_mc_t = m_V_PFE_mc/Zm[1:]
#PFE at time 1 to 5
m_PFE_mc_e = np.round(m_V_PFE_mc_t*notional_i,2)

#CVA and DVA
Qm = [np.exp(-q/12*t)-np.exp(-q/12*(t+1)) for t in range(num_years*12)]
m_dis_EPE_mc_q = Zm[1:]*m_EPE_Q_list
m_EPE_mc_q = np.cumsum(m_dis_EPE_mc_q[::-1])[::-1]/Zm[1:]

m_dis_ENE_mc_q = Zm[1:]*m_ENE_Q_list
m_ENE_mc_q = np.cumsum(m_dis_ENE_mc_q[::-1])[::-1]/Zm[1:]

m_CVA_mc = (1-0.4)*sum(Qm*m_EPE_mc_q*notional_i)
m_DVA_mc = (1-0.4)*sum(Qm*m_ENE_mc_q*notional_i)  
print(m_CVA_mc)
print(m_DVA_mc)

df = pd.DataFrame.from_dict({'EE':np.append(m_V_mc,m_EE_mc_e),'EPE':np.append(m_V_EPE_mc_0,m_EPE_mc_e)
, 'ENE':np.append(m_V_ENE_mc_0,m_ENE_mc_e), 'PFE':np.append(m_V_PFE_mc_0,m_PFE_mc_e) }) 
df.to_excel('Exposure.xlsx')

