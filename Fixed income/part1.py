#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 21:14:39 2020

@author: zhangcheng
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from collections import deque
import copy

xls=pd.ExcelFile('IR Data.xlsx')
IRS=pd.read_excel(xls,'IRS',header=0)
OIS=pd.read_excel(xls,'OIS',header=0)
swaption=pd.read_excel(xls,'Swaption',header=2)
IRSR=IRS['Rate'].values
Tenor=[0.5,1,2,3,4,5,7,10,15,20,30]
OISR=OIS['Rate'].values

#Q1

#A helper function to interpolate inbetween discount factors
def interp(a,b,n):
    p=(b-a)/n
    points=[a+i*p for i in range(1,n)]
    return points


OISdisc=[]

OISdisc.append(1/(1+OISR[1]))
for i in range(2,len(Tenor)):
    if len(OISdisc)+1<Tenor[i]:
        n=Tenor[i]-len(OISdisc)-1+1
#        the fsolve rootfinding the OIS discount factor that equates the fixed leg with floating leg
        x=fsolve(lambda d:(sum(OISdisc)+d+np.sum(interp(OISdisc[-1],d,n)))*OISR[i]-(1-d),x0=0.9)[0]
#        Append the interpolated discountfactors first
        OISdisc.extend(interp(OISdisc[-1],x,n))
        OISdisc.append(x)
        
    else:
        OISdisc.append(fsolve(lambda d:(sum(OISdisc)+d)*OISR[i]-(1-d),x0=0.9)[0])
        
OISdisc1=copy.deepcopy(OISdisc)
OISD1=deque(OISdisc1)
#The discount factors from 6m libor rate is appended left in the last step
OISD1.appendleft(1/(1+0.5*OISR[0]))


t=np.linspace(0.5,31,31)
plt.plot(t,OISD1,label='OIS Discount Curve')
plt.legend()
plt.grid(True)
plt.xlabel('Tenor/Year')
plt.title('OIS Discount Curve')

#Q2
OISDh=[]
# Since the IRS is semiannual paied, the half-year OIS discount factor should be intepolated
for i in range(1,len(OISdisc)):
    d=(OISdisc[i]+OISdisc[i-1])/2
    OISDh.append(d)

#Insert the intepolated OIS discount factors into the OIS DF list in order
epsilon=0
for j,d in enumerate(OISDh,1):
    OISdisc.insert(j+epsilon,d)
    epsilon+=1

OISD2=deque(OISdisc)
OISD2.appendleft(1/(1+0.5*OISR[0]))
#The resulting OIS discount factors in array
OISDarray=np.array(OISD2)


#A helper function that calculates forward Libor rate from Libor discount factors.
def DF_to_rate(df,x,n,gap,delta=0.5):
    if n<=2:
        df1=np.append(df,x)
        rate=[(df1[i-1]-df1[i])/(df1[i]*delta) for i in range(1,n)]
    else:
        midrate=interp(df[-1],x,gap+1)
        df1=np.append(df,midrate)
        df1=np.append(df1,x)
        rate=[(df1[i-1]-df1[i])/(df1[i]*delta) for i in range(1,n)]
    rate0=[IRSR[0]]
    rate0.extend(rate)
    return rate0



LiborDF=[]
LiborDF.append(1/(1+0.5*IRSR[0]))



for i in range(1,len(Tenor)):
    j=Tenor[i]
    gap=j*2-len(LiborDF)-1
#    fsovle rootfinding the libor discount factor that equates fixed payments with floating payments
#    payments are discounted with OIS discount factors
    df=fsolve(lambda d:sum(OISDarray[:j*2])*IRSR[i]-\
              sum(OISDarray[:j*2]*(DF_to_rate(LiborDF,d,j*2,gap))),x0=0.9)[0]
    if i>=2:
        LiborDF.extend(interp(LiborDF[-1],df,gap+1))
        LiborDF.append(df)
    else:
        LiborDF.append(df)
#The resulting Libor discount factor in list 
    
t2=np.linspace(0.5,31,60)
plt.plot(t2,LiborDF)
plt.legend()
plt.grid(True)
plt.title('Libor Discount Curve')
plt.xlabel('Tenor/Year')

#Q3
#There are 60 nodes for semiannual payments IRS up to 30 years
Tenor2=[0.5+i*0.5 for i in range(60)]
start=[1,5,10]
end=[1,2,3,5,10]
#Swap rates calculation
def swp_rate(expire,tenor,LiborDF,OISDarray,date):
    assert isinstance(expire,list)
    assert isinstance(tenor,list)
    Swaprates=[]
    for i in expire:
        for j in tenor:
#        Specify individual node of payment for each forward swap
            node=[i+n*0.5 for n in range(0,j*2+1)]
#        For every node there is a index which corresponds to a particular discount factor
            m=[date.index(t) for t in node]
            fwdrate=np.array([(LiborDF[n]-LiborDF[n+1])/(0.5*LiborDF[n+1]) for n in m[:-1]])
            fwddf=np.array([OISDarray[n] for n in m[1:]])
#        The forward swap rate that equates fixed payments with floating payments
            swprate=fwddf@fwdrate/np.sum(fwddf)
            Swaprates.append((str(i)+'y'+'*'+str(j)+'y',swprate))
    return Swaprates

Swaprates=swp_rate(start,end,LiborDF,OISDarray,Tenor2)
#Dictonary comprehension, then export into DataFrame
SwprateDict={y:rate for y,rate in Swaprates }
Swpdf=pd.DataFrame.from_dict(SwprateDict, orient='index',columns=['Fwd Swap rates'])

        
        

   
    
    
