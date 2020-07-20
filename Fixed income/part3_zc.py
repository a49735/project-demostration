#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 18:47:45 2020

@author: zhangcheng
"""

from scipy.integrate import quad
import numpy as np
import pandas as pd
from project0_part1 import Swpdf,OISDarray,LiborDF,swp_rate,SwprateDict,OISR,IRSR
from scipy.stats import norm
from scipy.misc import derivative

#Import SABR parameters exported by part 2
xls=pd.ExcelFile('model_data.xlsx')
Alphadf=pd.read_excel(xls,'Alpha',header=0,index_col=0)
Nudf=pd.read_excel(xls,'Nu',header=0,index_col=0)
Rhodf=pd.read_excel(xls,'Rho',header=0,index_col=0)

Alpha=Alphadf.values.flatten()
Nu=Nudf.values.flatten()
Rho=Rhodf.values.flatten()




def Black76payer(F,K,sigma,T,df=1):
    d1 = (np.log(F/K)+(sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return df*(F*norm.cdf(d1) - K*norm.cdf(d2))

def Black76receiver(F,K,sigma,T,df=1):
    d1 = (np.log(F/K)+(sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return df*((K*norm.cdf(-d2) - F*norm.cdf(-d1)))

def SABR(F, K, T, alpha, beta, rho, nu):
    X = K
    # if K is at-the-money-forward
    if abs(F - K) < 1e-12:
        numer1 = (((1 - beta)**2)/24)*alpha*alpha/(F**(2 - 2*beta))
        numer2 = 0.25*rho*beta*nu*alpha/(F**(1 - beta))
        numer3 = ((2 - 3*rho*rho)/24)*nu*nu
        VolAtm = alpha*(1 + (numer1 + numer2 + numer3)*T)/(F**(1-beta))
        sabrsigma = VolAtm
    else:
        z = (nu/alpha)*((F*X)**(0.5*(1-beta)))*np.log(F/X)
        zhi = np.log((((1 - 2*rho*z + z*z)**0.5) + z - rho)/(1 - rho))
        numer1 = (((1 - beta)**2)/24)*((alpha*alpha)/((F*X)**(1 - beta)))
        numer2 = 0.25*rho*beta*nu*alpha/((F*X)**((1 - beta)/2))
        numer3 = ((2 - 3*rho*rho)/24)*nu*nu
        numer = alpha*(1 + (numer1 + numer2 + numer3)*T)*z
        denom1 = ((1 - beta)**2/24)*(np.log(F/X))**2
        denom2 = (((1 - beta)**4)/1920)*((np.log(F/X))**4)
        denom = ((F*X)**((1 - beta)/2))*(1 + denom1 + denom2)*zhi
        sabrsigma = numer/denom

    return sabrsigma

# A helper function that inserts intepolated quarterly discount factors
def insertion(new,old):
    epsilon=0
    for j,d in enumerate(old,1):
        new.insert(j+epsilon,d)
        epsilon+=1
    return new



#A helper function to interpolate parameters for the specified expire/tenor
def param_interp(df,tenor,date,expire):
    assert isinstance(df,pd.DataFrame)
    if date<expire[0]:
            new_p1=df[str(tenor)+'Y'][str(expire[0])+'Y']-\
            (expire[0]-date)/expire[0]*df[str(tenor)+'Y'][str(expire[0])+'Y']
            return new_p1
    else:
        for i in range(len(expire)):
            if expire[i]<date<expire[i+1]:
                new_p2=df[str(tenor)+'Y'][str(expire[i])+'Y']+(date-expire[i])/(expire[i+1]-expire[i])*\
            (df[str(tenor)+'Y'][str(expire[i+1])+'Y']- df[str(tenor)+'Y'][str(expire[i])+'Y'])
                return new_p2

# A helper function to present parameters in the required format
def Tabulate(data):
    assert isinstance(data,list)
    columns=[str(i)+'Y' for i in tenor]   
    datadict={str(expire[i])+'Y':data[i*5:5+i*5] for i in range(len(expire))}
    datadf=pd.DataFrame.from_dict(datadict,orient='index',columns=columns)
    return datadf

def IRR(S,n,m):
    irr=[(1/m)/(1+S/m)**i for i in range(1,n*m+1)]
    return sum(irr)

def h_2prime(k,n,m):
    p2=derivative(lambda x:IRR(x,n,m),k,dx=0.01,n=2)
    p1=derivative(lambda x:IRR(x,n,m),k,dx=0.01,n=1)
    return (-p2*k-2*p1)/IRR(k,n,m)**2+2*p1**2*k/IRR(k,n,m)**3

def swp_replica(F,T,alpha,beta,rho,nu,n,m):
    rec=quad(lambda x:Black76receiver(F,x,SABR(F,x,T,alpha,beta,rho,nu),T)\
             *h_2prime(x,n,m)*IRR(F,n,m),0,F)[0]
    pay=quad(lambda x:Black76payer(F,x,SABR(F,x,T,alpha,beta,rho,nu),T,)\
             *h_2prime(x,n,m)*IRR(F,n,m),F,F+0.02+0.01)[0]
    return (F+rec+pay)


#Calcualte CMS rates

expire=[1,5,10]
tenor=[1,2,3,5,10]
#The corresponding index for discount factors
dfidx=[0.5+i*0.5 for i in range(60)]
m=2
CMS=[]
N=tenor*len(expire)
for i in range(len(expire)*len(tenor)):
    F=Swpdf.iloc[i,0]
    if i<5:
        T=expire[0]
    elif i>=5 and i<10:
        T=expire[1]
    else:
        T=expire[2]
    CMS.append(swp_replica(F,T,Alpha[i],0.9,Rho[i],Nu[i],N[i],2))

#Incorporate data into DataFrame and compare with Forward swap rates    
CMSindx=[str(i)+'y'+'*'+str(j)+'y' for i in expire for j in tenor]
CMSdict={idx:cms for idx,cms in zip(CMSindx,CMS)}
CMSdf=pd.DataFrame.from_dict(CMSdict, orient='index',columns=['CMS rates'])
CMS_Swp_df=pd.concat([Swpdf,CMSdf],axis=1)
CMS_Swp_df.round(6).to_csv('CMS_swp.csv') 
#Convexity Correction in basis points
CMS_Swp_df['Convexity Correction']=(CMS_Swp_df['CMS rates']-CMS_Swp_df['Fwd Swap rates'])*10000
CCorrection=CMS_Swp_df['Convexity Correction'].values
CC_df=Tabulate(list(CCorrection))
CC_df.round(6).to_csv('CC.csv') 
#Valuing CMS leg
#CMS10y semi-annual payments for 5 years;CMS2y quarterly payments over 10 years

def CMS_leg(tenor,year,DFLibor,DFOIS,dfidx,freq='semi-annually'):
    CMS=[]
    DiscountF=[]
    if freq=='semi-annually':
        T=[0.5+i*0.5 for i in range(year*2)]
        for t in T:
#            locate the correponding discount factors
            DiscountF.append(OISDarray[dfidx.index(t)])
#        use param_interp function imported from part2 to intepolate the corresponding
#        parameters for SABR model.
            if t not in expire:
                alpha=param_interp(Alphadf,tenor,t,expire)
                rho=param_interp(Rhodf,tenor,t,expire)
                nu=param_interp(Nudf,tenor,t,expire)
#        use swp_rate function imported from part1 to calculate the spot swap rates with
#        the corresponding expire/tenor
                F=swp_rate([t],[tenor],DFLibor,DFOIS,dfidx)[0][1]
                CMS.append(swp_replica(F,t,alpha,0.9,rho,nu,tenor,m))
            else:
                alpha=Alphadf[str(tenor)+'Y'][str(int(t))+'Y']
                rho=Rhodf[str(tenor)+'Y'][str(int(t))+'Y']
                nu=Nudf[str(tenor)+'Y'][str(int(t))+'Y']
                F=SwprateDict[str(int(t))+'y'+'*'+str(tenor)+'y']
                CMS.append(swp_replica(F,t,alpha,0.9,rho,nu,tenor,m))    
        return 0.5*np.array(DiscountF)@np.array(CMS)
        
    if freq=='quarterly':
        T=[0.25+i*0.25 for i in range(year*4)]
#    The OIS discount factor is stored semi-annually by default, if the leg is not
#    semi-annual, intepolations should be made.
        new_df=[1/(1+0.25*OISR[0])]+[(DFOIS[i]+DFOIS[i+1])*0.5 for i in range(59)]
        new_libordf=[1/(1+0.25*IRSR[0])]+[(DFLibor[i]+DFLibor[i+1])*0.5 for i in range(59)]
        new_OIS=insertion(new_df,DFOIS)
        new_Libor=insertion(new_libordf,DFLibor)
        new_dfidx=[0.25+i*0.25 for i in range(len(new_df))]
        for t in T:
            DiscountF.append(new_OIS[new_dfidx.index(t)])
            if t not in expire:
                alpha=param_interp(Alphadf,tenor,t,expire)
                rho=param_interp(Rhodf,tenor,t,expire)
                nu=param_interp(Nudf,tenor,t,expire)
#                swp_rate returns a list of tuples
                F=swp_rate([t],[tenor],new_Libor,new_OIS,new_dfidx)[0][1]
                CMS.append(swp_replica(F,t,alpha,0.9,rho,nu,tenor,m))
                if t==0.25:
                    print(alpha,rho,nu,F)
            else:
                alpha=Alphadf[str(tenor)+'Y'][str(int(t))+'Y']
                rho=Rhodf[str(tenor)+'Y'][str(int(t))+'Y']
                nu=Nudf[str(tenor)+'Y'][str(int(t))+'Y']
                F=SwprateDict[str(int(t))+'y'+'*'+str(tenor)+'y']
                CMS.append(swp_replica(F,t,alpha,0.9,rho,nu,tenor,m))
        return 0.25*np.array(DiscountF)@np.array(CMS)
            
CMS10y_5=CMS_leg(10,5,LiborDF,OISDarray,dfidx,freq='semi-annually') 
CMS2y_10=CMS_leg(2,10,LiborDF,OISDarray,dfidx,freq='quarterly') 
  
    
