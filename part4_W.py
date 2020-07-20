#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 00:15:23 2020

@author: wuruida
"""
from project0_part1 import Swpdf,OISDarray,LiborDF,swp_rate,SwprateDict,OISR,IRSR
from part3_zc import  SABR, IRR,CMS_Swp_df,Alpha, Nu, Rho,Black76payer, Black76receiver
from scipy.misc import derivative
from scipy.integrate import quad



def g(k):
    CMS10y = k
    return CMS10y**0.25-0.04**0.5
def gp(k):
    return 0.25*k**(-0.75)
def gpp(k):
    return -0.1875*k**(-1.75)

def h(k):
    return g(k)/IRR(k,tenor,freq)

def hp(k):
    IRRp=derivative(lambda x:IRR(x,tenor,freq),k,dx=0.01,n=1)
    return (IRR(k,tenor,freq)*gp(k)-g(k)*IRRp)/(IRR(k,tenor,freq)**2)

def hpp(k):
    IRRp=derivative(lambda x:IRR(x,tenor,freq),k,dx=0.01,n=1)
    IRRpp=derivative(lambda x:IRR(x,tenor,freq),k,dx=0.01,n=2)
    #print("k:{}".format(k))
    #print("IRR(k,tenor,freq):{}".format(IRR(k,tenor,freq)))
    #print("IRRp:{}".format(IRRp))
    #print("IRRpp:{}".format(IRRpp))
    left = (IRR(k,tenor,freq)*gpp(k)-IRRpp*g(k)-2*IRRp*gp(k))/(IRR(k,tenor,freq)**2)
    right = (2*IRRp**2*g(k))/(IRR(k,tenor,freq)**3)
    return left+right

def CMSreplic(t,tenor,freq):
    alpha = Alpha[9]
    beta = 0.9
    rho = Rho[9]
    nu = Nu[9]
    F=CMS_Swp_df['Fwd Swap rates']['5y*10y']
    Vpay =lambda x: Black76payer(F,x,SABR(F,x,t,alpha,beta,rho,nu),t)*IRR(F,tenor,freq)
    Vrec = lambda x: Black76receiver(F,x,SABR(F,x,t,alpha,beta,rho,nu),t)*IRR(F,tenor,freq)
    print("Vpay:{}".format(Vpay(F)))
    IntRec = quad(lambda k: hpp(k)*Vrec(k),0,F )[0]
    IntPay = quad(lambda k: hpp(k)*Vpay(k),F,F+0.02+0.01 )[0]
    print("IntRec:{}".format(IntRec))
    print("IntPay:{}".format(IntPay))    
    return g(F)+hp(F)*(Vpay(F)-Vrec(F))+IntRec+IntPay
        



#1
#The corresponding index for discount factors
dfidx=[0.5+i*0.5 for i in range(60)]
df=OISDarray[dfidx.index(5)]
t = 5
tenor = 10
freq = 2
value1 =df*CMSreplic(t,tenor,freq)
print("value1:{}".format(value1))

#2
def CMSreplic2(t,tenor,freq):
    alpha = Alpha[9]
    beta = 0.9
    rho = Rho[9]
    nu = Nu[9]
    B=(0.04**0.5)**4
    F=CMS_Swp_df['Fwd Swap rates']['5y*10y']
    Vpay =lambda x: Black76payer(F,x,SABR(F,x,t,alpha,beta,rho,nu),t)*IRR(F,tenor,freq)
    #Vrec = lambda x: Black76receiver(F,x,SABR(F,x,t,alpha,beta,rho,nu),t)*IRR(x,tenor,freq)
    print("Vpay:{}".format(Vpay(B)))
    #IntRec = quad(lambda k: hpp(k)*Vrec(k),0,F )
    IntPay = quad(lambda k: hpp(k)*Vpay(k),B,F+0.02+0.01 )[0]
    print("IntPay:{}".format(IntPay))
    print("hp:{}".format(hp(B)))
    return hp(B)*Vpay(B)+IntPay
    



value2 = df*CMSreplic2(t,tenor,freq)
print("value2:{}".format(value2))


















