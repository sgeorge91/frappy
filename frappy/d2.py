# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 18:17:42 2022

@author: SVG
"""

import numpy as np
import scipy.stats as sp
import random
from scipy.optimize import curve_fit
def d2model(M,Md,Dsat):
    f=[]
    for i in M:
        if(i<Md):
            f.append((((Dsat-1)/(Md-1))*(i-1))+1)
        else:
            f.append(Dsat)
    
    return(np.array(f))
def calc_chi2(ym, sig, yf):
    diff = pow(ym-yf, 2.)
    chi2 = (diff / pow(sig,2.)).sum()
    df=len(ym)-2
    rchi2=chi2/float(df)
    return rchi2
def uniform_deviate(x):
    x=np.array(x)
    rx=sp.rankdata(x)
    rx=rx/len(rx)
    return(rx)


def embedding(x,d=2,tau=0):
    """
    Returns embedded vectors in dimension d from a time series.
    Parameter
    ---------
    x : 1-d numpy array
        Input time series for which embedding is done
    d : int
        Dimension of the embedding space. Default, 2
    tau : int
	Time delay for the embedding.
    Returns
    -------
    dvec : n-d numpy array
	Array of embedded vectors
    """     
    x=np.array(x)
    if(tau==0):
        mu=np.mean(x)
        sig2=np.var(x)
        xn=x-mu
        acf=np.correlate(xn,xn,'full')[len(xn)-1:]
        acf=acf/sig2/len(xn)
        tau=np.where(acf<(1./np.exp(1)))[0][0]
    n=int(len(x)-d*tau)
    dvec=np.zeros((n,d))
    for i in range(0,len(dvec)):
        for j in range(0,d):
            dvec[i][j]=x[i+j*tau]
    
    return(dvec)


def corrdim(dvec, Rmin=-1,Nc=-1):
    """
    Returns d2 for a series of embedded vectors.
    Parameter
    ---------
    dvec : n-d numpy array
        Input vectors for which D2 is caluculated
    Rmin : float
        Minimum R for which the scaling is calculated (See Harikrishnan et. al (2006) Physica D)
    Nc : int
	Number of centers around which the scaling is calculated
    Returns
    -------
    slope: float 
	Slope of the scaling region, i.e. the correlation dimension
    std_err: float
	Error for the straight line fit 
    Rmin : float
	Rmin for next value of embedding. (See find_d2)
    """     
    R_mat=[]
    d=len(dvec[0]) #Dimension of the data
    N=len(dvec) #Length of the data
    if(Nc==-1):
	    Nc=int(max(len(dvec)/10,1000)) #Number of centers
    ci=random.sample(range(len(dvec)),Nc)
    Rmina=(1./N)#**(1./d)
    Rmina=Rmina/4.
    if(d==1)or(Rmin==-1):
        Rmin=float(Rmina)
    Rmax=0.5
    irmaxa=25
    val = Rmax/Rmin
    val = np.log(val)/float(irmaxa)
    rfact = np.exp(val)
    R_mat.append(Rmin)
    
    temp=Rmin
    ivec=np.ones(d)
    for i in range (1,irmaxa):
        R_mat.append(temp*rfact)
        temp=temp*rfact
    cor_mat=np.zeros(irmaxa)
    Tcen_mat=np.zeros(irmaxa)
    for i in range (0,len(ci)):
        cent=dvec[ci[i]]
        mcom=min(cent)
        mpcom=min(ivec-cent)
        Rcenmax=min(.5,mpcom,mcom)
        
        for ir in range(0,irmaxa):
            if(R_mat[ir]>Rcenmax):
                continue
            disvec=abs(dvec-cent)
            maxnorm=np.amax(disvec,axis=1)
            maxnorm_1=np.where(maxnorm<=R_mat[ir])[0]
            csum=len(maxnorm_1)#k1 < R_mat[ir] for k1 in maxnorm)#
            cor_mat[ir]=cor_mat[ir]+csum-1
            Tcen_mat[ir]+=1
    cor_min = 10.0
    amin_cent = int(N/100.0)
    irmin=0
    irmax=0
    for ir in range(0,irmaxa-1):
        if(Tcen_mat[ir] > 0.0):
            cor_mat[ir] = cor_mat[ir]/Tcen_mat[ir]
            #write(47,720) R_mat(ir),cor_mat(ir), Tcen_mat(ir),float(m)
            if(cor_mat[ir] < cor_min):
                irmin=ir
            if((R_mat[ir+1] - R_mat[ir]) < Rmina):
                irmin=ir
            if(Tcen_mat[ir] > amin_cent):
                irmax=ir#-1
    Rmin=R_mat[irmin]
    #irmin=irmin+1
    
    #print(d, Rmina, R_mat[irmin],R_mat[irmax],rfact)
    if(irmin+3>irmax):
        return([0,0,Rmin])
    ja = 0
    y1=np.zeros(irmax-irmin)
    x1=np.zeros(irmax-irmin)
    sig=np.zeros(irmax-irmin)
    for ir in range(irmin,irmax):
      
       y1[ja] = np.log(cor_mat[ir])
       x1[ja] = np.log(R_mat[ir])
       sig[ja] = y1[ja]/np.sqrt(y1[ja])
       ja = ja+1
       #jamax = ja
    slope, intercept, r_value, p_value, std_err = sp.linregress(x1,y1)
    print(d,slope,std_err)
    return(slope,std_err*np.sqrt(len(x1)),Rmin)

def find_d2(ts,max_ed=10):
    """
    Returns d2 vs M of given time series.
    Parameter
    ---------
    ts : list or 1-d array
        Input time series for which D2 is caluculated
    max_ed : int
        Maximum dimension till which the data is embedded.
    Returns
    -------
    d2_ts : 1 d array 
	D2 for embedding dimensions 1 to max_ed
    err_ts: 1-d array 
	Error for each value of D2
    """ 
    uts=uniform_deviate(ts)
    d2_ts=np.zeros(max_ed)
    err_ts=np.zeros(max_ed)
    Rmin=-1
    for m in range (1,max_ed+1):
        uts_vecs=embedding(uts,m)
        slope,err,Rmin=corrdim(uts_vecs,Rmin)
        d2_ts[m-1]=slope
        err_ts[m-1]=err
    return(d2_ts,err_ts)
def find_d2s(M,D2, errs=None):
    """
    Returns D2sat for given values of D2 and M (embedding dim).
    Parameter
    ---------
    M : list or 1-d array
        Embedding dimensions for values of D2
    D2 : list or 1-d array
        D2 for each value of M.
    errs: list or 1-d array
	Error for each value of D2
    Returns
    -------
    op2[0][1]: float
	Saturated D2 value after function fitting
    int(op2[0][0])+1: int
	Embedding dimension at which D2 saturates
    chi2: float
	Chi2 value for fit.
    """
    D2=np.array(D2)
    M=np.array(M)
    
    D2i=np.where(D2>.2)[0]
    D2=D2[D2i]
    M=M[D2i]
    if errs is not None:
        errs=np.array(errs)
        errs=errs[D2i]
    if errs is not None:
      op2=curve_fit(d2model, M, D2,p0=[2,2] ,sigma=errs)
    else:
      op2=curve_fit(d2model, M, D2,p0=[2,2])
    bf=d2model(M, op2[0][0], op2[0][1])
    if errs is not None:
        chi2=calc_chi2(D2,errs,bf)
    else:
        chi2=np.nan
    return(op2[0][1],int(op2[0][0])+1,chi2)

def check_sat(M,D2,errs=None):
    """
    Checks for saturation and returns D2sat for given D2 and M.
    Parameter
    ---------
    M : list or 1-d array
        Embedding dimensions for values of D2
    D2 : list or 1-d array
        D2 for each value of M.
    errs: list or 1-d array
	Error for each value of D2
    Returns
    -------
    d2sat: float
	Saturated D2 value after function fitting
    OR 'NS' if no saturation
    """
    if errs is None:
        print("No error array found. Termininating")
        return(np.nan)
    d2sat,em,chi2=find_d2s(M,D2,errs)
    print (em,d2sat)
    if((d2sat>em-1) and(d2sat<em+1)):
        print("Saturated D2 very close to maximum embedding dimension")
        return("NS")
    D2=np.array(D2)
    M=np.array(M)
    D2i=np.where(D2>.2)[0]
    D2=D2[D2i]
    M=M[D2i]
    errs=np.array(errs)
    errs=errs[D2i]
    slope, intercept, r_value, p_value, std_err = sp.linregress(M,D2)
    d2f=np.zeros(len(M))
    for i in range (0,len(M)):
        d2f[i]=slope*M[i]+intercept
    chi2_lin=calc_chi2(D2,errs,d2f)
    print(chi2,chi2_lin)
    if(chi2_lin<chi2):
        print("Linear fit is better. No Saturation.y=",slope,'x+', intercept)
        return('NS')
    else:
        print("Saturated D2 fit is better than linear fit")
        return(d2sat)
