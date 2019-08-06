import math
import numpy as np
from scipy import special
#import scipy as sp
#from scipy import interpolate
#import scipy.signal as sig
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
#from matplotlib.widgets import TextBox
#from matplotlib.widgets import Button
import json
import os.path
import lmfit
from lmfit import Model, Parameter
from lmfit.models import SkewedVoigtModel, LinearModel, GaussianModel, LorentzianModel
import emcee
#import csv
import time
import numdifftools
import LoadingAndMungingData as lmd
import FrequencyConvertedUtilityKit as FCUK

def fitNPeaks(m, datFrame, peaksList, peakSigmas=np.array([]), method='leastsq', useWeights=False, sameSkew=True, skewList=np.array([]), skew0="NaN", initGamma=1): #fit scan data with rebinning to spectrum with pre-guessed peaks
#TODO: Allow skew as input parameter, and gamma vs sigma
  N = len(peaksList)
  if len(peakSigmas) == 0:
    peakSigmas = 2*np.ones_like(peaksList)
  else:
    assert(len(peakSigmas) == N)
    assert(np.all(peakSigmas>0))
  if type(skew0) == 'float':
    skewList = skew0*np.ones_like(peaksList)
  else:
    if len(skewList) == 0:
      skewList = -2*np.ones_like(peaksList)
    else:
      assert(len(skewList) == N)
    if sameSkew and (np.any(skewList[1:]!=skewList[0])):
      print("Just a heads up, you input a list of distinct skew values, but sameSkew==True, so the first skew value will be used for each peak. Set sameSkew to False if you dislike this.")
      skewList=skewList[0]*np.ones_like(skewList)
  #if sameSkew: skewList=skewList[0]*np.ones_like(skewList)
  print("test: skewList:",skewList,"\npeaksList:",peaksList)
  xDat = np.array(datFrame.loc[:,'wavenumber_mean']); yDat = np.array(datFrame.loc[:,'signal_value']); ySigDat = np.array(datFrame.loc[:,'signal_uncertainty'])
  xRange = np.max(xDat) - np.min(xDat); yRange = np.max(yDat) - np.min(yDat)
  bg = backgroundEstimator(yDat)
  lmod = LinearModel(prefix='l0_')
  lmod.set_param_hint('slope', value=0, min = -(np.max(yDat)-np.min(yDat)), max=np.max(yDat)-np.min(yDat))
  #params = lmod.guess(yDat, x=xDat)
  params = lmod.make_params(intercept=bg, slope=0)
  peakModelsArray = []
  warningStatus=0
  for i in range(N):
    #print("Adding peak%d to model fit. center at %.2f"%(i,peaksList[i]))
    k = peaksList[i]; sigmaK = peakSigmas[i] 
    print("k=%.2f"%k)
    ind1 = np.argmin(abs(xDat-k))
    print("ind1=%d"%ind1)
    ind2 = np.argmax(yDat[ind1-2:ind1+3])+ind1-2
    estimHeight = yDat[ind2] - bg
    print("testing estims... bg=%d, i=%d, k=%.2f, ind1=%d, ind2=%d, xDat[ind2]=%.2f, yDat[ind2=]%.2f, estimHeight=%.2f"%(bg, i,k,ind1,ind2,xDat[ind2],yDat[ind2],estimHeight))
    if ind1 <= 3 or len(xDat)-ind1<=3:
      print("WARNING: Peak occurs too closely to edge of dataset. A lower rebin setting is recommended.")
      warningStatus=-1
      return(False, warningStatus)
    if k-.66>xDat[8]:
      k2 = peaksList[i] -.66; #pea8ksList[i], may be the "center", but it may not be where distribution is maximized. -.66 cm^{-1} is the shift for gamma=1.5,sigma=.8,skew=-2
      ind1 = np.argmin(abs(xDat-k2))
      ind2 = np.argmax(yDat[ind1-7:ind1+8])+ind1-7
      estimHeight2 = yDat[ind2] - bg
      if estimHeight2>estimHeight: print("aha! Had to look left to find global maximum!")
      estimHeight = max(estimHeight,estimHeight2)
    amp=estimHeight*(peakSigmas[i]*math.sqrt(2*math.pi))/special.wofz((1j*initGamma)/(peakSigmas[i]*math.sqrt(2))).real
    height1=amp*special.wofz((1j*initGamma)/(peakSigmas[i]*math.sqrt(2))).real/(peakSigmas[i]*math.sqrt(2*math.pi))
    print("testing math stuff... amp=%.2f; height=%.2f"%(amp,height1))
    svmod = SkewedVoigtModel(prefix="sv"+str(i)+"_")
    svmod.set_param_hint('center', value=peaksList[i], min=max(peaksList[i]-2*peakSigmas[i], xDat[3]), max=min(peaksList[i]+2*peakSigmas[i],xDat[-3]))
    svmod.set_param_hint('sigma', value=peakSigmas[i], min=0.1, max=2*peakSigmas[i])
    #svmod.set_param_hint('amplitude', value=estimHeight*((initGamma+peakSigmas[i])/(1*.45)), min=3*np.mean(ySigDat))
    #svmod.set_param_hint('amplitude', value=amp, min=3*np.mean(ySigDat))
    svmod.set_param_hint('amplitude', value=amp*.6, min=3*np.mean(ySigDat))#?
    svmod.set_param_hint('skew', value = skewList[i], min=-12,max=12)
    """if N>2: svmod.set_param_hint('skew', value = -4, min=-12,max=12)
    elif N<=2: svmod.set_param_hint('skew', value = 0, min=-1,max=1, vary=True)"""
    peakModelsArray.append(svmod)
    params += svmod.make_params()#svmod.guess(yDat, x=xDat)#
    if i == 0: params['sv'+str(i)+'_gamma']= Parameter(value=initGamma, min=0, max = 3*peakSigmas[i], vary=True)
    elif i>0:
      params['sv'+str(i)+'_gamma'] = Parameter(expr='sv0_gamma')
      if sameSkew:
        params['sv'+str(i)+'_skew'] = Parameter(expr='sv0_skew') #TODO: Test out this constraint now that my initial parm ests are good.

  mod = np.sum(peakModelsArray)+lmod
  #print("test: gamma sv0_= %d"%params.valuesdict()['sv0_gamma'])
  #for pname, par in params.items():
    #print(pname,par)
  print("parameters initialized. Starting fit now.")
  if useWeights: fitResult=mod.fit(yDat, params, x=xDat, method=method, weights=(1/np.square(ySigDat))/np.sum(1/np.square(ySigDat)) )
  else: fitResult=mod.fit(yDat, params, x=xDat, method=method)
  #print(fitResult.fit_report(min_correl=0.25))
  return(fitResult, warningStatus)

#will this help with speed, or are these values already stored in memory anyway?
sqrt2 = math.sqrt(2)
sqrt2pi = math.sqrt(2*math.pi)

def voigt(x, A, mu, sigma, gamma):
  #returns a Voigt distribution
  u=x-mu
  return(A*np.real( special.wofz((u+1j*gamma)/(sqrt2*sigma)) )/(math.sqrt(2*math.pi)*sigma) )

def skewedVoigt(x, A, mu, sigma, gamma, skew=0):
  skewFactor=1+(special.erf(skew*(x-mu))/(sqrt2*sigma))
  return(voigt(x,A,mu,sigma,gamma)*skewFactor)

def fit_function(params, xArray=None, yArray=None):
  #TODOOOO
  return(-1)
if __name__ == '__main__':
  xDat = np.linspace(13255,13290, num=350)
  yDat = skewedVoigt(xDat, 100, 13284, .8, 1.6, skew=-4)
  plt.plot(xDat,yDat)
  plt.show()