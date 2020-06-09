import math
import numpy as np
import pandas as pd
import os.path
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from scipy import special
import lmfit
from lmfit import Model, Parameter
from lmfit.models import SkewedVoigtModel, LinearModel, ConstantModel, GaussianModel, LorentzianModel
import emcee
#import csv
import time
import numdifftools
import LoadingAndMungingData as lmd

'''3. "Analyse each scan individually and extract an average "peak position" for each electronic transition "'''
def backgroundEstimator(yArray):
  l = len(yArray)
  orderedByHeight = np.sort(yArray)
  beegee = np.mean(orderedByHeight[0:int(l/8)])
  return(beegee)

def voigt(x, A, mu, sigma, gamma):
  #returns a Voigt distribution
  return(A*np.real( special.wofz(((x-mu)+1j*gamma)/(math.sqrt(2)*sigma)) )/(math.sqrt(2*math.pi)*sigma) )

def skewedVoigt(x, A, mu, sigma, gamma, skew):
  skewFactor=1+(special.erf(skew*(x-mu))/(math.sqrt(2)*sigma))
  return(voigt(x,A,mu,sigma,gamma)*skewFactor)

'''
def fitPlottingSubRoutine(datFrame, mass, s, r, n, fitRes, redchi=-1, currDir='./'):
  #fullDatArray=datDic[x]
  #datArray=dataRebinner(fullDatArray,r) #inputting full dataset so that fit can be plotted at higher resolution for large-r datasets
  scan = str(s)
  print("making res=%.3f, n=%d plot for mass%d scan %s"%(r,n,mass,scan))
  fitDic = fitRes.best_values
  fitPlotFig = plt.figure()
  plt.gcf().set_size_inches(20, 12)
  ax = plt.gca()
  xDat = np.array(datFrame.loc[:,'wavenumber_mean']); yDat = np.array(datFrame.loc[:,'signal_value']); ySigDat = np.array(datFrame.loc[:,'signal_uncertainty'])
  xFull = np.arange(np.min(xDat)-.05,np.max(xDat)+.05,.01)
  plt.errorbar(xDat, yDat, yerr=ySigDat, fmt='b.-',ecolor='k', alpha=.3)
  plt.fill_between(xDat, yDat,color='blue', alpha=.25)
  plt.xlabel(r'Wavenumber (cm$^{-1}$)', fontsize=18)
  plt.ylabel('rate (counts/s)', fontsize=18)
  plt.title(r'$^{%d}$Ra$^{19}$F Spectrum; Scan %s; Resolution = %.2f, Fitting to %d peaks' %(mass-19, scan, r, n), fontsize=24)
  plt.plot(xFull, fitRes.eval(x=xFull,params=fitRes.init_params), 'k--', label="init_Fit", linewidth=2)
  plt.plot(xFull, fitRes.eval(x=xFull,params=fitRes.params), 'r-', label="best_fit",linewidth=4)
  comps = fitRes.eval_components(x=xFull)
  bg = backgroundEstimator(yDat)
  plt.plot(xFull, comps['l0_'],'--', label="linear term")
  for i in range(n):
    color = next(ax._get_lines.prop_cycler)['color']
    xPoint=xFull[np.argmax(comps['sv'+str(i)+'_'])]; yPoint=np.max(comps['sv'+str(i)+'_']);
    datMaxEsts = findLocalMax(datFrame, xPoint, .5, uncertIndex=3)
    xLine=fitDic['sv'+str(i)+'_center']
    plt.plot(xFull, comps['sv'+str(i)+'_'], '--', color=color, label="peak "+str(i), linewidth=3)
    plt.plot(xPoint, yPoint, "o", color=color)
    plt.errorbar(datMaxEsts[0,0], datMaxEsts[1,0], xerr=datMaxEsts[0,1], yerr=datMaxEsts[1,1], fmt='bo', ecolor='k', alpha=.5)
    plt.axvline(xLine, 0,1, color=color, linestyle='dashed')
    plt.annotate(s=r'$\mu_%d = %.3f$'%(i, xLine), xy=(xLine, bg/4-i*bg/(n*4)), fontsize=14, ha='center', xycoords=('data','data'))
    plt.annotate(s=r'max$_{fit}\nu_%d = %.2f$'%(i, xPoint), xy=(xPoint, yPoint), fontsize=14, ha='center', va='bottom', xycoords=('data','data'), color=color)
    plt.annotate(s=r'max$_{dat}\nu_%d = %.2f$'%(i, datMaxEsts[0,0]), xy=(datMaxEsts[0,0], datMaxEsts[1,0]*1.05), fontsize=14, ha='center', va='bottom', xycoords=('data','data'),color='blue')
    
  plt.legend(loc=2, fontsize=18)
  if redchi>0: plt.annotate(s=r'$\chi_{red}^2 = %f$'%redchi, xy=(.25,.2), fontsize=14, ha='center', xycoords=('figure fraction','figure fraction'))
  if currDir=='./': currDir = './FitResults/mass%dFits/Scan%sFits/'%(mass,scan)
  if not os.path.exists(currDir): os.makedirs(currDir)
  plt.gcf().savefig(currDir+"Mass%d_Scan%s_%dbins_%dpeaksFit.png"%(mass,scan,len(datFrame.index),n))
  plt.close()'''

def fitPlottingSubsampleRoutine(datFrame, mass, s, r, k, n, fitRes, redchi=-1, currDir='./'):
  #plots fit for data in datFrame, corresponding to isotope of certain mass, scan s, resolution setting r, random subsample k, with fitresults fitRes
  #fullDatArray=datDic[x]
  #datArray=dataRebinner(fullDatArray,r) #inputting full dataset so that fit can be plotted at higher resolution for large-r datasets
  scan = str(s)
  #print("making res=%.3f, n=%d plot for mass%d scan %s\nrandom subsample # %d"%(r,n,mass,scan,k))
  fitDic = fitRes.best_values
  fitPlotFig = plt.figure()
  plt.gcf().set_size_inches(20, 12)
  ax = plt.gca()
  xDat = np.array(datFrame.loc[:,'wavenumber_mean']); yDat = np.array(datFrame.loc[:,'signal_value']); ySigDat = np.array(datFrame.loc[:,'signal_uncertainty'])
  xFull = np.arange(np.min(xDat)-.05,np.max(xDat)+.05,.01)
  plt.errorbar(xDat, yDat, yerr=ySigDat, fmt='b.-',ecolor='k', alpha=.3)
  plt.fill_between(xDat, yDat,color='blue', alpha=.25)
  plt.xlabel(r'Wavenumber (cm$^{-1}$)', fontsize=18)
  plt.ylabel('rate (counts/s)', fontsize=18)
  plt.title(r'$^{%d}$Ra$^{19}$F Spectrum; Scan %s; Resolution = %.2f, Fitting to %d peaks;  random subsample # %d' %(mass-19, scan, r, n, k), fontsize=24)
  plt.plot(xFull, fitRes.eval(x=xFull,params=fitRes.init_params), 'k--', label="init_Fit", linewidth=2)
  plt.plot(xFull, fitRes.eval(x=xFull,params=fitRes.params), 'r-', label="best_fit",linewidth=4)
  comps = fitRes.eval_components(x=xFull)
  bg = backgroundEstimator(yDat)
  plt.plot(xFull, comps['l0_'],'--', label="linear term")
  for i in range(n):
    color = next(ax._get_lines.prop_cycler)['color']
    xPoint=xFull[np.argmax(comps['sv'+str(i)+'_'])]; yPoint=np.max(comps['sv'+str(i)+'_']);
    datMaxEsts = findLocalMax(datFrame, xPoint, .5, uncertIndex=3)
    xLine=fitDic['sv'+str(i)+'_center']
    plt.plot(xFull, comps['sv'+str(i)+'_'], '--', color=color, label="peak "+str(i), linewidth=3)
    plt.plot(xPoint, yPoint, "o", color=color)
    plt.errorbar(datMaxEsts[0,0], datMaxEsts[1,0], xerr=datMaxEsts[0,1], yerr=datMaxEsts[1,1], fmt='bo', ecolor='k', alpha=.5)
    plt.axvline(xLine, 0,1, color=color, linestyle='dashed')
    plt.annotate(s=r'$\mu_%d = %.3f$'%(i, xLine), xy=(xLine, bg/4-i*bg/(n*4)), fontsize=14, ha='center', xycoords=('data','data'))
    plt.annotate(s=r'max$_{fit}\nu_%d = %.2f$'%(i, xPoint), xy=(xPoint, yPoint), fontsize=14, ha='center', va='bottom', xycoords=('data','data'), color=color)
    plt.annotate(s=r'max$_{dat}\nu_%d = %.2f$'%(i, datMaxEsts[0,0]), xy=(datMaxEsts[0,0], datMaxEsts[1,0]*1.05), fontsize=14, ha='center', va='bottom', xycoords=('data','data'),color='blue')
    
  plt.legend(loc=2, fontsize=18)
  if redchi>0: plt.annotate(s=r'$\chi_{red}^2 = %f$'%redchi, xy=(.25,.2), fontsize=14, ha='center', xycoords=('figure fraction','figure fraction'))
  if currDir=='./': currDir = './FitResults/mass%dFits/Scan%sFits/'%(mass,scan)
  if not os.path.exists(currDir): os.makedirs(currDir)
  plt.gcf().savefig(currDir+"Mass%d_Scan%s_%dbins_%dpeaksFit_Subsamp%d.png"%(mass,scan,len(datFrame.index),n,k))
  plt.close()

def fitNPeaks(datFrame, peaksList, peakSigmas=np.array([]), method='leastsq', useWeights=True, sameSkew=True, skewList=np.array([]), skew0="NaN", initGamma=1,sameSigma=True, similarSigma=True, linearTerm=False, BolzmannHeights=False, T=-1,Ei=-1, attempt=1, verbose=False):
#fit scan data with rebinning to spectrum with pre-guessed peaks
  # TODO incorporate similarSigma idea!
  N = len(peaksList)
  if type(peakSigmas)==float:
    peakSigmas = peakSigmas*np.ones_like(peaksList)
  elif len(peakSigmas) == 0:
    peakSigmas = 1*np.ones_like(peaksList)
  assert(len(peakSigmas) == N)
  assert(np.all(peakSigmas>0))
  if sameSigma and (np.any(peakSigmas[1:]!=peakSigmas[0]))  :
    print("Just a heads up, you input a list of distinct sigma values, but sameSigma==True, so the first sigma value will be used for each peak. Set sameSigma to False if you dislike this.")
    peakSigmas=peakSigmas[0]*np.ones_like(peakSigmas)

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
  #print("test: skewList:",skewList,"\npeaksList:",peaksList)
  xDat = np.array(datFrame.loc[:,'wavenumber_mean']); yDat = np.array(datFrame.loc[:,'signal_value']); ySigDat = np.array(datFrame.loc[:,'signal_uncertainty'])
  xRange = np.max(xDat) - np.min(xDat); yRange = np.max(yDat) - np.min(yDat)
  bg = backgroundEstimator(yDat)
  lmod = LinearModel(prefix='l0_')
  lmod.set_param_hint('slope', value=0, min = -(np.max(yDat)-np.min(yDat)), max=np.max(yDat)-np.min(yDat),vary=linearTerm)
  #params = lmod.guess(yDat, x=xDat)
  params = lmod.make_params(intercept=bg)
  peakModelsArray = []
  warningStatus=0

  #Putting Height estimation for peak 0 here just so I can allow for Boltzmann estimates in the future
  locMaxOutput = findLocalMax(datFrame, peaksList[0]-.5, 1, uncertIndex=4)
  estimHeight0 = (locMaxOutput[1,0]-locMaxOutput[1,1]/2) - bg #I'm subtracting off half of the "uncertainty" from my height estimate, so random spikes won't ruing my initial guess on noisier scans
  amp0=estimHeight0*(peakSigmas[0]*math.sqrt(2*math.pi))/special.wofz((1j*initGamma)/(peakSigmas[0]*math.sqrt(2))).real

  #Beginning to add peaks to fit and guess initial parameter values
  for i in range(N):
    k = peaksList[i]; sigmaK = peakSigmas[i] 
    ind1 = np.argmin(abs(xDat-k))
    #print("testing estims... bg=%d, i=%d, k=%.2f, ind1=%d, ind2=%d, xDat[ind2]=%.2f, yDat[ind2=]%.2f, estimHeight=%.2f"%(bg, i,k,ind1,ind2,xDat[ind2],yDat[ind2],estimHeight))
    if ind1 <= 3 or len(xDat)-ind1<=3:
      print("WARNING: Peak occurs too closely to edge of dataset. A lower rebin setting is recommended.")
      warningStatus=-1
      return(False, warningStatus)
    if i==0:
      amp=amp0; 
      if verbose: print("Adding peak %d to model. new estimHeight = %.2f"%(i,estimHeight0))
    elif (BolzmannHeights and i>0) and (T>0 and Ei>0):
      kBoltzmann = 0.695035 #Boltzmann constant in cm^-1/k (aka give energies Ei as cm^-1)
      propFactor = math.exp(-i*Ei/(T*kBoltzmann))
      print("test: propFactor=%.3f"%propFactor)
      amp=propFactor*amp0
    else:
      locMaxOutput = findLocalMax(datFrame, peaksList[i]-.5, 1, uncertIndex=4)
      estimHeight = (locMaxOutput[1,0]-locMaxOutput[1,1]/2) - bg - math.sqrt(bg)/2 #I'm subtracting off half of the "uncertainty" from my height estimate, so random spikes won't ruin my initial guess on noisier scans
      if verbose: print("Adding peak %d to model. new estimHeight = %.2f"%(i,estimHeight))
      amp=estimHeight*(peakSigmas[i]*math.sqrt(2*math.pi))/special.wofz((1j*initGamma)/(peakSigmas[i]*math.sqrt(2))).real
    
    #height1=amp*special.wofz((1j*initGamma)/(peakSigmas[i]*math.sqrt(2))).real/(peakSigmas[i]*math.sqrt(2*math.pi))
    #print("testing math stuff... amp=%.2f; height=%.2f"%(amp,height1))
    svmod = SkewedVoigtModel(prefix="sv"+str(i)+"_")
    svmod.set_param_hint('center', value=peaksList[i], min=max(peaksList[i]-peakSigmas[i], xDat[3]), max=min(peaksList[i]+peakSigmas[i],xDat[-3]))
    svmod.set_param_hint('sigma', value=peakSigmas[i], min=0.1, max=3)
    #svmod.set_param_hint('amplitude', value=estimHeight*((initGamma+peakSigmas[i])/(1*.45)), min=3*np.mean(ySigDat))
    #svmod.set_param_hint('amplitude', value=amp, min=3*np.mean(ySigDat))
    svmod.set_param_hint('amplitude', value=amp*.5, min=3*np.mean(ySigDat)*(np.min(peakSigmas)*math.sqrt(2*math.pi))/special.wofz((1j*initGamma)/(peakSigmas[i]*math.sqrt(2))).real, max = 1.5*amp)#?
    svmod.set_param_hint('skew', value = skewList[i], min=-4,max=0)
    peakModelsArray.append(svmod)
    params += svmod.make_params()#svmod.guess(yDat, x=xDat)#
    if i == 0: params['sv'+str(i)+'_gamma']= Parameter('sv'+str(i)+'_gamma', value=initGamma, min=0, max = 3, vary=True)
    elif i>0:
      params['sv'+str(i)+'_gamma'] = Parameter('sv'+str(i)+'_gamma', expr='sv0_gamma')
      if sameSkew: params['sv'+str(i)+'_skew'] = Parameter('sv'+str(i)+'_skew', expr='sv0_skew')
      if i<3 and sameSigma: params['sv'+str(i)+'_sigma'] = Parameter('sv'+str(i)+'_sigma', expr='sv0_sigma')

  mod = np.sum(peakModelsArray)+lmod
  if attempt==1: print("parameters initialized. Starting fit now.")
  if useWeights: fitResult=mod.fit(yDat, params, x=xDat, method=method, weights=(1/np.square(ySigDat))/np.sum(1/np.square(ySigDat)) )
  else: fitResult=mod.fit(yDat, params, x=xDat, method=method)
  minAmp = 3*np.mean(ySigDat)*(peakSigmas[-1]*math.sqrt(2*math.pi))/special.wofz((1j*initGamma)/(peakSigmas[i]*math.sqrt(2))).real
  reducePeaksCondit = (abs(fitResult.best_values['sv'+str(N-1)+'_amplitude'] - minAmp)<1 or fitResult.best_values['sv'+str(N-1)+'_amplitude'] > fitResult.best_values['sv0_amplitude'] or
    abs(fitResult.best_values['sv'+str(N-1)+'_sigma'] - 3)<0.05 or abs(fitResult.best_values['sv'+str(N-1)+'_gamma'] - 3)<0.05 or
    abs(fitResult.best_values['sv'+str(N-1)+'_skew']+4)<.05 or abs(fitResult.best_values['sv'+str(N-1)+'_skew'])<0.05 or
    abs(fitResult.best_values['sv'+str(N-1)+'_center'] - max(peaksList[-1]-peakSigmas[-1], xDat[3]))<0.05 or 
    abs(fitResult.best_values['sv'+str(N-1)+'_center'] - min(peaksList[-1]+peakSigmas[-1],xDat[-3]))<0.05 )
  if fitResult.errorbars and not(reducePeaksCondit): return(fitResult, warningStatus, len(peaksList))
  else:
    if attempt==3:
      print("bad last try :( .test:", fitResult.best_values['sv'+str(N-1)+'_amplitude'], minAmp, "; reducePeaksCondit = ", reducePeaksCondit)
      if len(peaksList)>2 and reducePeaksCondit:  #If estimated amp of left-most peak is minimum allowed value, try again without fitting to that peak.
        print("3rd attempt to fit with errorbars has failed due to poor SNR for left-most peak. Will start over fitting to only %d peaks."%len(peaksList[:-1]))
        return(fitNPeaks(datFrame, peaksList[:-1]+0.1*np.ones_like(peaksList[:-1]), peakSigmas=peakSigmas[:-1], initGamma=initGamma, useWeights=useWeights, sameSkew=sameSkew, sameSigma=sameSigma, similarSigma=similarSigma, linearTerm=linearTerm, method=method))
      
      else: print("this was the 3rd attempt and we still couldn't give errorbars. I give up :/"); return(fitResult, warningStatus, len(peaksList))
    else:
      print("Fook... Attempt %d unsuccessful. Will attempt again with something changed?"%attempt)
      return(fitNPeaks(datFrame, peaksList-0.05*attempt*np.ones_like(peaksList), peakSigmas=peakSigmas, initGamma=initGamma, useWeights=useWeights, sameSkew=sameSkew, sameSigma=sameSigma, similarSigma=similarSigma, linearTerm=linearTerm, method=method,attempt=attempt+1))

def findLocalMax(datFrame, guess, searchWidth, xcol="wavenumber_mean",ycol="signal_value", uncertIndex=3, verbose=False):
  cropDat=lmd.trimRange(datFrame.copy(),ltrim=guess-searchWidth,rtrim=guess+searchWidth).loc[:,[xcol,ycol]].sort_values(by=ycol,ascending=False)
  xValsByY = np.array(cropDat.loc[:,xcol]); yValsByY = np.array(cropDat.loc[:,ycol])
  if len(xValsByY)<uncertIndex+1:
    uncertIndex=len(xValsByY)-1
  if verbose: print("uncertIndex was larger than length of trimmed array, had to reduce to uncertIndex=%d"%uncertIndex)
  #print("findLocalMaxTests. cropDat:\n",cropDat.head(),"\n, xValsByY:%s\nyValsByY:%s"%(xValsByY,yValsByY))
  return(np.array([[xValsByY[0],abs(xValsByY[0]-xValsByY[uncertIndex])],[yValsByY[0],abs(yValsByY[0]-yValsByY[uncertIndex])]]))

def fitScanX(mass, s, peaksList, peakSigmas=np.array([]), initGamma=1, resList=[.01,.02,.05,.1,.2,.5], method='leastsq',similarSigma=True, makePlots=True,spreadPlot=True, 
  sameSkew=True, sameSigma=True, useWeights=True, ltrim=-1, rtrim=-1, skewList=np.array([]), skew0='NaN',linearTerm=False,frac=1, nSamps=1):
  #fit scan x data with rebinning R to spectrum with N peaks
  scan = str(s)
  print("now running fitScanX for mass = %d; scan: %s"%(mass, scan))
  if type(s) == list:
    print("ooh boy! We're doing some combo fits today, buddy!")
    scanFrame = lmd.mergeDatRaw(mass, s)
  else:
    scanFrame = lmd.rawDatPrep(mass,s)
  currDir = './FitResults/Mass%dFits/Scan%sFits/'%(mass,scan)
  if not os.path.exists(currDir): os.makedirs(currDir)

  if type(peakSigmas)==float:
    peakSigmas = peakSigmas*np.ones_like(peaksList)
  elif len(peakSigmas) == 0:
    peakSigmas = 1*np.ones_like(peaksList)
  else:
    assert(len(peakSigmas) == len(peaksList))
    assert(np.all(peakSigmas>0))
  assert(len(resList)>0)
  R = len(resList)
  minR=1
  N = len(peaksList)
  resList = np.sort(resList)

  compiledGoFs = np.full((R-minR+1,nSamps), np.nan)
  compiledFitResults = {}
  compiledCenterEsts = np.full((R, nSamps, N, 2), np.nan)
  compiledDataLocalMax = np.full((R, nSamps, N, 2), np.nan)
  compiledFitLocalMax = np.full((R, nSamps, N, 2), np.nan)
  multiInd=pd.MultiIndex.from_product([resList,np.array(range(nSamps))],names=['resolution','sample#'])
  compiledResults = pd.DataFrame(columns=["GoF", 'fitResults', 'PeakEsts', 'dataLocalMax', 'fitLocalMax'], index=multiInd)#,type=['float',])

  """Above this line is just prepwork"""
  for i in range(len(resList)):
    r=resList[i]
    #resolutionPath = currDir+'Mass%dScan%sres%s/'%(mass,scan,str(r))
    resolutionPath = currDir+'res%s/'%str(r)
    if not os.path.exists(resolutionPath): os.makedirs(resolutionPath)
    print("resolution=", r)
    subSamps = lmd.randSubsets(scanFrame, frac, nSamps)
    for k in range(nSamps):
      print("resolution %d of %d, subsample %d of %d" %(i, len(resList)-1, k, nSamps-1) )
      #print("Another Test: k = ",k)
      datFrame = lmd.makeUseable(subSamps[k], resolution=r, cropSparseEnds=True, noNaNsense=True, ltrim=ltrim, rtrim=rtrim, verbose=False)
      #print("TESTSTSSTST:\n", datFrame)
      (fitRes, warningStatus, numPeaksUsed) = fitNPeaks(datFrame, peaksList, peakSigmas=peakSigmas, initGamma=initGamma, useWeights=useWeights, sameSkew=sameSkew, sameSigma=sameSigma, similarSigma=similarSigma,linearTerm=linearTerm)#add other opts?
      if warningStatus == -1:
        print("That's it for this scan, boys. Don't. push. these peaks. They're. close. to. the. eeeedge. (One of the peaks is leaking out of the scan window at rebin setting%d)"%r)
        (ccrx,ccex,cdlm,cflm)=(compiledGoFs[:i], compiledCenterEsts[:i], compiledDataLocalMax[:i],compiledFitLocalMax[:i])
        ExportResults(currDir+'Mass%d_Scan%s_%dPeaks'%(mass,scan,N), [ccrx,ccex,cdlm,cflm])
        return(ccrx,ccex,cdlm,cflm)#return(compiledGoFs[:i], compiledCenterEsts[:i], compiledDataLocalMax[:i],compiledFitLocalMax[:i])
      fitReportFile = open(resolutionPath+'randSamp%d_FitReport.txt'%k,'w+')
      fitReportFile.write("Fit Report for: Mass = %d, Scan = %s, Resolution = %.3f, randSamp#%d\n"%(mass,scan,r,k)); fitReportFile.write(fitRes.fit_report(min_correl=0.25)); fitReportFile.close()
      compiledFitResults[r,k] = fitRes.best_values
      compiledResults.loc[(r,k),"fitResults"] = [fitRes.best_values]
      fitCenterEsts = np.full((N,2),np.nan); fitLocalMaxEsts = np.full((N,2),np.nan); datLocalMaxEsts = np.full((N,2),np.nan);
      parmCenterNames = ['sv'+str(j)+'_center' for j in range(N)].append(['l0_slope', 'l0_intercept'])
      kwargs = {'p_names':parmCenterNames}
      #print("fitRes.errorbars", fitRes.errorbars)
      xDat = np.array(datFrame.loc[:,'wavenumber_mean']); xFull = np.arange(np.min(xDat)-.05,np.max(xDat)+.05,.01)
      comps = fitRes.eval_components(x=xFull)
      for p in range(numPeaksUsed):
        mu_p = fitRes.best_values['sv'+str(p)+'_center'];
        sigma_p = fitRes.best_values['sv'+str(p)+'_sigma']; gamma_p = fitRes.best_values['sv'+str(p)+'_gamma'];
        mu_pUncert = fitRes.params['sv'+str(p)+'_center'].stderr if fitRes.errorbars else max(sigma_p, gamma_p)
        '''sigma_p = fitRes.best_values['sv'+str(p)+'_sigma'];
        gamma_p = fitRes.best_values['sv'+str(p)+'_gamma'];
        skew_p = fitRes.best_values['sv'+str(p)+'_skew'];
        xSimp = np.arange(mu_p-(sigma_p+gamma_p),mu_p+(sigma_p+gamma_p),.01); ySimp=skewedVoigt(xSimp,1,mu_p,sigma_p,gamma_p,skew_p)
        fitLocMax_p = xSimp[np.argmax(ySimp)];''' #WTF? Is my skewedVoigt function not good enough for you, bitch!? (p.s. it looks like no, it's not...)
        fitLocMax_p=xFull[np.argmax(comps['sv'+str(p)+'_'])];
        datLocalEst_p = findLocalMax(datFrame, fitLocMax_p, 0.5, uncertIndex=3)
        fitCenterEsts[p, 0] = mu_p; fitCenterEsts[p,1] = mu_pUncert
        fitLocalMaxEsts[p,0] = fitLocMax_p; fitLocalMaxEsts[p,1] = mu_pUncert
        datLocalMaxEsts[p,0] = datLocalEst_p[0,0] ; datLocalMaxEsts[p,1] = datLocalEst_p[0,1]

      if makePlots == True: fitPlottingSubsampleRoutine(datFrame, mass, s, r, k, numPeaksUsed, fitRes, redchi=fitRes.redchi, currDir=resolutionPath)
      compiledCenterEsts[i,k,:,:] = fitCenterEsts
      compiledDataLocalMax[i,k,:,:] = datLocalMaxEsts
      compiledFitLocalMax[i,k,:,:] = fitLocalMaxEsts
      compiledGoFs[i,k] = fitRes.redchi
      #print("test2: type(fitCenterEsts) = ", type(fitCenterEsts) )
      compiledResults.loc[(r,k),'PeakEsts'] = [fitCenterEsts]
      compiledResults.loc[(r,k),'dataLocalMax'] = [datLocalMaxEsts]
      compiledResults.loc[(r,k),'fitLocalMax'] = [fitLocalMaxEsts]
      compiledResults.loc[(r,k),'GoF'] = fitRes.redchi
      print("Just ran fitScanX routine for: scan %s; res = %.3f; N=%d, nSamp=%d and found redchi = %f"%(scan,r,N,k,fitRes.redchi))#, "\nfitCenterEsts:", fitCenterEsts[:,0], "\nfitPeakEsts:",datLocalMaxEsts[:,0])    

  ccrx=compiledGoFs; ccex=compiledCenterEsts; cdlm=compiledDataLocalMax; cflm=compiledFitLocalMax
  #print("Hmmmm.. So Close. ccex:\n", ccex)
  #print("cdlm:\n", cdlm)
  #print("cflm:\n", cflm)
  finCenterEst = scanFitsFinalEsts(ccex)
  #print("Does this work? finCenterEst = ", finCenterEst)
  finDatLocMaxEst = scanFitsFinalEsts(cdlm)
  finFitLocMaxEst = scanFitsFinalEsts(cflm)
  print("Final Center Estimate:\n%s\nFinal Local Max Ests from fits:\n%s\nFinal Local Max Ests from data:\n%s\n"%(str(finCenterEst),str(finFitLocMaxEst),str(finDatLocMaxEst)))
  #print("report from last fit performed:\n",fitRes.fit_report())
  if not os.path.exists(currDir+'AllFitResultsCompiled/'): os.makedirs(currDir+'AllFitResultsCompiled/')
  compiledResultsFile = open(currDir+'AllFitResultsCompiled/Mass%dScan%s-%dPeaks.txt'%(mass,scan,N),'w+')
  compiledResultsFile.write('Mass: %d ; Scan: %s\nInitial peak center estimates:'%(mass,scan)+str(peaksList)); compiledResultsFile.write('\nInitial peak width estimates:'+ str(peakSigmas));
  compiledResultsFile.write("\nCompilation of Results:\n")
  compiledResultsFile.write("Final Center Estimate:\n%s\nFinal Local Max Ests from fits:\n%s\nFinal Local Max Ests from data:\n%s\n"%(str(finCenterEst),str(finFitLocMaxEst),str(finDatLocMaxEst)));
  with pd.option_context('display.max_rows', 100, 'display.max_columns', 200): compiledResultsFile.write(compiledResults.to_csv()); compiledResultsFile.close()
  ExportResults(currDir+'Mass%d_Scan%s_%dPeaks'%(mass,scan,N), [ccrx,ccex,cdlm,cflm])

  if spreadPlot==True:
    MultiBinSpreadPlotter(mass, scan, ccex, resList, currDir=currDir)
    MultiBinSpreadPlotter(mass, scan, cdlm, resList, currDir=currDir, quantity="local maxima from data")
    MultiBinSpreadPlotter(mass, scan, cflm, resList, currDir=currDir, quantity="local maxima from fits")
  return(ccrx,ccex,cdlm,cflm) #TODO?: get peak estimates, apply them to next fit. Carry out next fit(s) -> Nahhh

def ExportResults(filePrefix, resultsList):
  #exports fitScanX() results to json files. TODO
  ccrx=resultsList[0]; ccex=resultsList[1]; cdlm=resultsList[2]; cflm=resultsList[3]
  with open(filePrefix+'-CompiledFitReducedChiSq.txt','wb') as outputFile: #,'w+') as outputFile:#?
    outputFile.write(json.dumps(ccrx.tolist()).encode("utf-8"));
    outputFile.close()
  with open(filePrefix+'-CompiledCenterEsts.txt','wb') as outputFile:
    outputFile.write(json.dumps(ccex.tolist()).encode("utf-8"));
    outputFile.close()
  with open(filePrefix+'-CompiledDataLocalMaxima.txt','wb') as outputFile:
    outputFile.write(json.dumps(cdlm.tolist()).encode("utf-8"));
    outputFile.close()
  with open(filePrefix+'-CompiledFitLocalMaxima.txt','wb') as outputFile:
    outputFile.write(json.dumps(cflm.tolist()).encode("utf-8"));
    outputFile.close()

def MultiBinSpreadPlotter(mass, scan, compRay, resList, quantity="Center Parameter", currDir='./'):
  finScanEsts = scanFitsFinalEsts(compRay)
  fig = plt.figure()
  plt.clf()
  fig.set_size_inches(20, 12)
  colorCoding=['red','orange','yellow','green','blue','purple']
  if len(compRay.shape)==3:    yVals = np.arange(len(compRay[:,0,0]));    nPeaks = len(compRay[0,:,0])
  elif len(compRay.shape)==4:  yVals = np.arange(len(compRay[:,0,0,0]));  nPeaks = len(compRay[0,0,:,0])

  for p in range(nPeaks):
    plt.gca().add_patch(Rectangle((finScanEsts[p,0]-3*finScanEsts[p,1], yVals[0]-1), 6*finScanEsts[p,1], yVals[-1]+1, color=colorCoding[p], alpha=.2))
    plt.gca().add_patch(Rectangle((finScanEsts[p,0]-2*finScanEsts[p,1], yVals[0]-1), 4*finScanEsts[p,1], yVals[-1]+1, color=colorCoding[p], alpha=.2))
    plt.gca().add_patch(Rectangle((finScanEsts[p,0]-1*finScanEsts[p,1], yVals[0]-1), 2*finScanEsts[p,1], yVals[-1]+1, color=colorCoding[p], alpha=.2))
    plt.axvline(finScanEsts[p,0], ymin=0, ymax=1, color=colorCoding[p], linestyle='--')

    if len(compRay.shape)==3:
      centerVals = np.ma.array(compRay[:,p,0], mask=np.isnan(compRay[:,p,0])); errVals = np.ma.array(compRay[:,p,1], mask=np.isnan(compRay[:,p,0]));
      plt.errorbar(centerVals, yVals, xerr=errVals, fmt='o', color=colorCoding[p], markeredgecolor='black', markersize=7, label="Peak %d"%(p+1))

    elif len(compRay.shape)==4:
      for k in range(len(compRay[0,:,0,0])):
        centerVals = np.ma.array(compRay[:,k,p,0], mask=np.isnan(compRay[:,k,p,0])); errVals = np.ma.array(compRay[:,k,p,1], mask=np.isnan(compRay[:,k,p,0]));
        if k==0:
          plt.errorbar(centerVals, yVals, xerr=errVals, fmt='o', color=colorCoding[p], markeredgecolor='black', markersize=7, label="Peak %d"%(p+1))
        else:
          plt.errorbar(centerVals, yVals, xerr=errVals, fmt='o', color=colorCoding[p], markeredgecolor='black', markersize=7)

  plt.title(r'Estimated %s vs. Resolution Setting For $^{%d}$Ra$^{19}$F Scan %s'%(quantity, mass-19,scan), fontsize=18)
  plt.xlabel(r'$\nu (cm)^{-1}$', fontsize=16)
  plt.ylabel("Resolution Setting", fontsize=16)
  plt.yticks(ticks=yVals,labels=resList)
  plt.xticks(ticks=finScanEsts[:,0], labels=[r'$%.3f \pm %.3f$'%(finScanEsts[i,0],finScanEsts[i,1]) for i in range(len(finScanEsts[:,0]))])
  plt.ylim([-.5, yVals[-1]+.5])
  plt.xlim(plt.gca().get_xlim())
  #box = plt.gca().get_position()
  #plt.gca().set_position([box.x0, box.y0, box.width * 0.95, box.height])
  lgd=plt.legend(loc="center right", bbox_to_anchor=(1.05,.5), fontsize=12)
  #vLinePlotter(vlineArrayPI12, r'$^2\Sigma^{+}_{1/2} \rightarrow ^2\Pi_{1/2}$')
  #vLinePlotter(vlineArrayDELTA32, r'$^2\Delta_{3/2}$')
  #vLinePlotter(vlineArrayDELTA52, r'$^2\Delta_{5/2}$')

  if currDir=='./': currDir = './FitResults/mass%dFits/Scan%sFits/'%(mass,scan)
  if not os.path.exists(currDir): os.makedirs(currDir)
  plt.savefig(currDir+'Mass%dScan%s_%dPeaks_EstimateSpread-%s.png'%(mass,scan,nPeaks,quantity), bbox_extra_artists=(lgd,), bbox_inches='tight')
  if not os.path.exists('./FitResults/RebinDependencePlots/'): os.makedirs('./FitResults/RebinDependencePlots/')
  plt.savefig('./FitResults/RebinDependencePlots/Mass%dScan%s_%dPeaks_EstimateSpread-%s.png'%(mass,scan,nPeaks,quantity), bbox_extra_artists=(lgd,), bbox_inches='tight')
  plt.close()

def weightedStatistics(dataVals, dataSigs, transitionLabel="'this'", verbose=False):
  if len(dataVals)==1:
    print("yo... There was only one data point. What are you averaging, dawg?")
    return((dataVals[0], dataSigs[0]))
  dataVals = np.ma.array(dataVals, mask=np.isnan(dataVals)); dataSigs = np.ma.array(dataSigs, mask=np.isnan(dataVals)) #masking arrays to remove NaNs in case of fits to reduced number of peaks
  #dataVals = np.ma.array(dataVals, mask=np.isnan(dataSigs)); dataSigs = np.ma.array(dataSigs, mask=np.isnan(dataSigs)) #Also masking arrays on NaN vals in data Sigs, just in case?
  dataVarians = np.square(dataSigs)
  weightedMean = (1/np.sum(1/dataVarians))*np.sum(dataVals/dataVarians)
  statistVar = 1/np.sum(1/dataVarians)
  naiveScatterVar = np.var(dataVals)/len(dataVals)
  '''
  scatterVar = (1/(len(dataVals)-1))*np.sum( np.square(dataVals - weightedMean)/dataVarians )*statistVar
  print("statistVar = ",statistVar)
  print("scatterVar = ",scatterVar)
  if statistVar>scatterVar:
    if verbose: print("Statistical variance larger than scattering variance for "+transitionLabel+" transition. Will use statistVar to report final uncertainty.")
    weightedError=math.sqrt(statistVar)
  elif statistVar<scatterVar:
    if verbose: print("Scatter variance larger than statistical variance for "+transitionLabel+" transition. Will use scatterVar to report final uncertainty.")
    weightedError=math.sqrt(scatterVar)
  else:
    if verbose: print("wtf, what are the odds!? statistVar==scatterVar = ", statistVar==scatterVar)
    weightedError=math.sqrt(statistVar)
  return((weightedMean, weightedError))
  '''
  normalization=statistVar
  #totVar = (normalization**2)*np.sum( (np.square(weightedMean-dataVals) + dataVarians) / np.square(dataVarians) )
  #if verbose: print("statistVar = ", statistVar, "\nnaiveScatterVar = ", naiveScatterVar, "\npotentially totVar = ", totVar)
  totVar = (normalization)*np.sum( (np.square(weightedMean-dataVals)/(len(dataVals)-1) + dataVarians) / dataVarians )
  weightScatVar = (normalization)*np.sum( np.square(weightedMean-dataVals)  / dataVarians )/len(dataVals)
  if verbose:
    print("\nnaiveScatterVar = ", naiveScatterVar, "  statistVar = ", statistVar, "\nweightScatVar = ", weightScatVar, "  potentially totVar = ", totVar)
    print("weightedMean = ", weightedMean, "  std = ", math.sqrt(totVar))
  return((weightedMean, math.sqrt(totVar)))

def scanFitsFinalEsts(compRay):
  if len(compRay.shape)==3:
    #print("test: len = 3")
    finalScanEstimates = np.c_[np.nanmean(compRay[:,:,0], axis=0), np.nanstd(compRay[:,:,0], axis=0), np.nanmax(compRay[:,:,0], axis=0)-np.nanmin(compRay[:,:,0], axis=0)]
    #print("finalScanEstimates.shape=",finalScanEstimates.shape,"\nfinalScanEstimates:\n",finalScanEstimates)
    #finalScanEsts = np.copy(finalScanEstimates)
    for p in range(len(compRay[0,:,0])):
      weightStats = weightedStatistics(compRay[:,p,0], compRay[:,p,1], verbose=(p==0))
      #print("test: weightStats = ", weightStats)
      finalScanEstimates[p,0] = weightStats[0]; finalScanEstimates[p,1] = weightStats[1] #weighted stats over resolution setting, for each individual peak
    #print("finalScanEstimates:\n", finalScanEstimates)
    return(finalScanEstimates)

  elif len(compRay.shape)==4:
    #print("test: len = 4")
    '''nSamps = len(compRay[0,:,0,0])
    R = len(compRay[:,0,0,0])
    intermediateScanEstimates = np.empty_like(compRay[:,0,:,:])
    intermediateScanEstimates[:,:,0] = np.mean(compRay[:,:,:,0], axis=1)
    intermediateScanEstimates[:,:,1] = np.std(compRay[:,:,:,0], axis=1)
    #print("intermediateScanEstimates.shape=",intermediateScanEstimates.shape,"\nfinalScanEstimates:\n",intermediateScanEstimates)
    for r in range(R):
      for p in range(len(compRay[0,0,:,0])):
        weightStats = weightedStatistics(compRay[r,:,p,0], compRay[r,:,p,1]) #weighted stats over random subsamples, for each individual resolution setting (I hope)
        #print("test: weightStats = ", weightStats)
        #print("intermediateScanEstimates[r,p,0] = ",intermediateScanEstimates[r,p,0])
        intermediateScanEstimates[r,p,0] = weightStats[0]; intermediateScanEstimates[r,p,1] = weightStats[1]'''
    #print("compRay dims:\n", compRay.shape)
    resh=np.reshape(compRay,(-1, compRay.shape[-2],compRay.shape[-1]))
    #print("reshaped dims:\n",resh.shape)
    return(scanFitsFinalEsts(resh))

def Scanalyzer(mass, s, peaksList=[13285,13278.8,13272.8,13266.57], peakSigmas=np.array([]), initGamma=1,resList=[.01,.02,.05,.1,.2,.5], method='leastsq',similarSigma=True, makePlots=True, sameSkew=True, sameSigma=True, useWeights=True, ltrim=-1, rtrim=-1, skewList=np.array([]), skew0='NaN', linearTerm=False, rewrite=False, verbose=False,frac=1, nSamps=1):
  m = mass
  scan = str(s)
  print("Now running Scanalyzer for mass = %d; scan: %s"%(mass, scan))
  '''if type(s) == list: currDir = './FitResults/Mass%dFits/CombinedScanFits/'%mass
  else: currDir = './FitResults/Mass%dFits/Scan%sFits/'%(mass,scan)'''
  currDir = './FitResults/Mass%dFits/Scan%sFits/'%(mass,scan)  
  initCenterEsts=peaksList
  initWidthEsts=peakSigmas
  N = len(initCenterEsts)
  fileCondits = os.path.exists(currDir+'Mass%d_Scan%s_%dPeaks-CompiledCenterEsts.txt'%(mass,scan,N)) and os.path.exists(currDir+'Mass%d_Scan%s_%dPeaks-CompiledFitLocalMaxima.txt'%(mass,scan,N))
  #print("ccex found?",os.path.exists(currDir+'Mass%d_Scan%s_%dPeaks-CompiledCenterEsts.txt'%(mass,scan,N))); print("cflm found?",os.path.exists(currDir+'Mass%d_Scan%s_%dPeaks-CompiledFitLocalMaxima.txt'%(mass,scan,N)))
  if (fileCondits and (rewrite==False)):
    print("Success!")
    with open(currDir+'Mass%d_Scan%s_%dPeaks-CompiledCenterEsts.txt'%(mass,scan,N),'r') as ccexFile:
      ccex = np.array(json.load(ccexFile))
      #print("\nfinCenterEst:")
      finCenterEst = scanFitsFinalEsts(ccex); fce = finCenterEst
    with open(currDir+'Mass%d_Scan%s_%dPeaks-CompiledFitLocalMaxima.txt'%(mass,scan,N),'r') as cflmFile:
      cflm = np.array(json.load(cflmFile))
      #print("\nfinFitLocMaxEst:")
      finFitLocMaxEst = scanFitsFinalEsts(cflm); ffl = finFitLocMaxEst
    if os.path.exists(currDir+'Mass%d_Scan%s_%dPeaks-CompiledDataLocalMaxima.txt'%(mass,scan,N)):
      with open(currDir+'Mass%d_Scan%s_%dPeaks-CompiledDataLocalMaxima.txt'%(mass,scan,N),'r') as cdlmFile:
        cdlm = np.array(json.load(cdlmFile))
        #print("\nfinDatLocMaxEst:")
        finDatLocMaxEst = scanFitsFinalEsts(cdlm); fdl = finDatLocMaxEst
    else: #In theory I should just recompute cdlm from findLocalMaxima() without redoing fits, but this should never be relevant ever unless I moved only the cdlm file like some sort of fuccboi...
      cdlm = -1*np.ones_like(cflm)
      finDatLocMaxEst = -1*np.ones_like(finFitLocMaxEst)
    if makePlots:
      MultiBinSpreadPlotter(mass, s, ccex, resList, currDir=currDir)
      MultiBinSpreadPlotter(mass, s, cdlm, resList, currDir=currDir, quantity="local maxima from data")
      MultiBinSpreadPlotter(mass, s, cflm, resList, currDir=currDir, quantity="local maxima from fits")

  else:#resList=[.01,.02,.05,.1,.2,.5], method='leastsq',similarSigma=True, makePlots=True,spreadPlot=True, sameSkew=True, useWeights=True, ltrim=-1, rtrim=-1, skewList=np.array([]), skew0='NaN'): 
    (cfrx,ccex, cdlm, cflm) = fitScanX(m,s,peaksList,peakSigmas=peakSigmas,initGamma=initGamma,resList=resList,method=method,makePlots=makePlots,useWeights=useWeights,sameSkew=sameSkew,skew0=skew0,skewList=skewList,ltrim=ltrim,rtrim=rtrim,sameSigma=sameSigma,similarSigma=similarSigma,linearTerm=linearTerm, frac=frac, nSamps=nSamps)
    #(cfrx,ccex, cdlm, cflm) = fitScanX(m,s,peaksList,peakSigmas=peakSigmas,initGamma=initGamma,resList=resList,method=method,makePlots=makePlots,useWeights=useWeights,sameSkew=sameSkew,skew0=skew0,skewList=skewList,ltrim=ltrim,rtrim=rtrim,sameSigma=sameSigma,similarSigma=similarSigma)
    finCenterEst = scanFitsFinalEsts(ccex); fce = finCenterEst
    finDatLocMaxEst = scanFitsFinalEsts(cdlm); fdl = finDatLocMaxEst
    finFitLocMaxEst = scanFitsFinalEsts(cflm); ffl = finFitLocMaxEst
  if verbose: print("Final Center Estimate:\n%s\nFinal Local Max Ests from fits:\n%s\nFinal Local Max Ests from data:\n%s\n"%(str(fce),str(fdl),str(ffl)))
  if not os.path.exists('./FitResults/OutputFiles/mass%d/'%(m)): os.makedirs('./FitResults/OutputFiles/mass%d/'%(m))
  xFile=open("./FitResults/OutputFiles/Mass%d/Mass%dScan%sCompiledFitCenterEstimates.txt"%(m,m,s),'w+')
  xFile.write("#Compiled Fit Center Estimates:\n%s\n\n#Scanalyzer Final Estimates:\n%s\n#1Sigma:\n%s\n#Range:\n%s"%(str(ccex), str(fce[:,0]), str(fce[:,1]), str(fce[:,2])))
  xFile.close()
  xFile=open("./FitResults/OutputFiles/Mass%d/Mass%dScan%sCompiledDatLocMaxEstimates.txt"%(m,m,s),'w+')
  xFile.write("#Compiled Data Local Max Estimates:\n%s\n\n#Scanalyzer Final Estimates:\n%s\n#1Sigma:\n%s\n#Range:\n%s"%(str(cdlm), str(fdl[:,0]), str(fdl[:,1]), str(fdl[:,2])))
  xFile.close()
  xFile=open("./FitResults/OutputFiles/Mass%d/Mass%dScan%sCompiledFitLocMaxEstimates.txt"%(m,m,s),'w+')
  xFile.write("#Compiled Fit Local Max Estimates:\n%s\n\n#Scanalyzer Final Estimates:\n%s\n#1Sigma:\n%s\n#Range:\n%s"%(str(cflm), str(ffl[:,0]), str(ffl[:,1]), str(ffl[:,2])))
  xFile.close()
  return({'fce':fce,'fdl':fdl,'ffl':ffl})

if __name__ == '__main__':
  pd.options.mode.chained_assignment = None  # default='warn' (Pandas keep harassing me and I'm doing nothing wrong!)
  rewrite=True; ltrim=13256.5; rtrim=13287; sameSigma=True
  massList=np.array([242,244])#np.array([242,243,244,245, 247])
  allScansBigDic = {}
  for m in massList: allScansBigDic[m] = lmd.makeScanToWavemeterDic(m, redo=False, verbose=False)
  colorDict={242:'red', 243:'orange',244:'green',245:'blue',247:'purple'}
  massScanDic={}
  massScanDic[242]=[[2312, 2313]] #These are good individually and combined!
  massScanDic[243]=[[2302,2303,2308]]#,2283]#2300? #2283(from a different time, when signals weren't as strong. use in future), 2301("wavemeter stopped working in the middle of a peak" + relatively weak signal)
  massScanDic[244]=[[2304,2305,2306]]#,2307] #possibly remove 2307?
  massScanDic[245]=[[2309,2310,2320]]#,[2341,2349],2350]#, [2341(75mW),2349],2350("re-tuned TiSa overlap upstairs --> additional 50% improvement")]#,[2346] is a pdl scan though. gross...#2178 is Ti:Sa, but actually gross af#
  massScanDic[247]=[[2311,2322]]#,[2188,2190]]#(also decent, but from diff era with different rates)
  initCenterEsts={}
  initCenterEsts[242]=[13285,13278.86,13272.76,13266.76]#,13260.35]
  initCenterEsts[243]=[13284.95,13278.80,13272.61,13266.61]#,13260.35]
  initCenterEsts[244]=[13284.85,13278.70,13272.66,13266.45]#,13260.35]
  initCenterEsts[245]=[13284.73,13278.60,13272.46,13266.48]#,13260.35]
  initCenterEsts[247]=[13284.54,13278.41,13272.24,13266.05]#,13260.35]
  sigmaEst=.5; gammaEst=2; skew0=-3
  resolutionList=[.03,.05,.07,.1,.2,.3] # [.01,.02,.03,.05,.07,.1,.2,.3]# struggling a bit with low count statistics for ^{224,225}RaF
  for m in massList:
    scanListsList = massScanDic[m]
    peaksList = initCenterEsts[m]
    print("Test: peaksList:\n",peaksList)
    for s in scanListsList:
      print("m=%d, scans:%s, peaksList:%s"%(m,str(s),str(peaksList)))
      finalfitResults = Scanalyzer(m,s,rewrite=rewrite,peaksList=peaksList,peakSigmas=sigmaEst, sameSigma=sameSigma,initGamma=gammaEst,resList=resolutionList, ltrim=ltrim, rtrim=rtrim, makePlots=True,sameSkew=True, linearTerm=False, useWeights=True, skew0=skew0, frac=0.45, nSamps=20)
      (fce,fdl,ffl)=(finalfitResults['fce'],finalfitResults['fdl'],finalfitResults['ffl'])
    print("\n\n")