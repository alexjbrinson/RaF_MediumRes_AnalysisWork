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
from lmfit.models import SkewedVoigtModel, LinearModel, GaussianModel, LorentzianModel
import emcee
#import csv
import time
import numdifftools
import LoadingAndMungingData as lmd
import FrequencyConvertedUtilityKit as FCUK

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
  plt.errorbar(xDat, yDat, yerr=ySigDat, fmt='b.-',ecolor='k', alpha=.5)
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
  plt.close()


def fitNPeaks(datFrame, peaksList, peakSigmas=np.array([]), method='leastsq', useWeights=True, sameSkew=True, skewList=np.array([]), skew0="NaN", initGamma=1,sameSigma=True, similarSigma=True):
#fit scan data with rebinning to spectrum with pre-guessed peaks
  # TODO incorporate similarSigma idea!
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
  #print("test: skewList:",skewList,"\npeaksList:",peaksList)
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
    #print("k=%.2f"%k)
    ind1 = np.argmin(abs(xDat-k))
    #print("ind1=%d"%ind1)
    ind2 = np.argmax(yDat[ind1-2:ind1+3])+ind1-2
    estimHeight = yDat[ind2] - bg
    #print("testing estims... bg=%d, i=%d, k=%.2f, ind1=%d, ind2=%d, xDat[ind2]=%.2f, yDat[ind2=]%.2f, estimHeight=%.2f"%(bg, i,k,ind1,ind2,xDat[ind2],yDat[ind2],estimHeight))
    if ind1 <= 3 or len(xDat)-ind1<=3:
      print("WARNING: Peak occurs too closely to edge of dataset. A lower rebin setting is recommended.")
      warningStatus=-1
      return(False, warningStatus)
    if k-.66>xDat[8]:
      k2 = peaksList[i] -.66; #pea8ksList[i], may be the "center", but it might not be where distribution is maximized. -.66 cm^{-1} is the shift for gamma=1.5,sigma=.8,skew=-2
      ind1 = np.argmin(abs(xDat-k2))
      ind2 = np.argmax(yDat[ind1-7:ind1+8])+ind1-7
      estimHeight2 = yDat[ind2] - bg
      if estimHeight2>estimHeight: print("aha! Had to look left to find global maximum!")
      estimHeight = max(estimHeight,estimHeight2)
    amp=estimHeight*(peakSigmas[i]*math.sqrt(2*math.pi))/special.wofz((1j*initGamma)/(peakSigmas[i]*math.sqrt(2))).real
    height1=amp*special.wofz((1j*initGamma)/(peakSigmas[i]*math.sqrt(2))).real/(peakSigmas[i]*math.sqrt(2*math.pi))
    #print("testing math stuff... amp=%.2f; height=%.2f"%(amp,height1))
    svmod = SkewedVoigtModel(prefix="sv"+str(i)+"_")
    svmod.set_param_hint('center', value=peaksList[i], min=max(peaksList[i]-2*peakSigmas[i], xDat[3]), max=min(peaksList[i]+2*peakSigmas[i],xDat[-3]))
    if sameSigma:
      if i == 0: params['sv'+str(i)+'_sigma']= Parameter(value=peakSigmas[0], min=0, max = 3*peakSigmas[0], vary=True)
      elif i>0:
        params['sv'+str(i)+'_sigma'] = Parameter(expr='sv0_sigma')
    else: svmod.set_param_hint('sigma', value=peakSigmas[i], min=0.1, max=2*peakSigmas[i])
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

def findLocalMax(datFrame, guess, searchWidth, xcol="wavenumber_mean",ycol="signal_value", uncertIndex=3, verbose=False):
  cropDat=lmd.trimRange(datFrame.copy(),ltrim=guess-searchWidth,rtrim=guess+searchWidth).loc[:,[xcol,ycol]].sort_values(by=ycol,ascending=False)
  xValsByY = np.array(cropDat.loc[:,xcol]); yValsByY = np.array(cropDat.loc[:,ycol])
  if len(xValsByY)<uncertIndex+1:
    uncertIndex=len(xValsByY)-1
  if verbose: print("uncertIndex was larger than length of trimmed array, had to reduce to uncertIndex=%d"%uncertIndex)
  #print("findLocalMaxTests. cropDat:\n",cropDat.head(),"\n, xValsByY:%s\nyValsByY:%s"%(xValsByY,yValsByY))
  return(np.array([[xValsByY[0],abs(xValsByY[0]-xValsByY[uncertIndex])],[yValsByY[0],yValsByY[0]-yValsByY[uncertIndex]]]))

def fitScanX(mass, s, peaksList, peakSigmas=np.array([]), initGamma=1,resList=[.01,.02,.05,.1,.2,.5], method='leastsq',similarSigma=True, makePlots=True,spreadPlot=True, sameSkew=True, sameSigma=True, useWeights=True, ltrim=-1, rtrim=-1, skewList=np.array([]), skew0='NaN'):
  #fit scan x data with rebinning R to spectrum with N peaks
  scan = str(s)
  print("now running fitScanX for mass = %d; scan: %s"%(mass, scan))
  if type(s) == list:
    print("ooh boy! We're doing some combo fits today, buddy!")
    scanFrame = lmd.mergeDatRaw(mass, s)
    currDir = './FitResults/Mass%dFits/CombinedScanFits/'%mass
  else:
    scanFrame = lmd.rawDatPrep(mass,s)
    currDir = './FitResults/Mass%dFits/Scan%sFits/'%(mass,scan)
  if not os.path.exists(currDir): os.makedirs(currDir)
  if len(peakSigmas) == 0:
    peakSigmas = 1*np.ones_like(peaksList)
  else:
    assert(len(peakSigmas) == len(peaksList))
    assert(np.all(peakSigmas>0))
  assert(len(resList)>0)
  R = len(resList)
  minR=1
  N = len(peaksList)
  resList = np.sort(resList)

  compiledGoFs = -1*np.ones(R-minR+1)
  compiledFitResults = {}
  compiledCenterEsts = -1*np.ones([R, N, 2])
  compiledDataLocalMax = -1*np.ones([R, N, 2])
  compiledFitLocalMax = -1*np.ones([R, N, 2])
  compiledResults = pd.DataFrame(columns=["GoF", 'fitResults', 'PeakEsts', 'dataLocalMax', 'fitLocalMax'], index=resList)#,type=['float',])
  """Above this line is just prepwork"""
  for i in range(len(resList)):
    r=resList[i]
    print("resolution=", r)
    datFrame = lmd.makeUseable(scanFrame, resolution=r, cropSparseEnds=True, noNaNsense=True, ltrim=ltrim, rtrim=rtrim)#TODO allow for range cropping
    (fitRes, warningStatus) = fitNPeaks(datFrame, peaksList, peakSigmas=peakSigmas, initGamma=initGamma, useWeights=useWeights, sameSkew=sameSkew, sameSigma=sameSigma, similarSigma=similarSigma)#add other opts?
    if warningStatus == -1:
      print("That's it for this scan, boys. Don't. push. these peaks. They're. close. to. the. eeeedge. (One of the peaks is leaking out of the scan window at rebin setting%d)"%r)
      (ccrx,ccex,cdlm,cflm)=(compiledGoFs[:i], compiledCenterEsts[:i], compiledDataLocalMax[:i],compiledFitLocalMax[:i])
      with open(currDir+'Mass%d_Scan%s_%dPeaks-CompiledFitReducedChiSq.txt'%(mass,scan,N),'wb') as outputFile:
        outputFile.write(json.dumps(ccrx.tolist()).encode("utf-8"));
        outputFile.close()
      with open(currDir+'Mass%d_Scan%s_%dPeaks-CompiledCenterEsts.txt'%(mass,scan,N),'wb') as outputFile:
        outputFile.write(json.dumps(ccex.tolist()).encode("utf-8"));
        outputFile.close()
      with open(currDir+'Mass%d_Scan%s_%dPeaks-CompiledDataLocalMaxima.txt'%(mass,scan,N),'wb') as outputFile:
        outputFile.write(json.dumps(cdlm.tolist()).encode("utf-8"));
        outputFile.close()
      with open(currDir+'Mass%d_Scan%s_%dPeaks-CompiledFitLocalMaxima.txt'%(mass,scan,N),'wb') as outputFile:
        outputFile.write(json.dumps(cflm.tolist()).encode("utf-8"));
        outputFile.close()
      return(ccrx,ccex,cdlm,cflm)#return(compiledGoFs[:i], compiledCenterEsts[:i], compiledDataLocalMax[:i],compiledFitLocalMax[:i])
    fitReportFile = open(currDir+'Mass%dScan%s_%dbins_FitReport.txt'%(mass,scan,len(datFrame.index)),'w+')
    fitReportFile.write("Fit Report for: Mass = %d, Scan = %s, Resolution = %.3f\n"%(mass,scan,r)); fitReportFile.write(fitRes.fit_report(min_correl=0.25)); fitReportFile.close()
    compiledFitResults[r] = fitRes.best_values
    #print("test1: type(fitRes.best_values) = ", type(fitRes.best_values) )
    compiledResults.loc[r,"FitResults"] = [fitRes.best_values]
    fitCenterEsts = np.zeros([N,2])
    fitLocalMaxEsts = np.zeros([N,2])
    datLocalMaxEsts = np.zeros([N,2])
    parmCenterNames = ['sv'+str(j)+'_center' for j in range(N)].append(['l0_slope', 'l0_intercept'])
    kwargs = {'p_names':parmCenterNames}
    print("fitRes.errorbars", fitRes.errorbars)
    
    #fitRes.conf_interval(**kwargs)#/Try adding some Try/Except shenannigans?
    for p in range(N):
      mu_p = fitRes.best_values['sv'+str(p)+'_center'];
      sigma_p = fitRes.best_values['sv'+str(p)+'_sigma'];
      gamma_p = fitRes.best_values['sv'+str(p)+'_gamma'];
      skew_p = fitRes.best_values['sv'+str(p)+'_skew'];
      mu_pUncert = fitRes.params['sv'+str(p)+'_center'].stderr if fitRes.errorbars else max(sigma_p, gamma_p)
      xSimp = np.arange(mu_p-(sigma_p+gamma_p),mu_p+(sigma_p+gamma_p),.01); ySimp=skewedVoigt(xSimp,1,mu_p,sigma_p,gamma_p,skew_p)
      fitLocMax_p = xSimp[np.argmax(ySimp)]; #cropDat = trimRange(datFrame.copy(),ltrim=fitLocMax_p-.5,rtrim=fitLocMax_p+.5);
      #xCrop=np.array(cropDat['wavenumber_mean']); yCrop=np.array(cropDat['signal_value']); datLocMax = xCrop[np.argmax(yCrop)]; datLocalMax=xCrop[np.argmax(yCrop)]
      #cropDat = np.array(trimRange(datFrame.copy(),ltrim=fitLocMax_p-.5,rtrim=fitLocMax_p+.5)["wavenumber_mean","signal_value"]);
      '''cropDat=trimRange(datFrame.copy(),ltrim=fitLocMax_p-.5,rtrim=fitLocMax_p+.5)["wavenumber_mean","signal_value"].sort_values(by="signal_value")
      xValsBySig = np.array(cropDat.loc[:,"wavenumber_mean"])'''
      datLocalEst_p = findLocalMax(datFrame, fitLocMax_p, 0.5, uncertIndex=3)
      fitCenterEsts[p, 0] = mu_p; fitCenterEsts[p,1] = mu_pUncert
      fitLocalMaxEsts[p,0] = fitLocMax_p; fitLocalMaxEsts[p,1] = mu_pUncert
      datLocalMaxEsts[p,0] = datLocalEst_p[0,0] ; datLocalMaxEsts[p,1] = datLocalEst_p[0,1]
    if makePlots == True: fitPlottingSubRoutine(datFrame, mass, s, r, N, fitRes, redchi=fitRes.redchi, currDir=currDir)
    compiledCenterEsts[i,:,:] = fitCenterEsts
    compiledDataLocalMax[i,:,:] = datLocalMaxEsts
    compiledFitLocalMax[i,:,:] = fitLocalMaxEsts
    compiledGoFs[i] = fitRes.redchi
    #print("test2: type(fitCenterEsts) = ", type(fitCenterEsts) )
    compiledResults.loc[r,'PeakEsts'] = [fitCenterEsts]
    compiledResults.loc[r,'dataLocalMax'] = [datLocalMaxEsts]
    compiledResults.loc[r,'fitLocalMax'] = [fitLocalMaxEsts]
    compiledResults.loc[r,'GoF'] = fitRes.redchi
    print("Just ran fitScanX routine for: scan %s; res = %.3f; N=%d, and found redchi = %f"%(scan,r,N,fitRes.redchi), "\nfitCenterEsts:", fitCenterEsts[:,0], "\nfitPeakEsts:",datLocalMaxEsts[:,0])    
  ccrx=compiledGoFs; ccex=compiledCenterEsts; cdlm=compiledDataLocalMax; cflm=compiledFitLocalMax
  finCenterEst = scanFitsFinalEsts(ccex)
  finDatLocMaxEst = scanFitsFinalEsts(cdlm)
  finFitLocMaxEst = scanFitsFinalEsts(cflm)
  print("Final Center Estimate:\n%s\nFinal Local Max Ests from fits:\n%s\nFinal Local Max Ests from data:\n%s\n"%(str(finCenterEst),str(finFitLocMaxEst),str(finDatLocMaxEst)))
  #print("report from last fit performed:\n",fitRes.fit_report())
  if not os.path.exists(currDir+'AllFitResultsCompiled/'): os.makedirs(currDir+'AllFitResultsCompiled/')
  compiledResultsFile = open(currDir+'AllFitResultsCompiled/Mass%dScan%s-%dPeaks.txt'%(mass,scan,N),'w+')
  compiledResultsFile.write('Mass: %d ; Scan: %s\nInitial peak center estimates:'%(mass,scan)+str(peaksList)); compiledResultsFile.write('\nInitial peak width estimates:'+ str(peakSigmas));
  compiledResultsFile.write("\nCompilation of Results:\n")
  compiledResultsFile.write("Final Center Estimate:\n%s\nFinal Local Max Ests from fits:\n%s\nFinal Local Max Ests from data:\n%s\n"%(str(finCenterEst),str(finFitLocMaxEst),str(finDatLocMaxEst)));
  with pd.option_context('display.max_rows', 100, 'display.max_columns', 200): compiledResultsFile.write(compiledResults.to_csv());
  compiledResultsFile.close()

  with open(currDir+'Mass%d_Scan%s_%dPeaks-CompiledFitReducedChiSq.txt'%(mass,scan,N),'wb') as outputFile: #,'w+') as outputFile:#?
    outputFile.write(json.dumps(ccrx.tolist()).encode("utf-8"));
    outputFile.close()
  with open(currDir+'Mass%d_Scan%s_%dPeaks-CompiledCenterEsts.txt'%(mass,scan,N),'wb') as outputFile:
    outputFile.write(json.dumps(ccex.tolist()).encode("utf-8"));
    outputFile.close()
  with open(currDir+'Mass%d_Scan%s_%dPeaks-CompiledDataLocalMaxima.txt'%(mass,scan,N),'wb') as outputFile:
    outputFile.write(json.dumps(cdlm.tolist()).encode("utf-8"));
    outputFile.close()
  with open(currDir+'Mass%d_Scan%s_%dPeaks-CompiledFitLocalMaxima.txt'%(mass,scan,N),'wb') as outputFile:
    outputFile.write(json.dumps(cflm.tolist()).encode("utf-8"));
    outputFile.close()
  if spreadPlot==True:
    MultiBinSpreadPlotter(mass, scan, ccex, resList, currDir=currDir)
    MultiBinSpreadPlotter(mass, scan, cdlm, resList, currDir=currDir, quantity="local maxima from data")
    MultiBinSpreadPlotter(mass, scan, cflm, resList, currDir=currDir, quantity="local maxima from fits")
  return(ccrx,ccex,cdlm,cflm) #TODO?: get peak estimates, apply them to next fit. Carry out next fit(s) -> Nahhh

def ExportResults(dir, resultsTuple, resList):
  #exports fitScanX() results to json files. TODO
  pass

def MultiBinSpreadPlotter(mass, scan, compRay, resList, quantity="Center Parameter", currDir='./'):
  finScanEsts = scanFitsFinalEsts(compRay)
  fig = plt.figure()
  plt.clf()
  fig.set_size_inches(20, 12)
  colorCoding=['red','orange','yellow','green','blue','purple']
  yVals = np.arange(len(compRay[:,0,0]))
  for p in range(len(compRay[0,:,0])):
    plt.gca().add_patch(Rectangle((finScanEsts[p,0]-3*finScanEsts[p,1], yVals[0]-1), 6*finScanEsts[p,1], yVals[-1]+1, color=colorCoding[p], alpha=.25))
    plt.axvline(finScanEsts[p,0], ymin=0, ymax=1, color=colorCoding[p], linestyle='--')
    plt.errorbar(compRay[:,p,0], yVals, xerr=compRay[:,p,1], fmt='o', color=colorCoding[p], markeredgecolor='black', markersize=7, label="Peak %d"%(p+1))
    #plt.annotate()#maybe TODO: label final ests with one of these instead?
  plt.title(r'Estimated %s vs. Resolution Setting For $^{%d}$Ra$^{19}$F Scan %s'%(quantity, mass-19,scan), fontsize=18)
  plt.xlabel(r'$\nu (cm)^{-1}$', fontsize=16)
  plt.ylabel("Resolution Setting", fontsize=16)
  plt.yticks(ticks=yVals,labels=resList)
  plt.xticks(ticks=finScanEsts[:,0], labels=['%.3f'%p for p in finScanEsts[:,0]])
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
  plt.savefig(currDir+'Mass%dScan%s_%dPeaks_EstimateSpread-%s.png'%(mass,scan,len(compRay[0,:,0]),quantity), bbox_extra_artists=(lgd,), bbox_inches='tight')
  if not os.path.exists('./FitResults/RebinDependencePlots/'): os.makedirs('./FitResults/RebinDependencePlots/')
  plt.savefig('./FitResults/RebinDependencePlots/Mass%dScan%s_%dPeaks_EstimateSpread-%s.png'%(mass,scan,len(compRay[0,:,0]),quantity), bbox_extra_artists=(lgd,), bbox_inches='tight')
  plt.close()

def weightedStatistics(dataVals, dataSigs, transitionLabel="'this'", verbose=False):
  if len(dataVals)==1:
    print("yo... There was only one data point. What are you averaging, dawg?")
    return((dataVals[0], dataSigs[0]))
  dataVarians = np.square(dataSigs)
  weightedMean = (1/np.sum(1/dataVarians))*np.sum(dataVals/dataVarians)
  statistVar = 1/np.sum(1/dataVarians)
  scatterVar = statistVar*(1/(len(dataVals)-1))*np.sum( np.square(dataVals - weightedMean)/dataVarians )
  if statistVar>scatterVar:
    if verbose: print("Statistical variance larger than scattering variance for "+transitionLabel+" transition. Will use statistVar to report final uncertainty.")
    weightedError=math.sqrt(statistVar)
  elif statistVar<scatterVar:
    if verbose: print("Scatter variance larger than scattering variance for "+transitionLabel+" transition. Will use scatterVar to report final uncertainty.")
    weightedError=math.sqrt(statistVar)
  else:
    if verbose: print("wtf, what are the odds!? statistVar==scatterVar = ", statistVar==scatterVar)
    weightedError=math.sqrt(statistVar)
  return((weightedMean, weightedError))

def scanFitsFinalEsts(compRay):
  finalScanEstimates = np.c_[np.mean(compRay[:,:,0], axis=0), np.std(compRay[:,:,0], axis=0), np.max(compRay[:,:,0], axis=0)-np.min(compRay[:,:,0], axis=0)]
  #print("finalScanEstimates.shape=",finalScanEstimates.shape,"\nfinalScanEstimates:\n",finalScanEstimates)
  #finalScanEsts = np.copy(finalScanEstimates)
  for p in range(len(compRay[0,:,0])):
    weightStats = weightedStatistics(compRay[:,p,0], compRay[:,p,1])
    #print("test: weightStats = ", weightStats)
    finalScanEstimates[p,0] = weightStats[0]; finalScanEstimates[p,1] = weightStats[1]
  return(finalScanEstimates)

def Scanalyzer(mass, s, peakList=[13285,13278.8,13272.8,13266.57], peakSigmas=np.array([]), initGamma=1,resList=[.01,.02,.05,.1,.2,.5], method='leastsq',similarSigma=True, makePlots=True, sameSkew=True, sameSigma=True, useWeights=True, ltrim=-1, rtrim=-1, skewList=np.array([]), skew0='NaN', rewrite=False, verbose=False):
  m = mass
  scan = str(s)
  print("Now running Scanalyzer for mass = %d; scan: %s"%(mass, scan))
  if type(s) == list: currDir = './FitResults/Mass%dFits/CombinedScanFits/'%mass
  else: currDir = './FitResults/Mass%dFits/Scan%sFits/'%(mass,scan)
    
  initCenterEsts=peakList
  initWidthEsts=peakSigmas
  resolutionList=resList
  N = len(initCenterEsts)
  fileCondits = os.path.exists(currDir+'Mass%d_Scan%s_%dPeaks-CompiledCenterEsts.txt'%(mass,scan,N)) and os.path.exists(currDir+'Mass%d_Scan%s_%dPeaks-CompiledFitLocalMaxima.txt'%(mass,scan,N))
  #print("ccex found?",os.path.exists(currDir+'Mass%d_Scan%s_%dPeaks-CompiledCenterEsts.txt'%(mass,scan,N)))
  #print("cflm found?",os.path.exists(currDir+'Mass%d_Scan%s_%dPeaks-CompiledFitLocalMaxima.txt'%(mass,scan,N)))
  if (fileCondits and (rewrite==False)):
    print("Success!")
    with open(currDir+'Mass%d_Scan%s_%dPeaks-CompiledCenterEsts.txt'%(mass,scan,N),'r') as ccexFile:
      ccex = np.array(json.load(ccexFile))
      finCenterEst = scanFitsFinalEsts(ccex); fce = finCenterEst
    with open(currDir+'Mass%d_Scan%s_%dPeaks-CompiledFitLocalMaxima.txt'%(mass,scan,N),'r') as cflmFile:
      cflm = np.array(json.load(cflmFile))
      finFitLocMaxEst = scanFitsFinalEsts(cflm); ffl = finFitLocMaxEst
    if os.path.exists(currDir+'Mass%d_Scan%s_%dPeaks-CompiledDataLocalMaxima.txt'%(mass,scan,N)):
      with open(currDir+'Mass%d_Scan%s_%dPeaks-CompiledDataLocalMaxima.txt'%(mass,scan,N),'r') as cdlmFile:
        cdlm = np.array(json.load(cdlmFile))
        finDatLocMaxEst = scanFitsFinalEsts(cdlm); fdl = finDatLocMaxEst
    else: #In theory I should just recompute cdlm from findLocalMaxima() without redoing fits, but this should never be relevant ever unless I moved only the cdlm file like some sort of fuccboi...
      cdlm = -1*np.ones_like(cflm)
      finDatLocMaxEst = -1*np.ones_like(finFitLocMaxEst)
  else:#resList=[.01,.02,.05,.1,.2,.5], method='leastsq',similarSigma=True, makePlots=True,spreadPlot=True, sameSkew=True, useWeights=True, ltrim=-1, rtrim=-1, skewList=np.array([]), skew0='NaN'): 
    (cfrx,ccex, cdlm, cflm) = fitScanX(m,s,peakList,peakSigmas=peakSigmas,initGamma=initGamma,resList=resList,method=method,makePlots=makePlots,useWeights=useWeights,sameSkew=sameSkew,skew0=skew0,skewList=skewList,ltrim=ltrim,rtrim=rtrim,sameSigma=sameSigma,similarSigma=similarSigma)
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
  if makePlots==True:
    MultiBinSpreadPlotter(mass, scan, ccex, resList, currDir=currDir)
    MultiBinSpreadPlotter(mass, scan, cdlm, resList, currDir=currDir, quantity="local maxima from data")
    MultiBinSpreadPlotter(mass, scan, cflm, resList, currDir=currDir, quantity="local maxima from fits")
  return({'fce':fce,'fdl':fdl,'ffl':ffl})

if __name__ == '__main__':
  pd.options.mode.chained_assignment = None  # default='warn'
  rewrite=False
  ltrim=13256.5; rtrim=13287
  sameSigma=True
  idx = pd.IndexSlice
  massList=np.array([242,243,244,245, 247])
  allScansBigDic = {}
  for m in massList: allScansBigDic[m] = lmd.makeScanToWavemeterDic(m, redo=False, verbose=False)
  colorDict={242:'red', 243:'orange',244:'green',245:'blue',247:'purple'}
  massScanDic={}
  massScanDic[242]=[2312, 2313] #These are good individually and combined!
  massScanDic[243]=[2302,2303,2308]#2300? #2283(from a different time, when signals weren't as strong. use in future), 2301("wavemeter stopped working in the middle of a peak" + relatively weak signal)
  massScanDic[244]=[2304,2305,2306]#,2307] #possibly remove 2307?
  massScanDic[245]=[2309,2310,2320]#, [2341(75mW),2349],2350("re-tuned TiSa overlap upstairs --> additional 50% improvement")]#,[2346] is a pdl scan though. gross...#2178 is Ti:Sa, but actually gross af#
  massScanDic[247]=[2311,2322]#,[2188,2190](also decent, but from diff era with different rates)
  initCenterEsts=[13284.75,13278.62,13272.50,13266.5]#,13260.35]
  transitionLabels = np.array(["%d->%d"%(i,i) for i in range(len(initCenterEsts))])
  estimationMethodLabels = np.array(['SkewedMu','LocMaxDat','LocMaxFit'])
  estimationStatisticsLabels = np.array(['mean','stderr','range'])
  sigmaEst=.55
  gammaEst=1.77
  skew0=-2
  resolutionList=[.01,.02,.05,.1,.2,.5]
  isotopeDataFrame = pd.DataFrame(index=massList, columns = pd.MultiIndex.from_product([transitionLabels,estimationMethodLabels, estimationStatisticsLabels], names=['Transitions','Methods','Stats']) )
  isotopeDataFrame=isotopeDataFrame.sort_index()
  #isotopeDataFrame.loc[242,('0->0','SkewedMu')]
  """FinalDataCompileArray = np.array()
  fig_fce = plt.figure("Final SkewedMu")
  fig_fce.title("Comparing Final SkewedMu For Different RaF Isotopes")
  fig_fdl = plt.figure("Final LocMaxDat Ests")
  fig_ffl = plt.figure("Final LocMaxFit Ests")
  figDiffs = plt.figure("figDiffs")"""
  for m in massList:
    s=massScanDic[m]
    peakList=initCenterEsts
    finalfitResults = Scanalyzer(m,s,rewrite=rewrite,peakList=peakList,peakSigmas=sigmaEst*np.ones_like(peakList),sameSigma=sameSigma,initGamma=gammaEst,resList=resolutionList, ltrim=ltrim, rtrim=rtrim, makePlots=True,sameSkew=True, useWeights=True, skew0=-4)
    (fce,fdl,ffl)=(finalfitResults['fce'],finalfitResults['fdl'],finalfitResults['ffl'])
    for p in range(len(peakList)):
      isotopeDataFrame.loc[m,(transitionLabels[p],'SkewedMu')]=fce[p,:]
      isotopeDataFrame.loc[m,(transitionLabels[p],'LocMaxDat')]=fdl[p,:]
      isotopeDataFrame.loc[m,(transitionLabels[p],'LocMaxFit')]=ffl[p,:]
    #with pd.option_context('display.max_rows', 100, 'display.max_columns', 60): print("newest isotope entry in dataframe:\n",isotopeDataFrame.loc[m,(idx[:,:,'mean'])])
    """for s in massScanDic[m]:
      print("m=%d, s=%d")
      if (s in [2312,2313,2283]): peakList = initCenterEsts
      elif (s in [2301,2302,2303]): peakList = [13284.7,13278.2,13272.8]
      elif s==2308: peakList=[13272.5,13266.6]
      elif (s in [2304,2305,2306]): peakList = initCenterEsts
      elif s == 2307: peakList = [13272.5,13266.67,13261]
      elif (s in [2188,2190]): [13284.7,13278.2,13272.8]
      elif s==2311: peakList = initCenterEsts
      else: peakList=initCenterEsts
      FCUK.Scanalyzer(m,s,rewrite=True,peakList=peakList,peakSigmas=sigmaEst*np.ones_like(peakList),initGamma=gammaEst,resList=resolutionList,method="leastsq", fitPlots=True, binSpreadPlot=True, sameSkew=True, useWeights=True, skew0=-4)"""
    
  with pd.option_context('display.max_rows', 100, 'display.max_columns', 100):print("ayy, are we done? isotopeDataFrame:\n",isotopeDataFrame)
  

  if not os.path.exists('./FitResults/OutputFiles/'): os.makedirs('./FitResults/OutputFiles/')
  isotopeDataFrame.to_csv(path_or_buf='./FitResults/OutputFiles/IsotopeDataFrame.csv')
  
  for m in massList:
    color = next(plt.gca()._get_lines.prop_cycler)['color']
    for i in range(len(estimationMethodLabels)):
      if i==0: plt.errorbar(x=isotopeDataFrame.loc[m,idx[:,estimationMethodLabels[i],'mean']],y=i*np.ones_like(initCenterEsts), xerr=isotopeDataFrame.loc[m,idx[:,estimationMethodLabels[i],'range']],fmt="o",label=r'$^{%d}Ra^{19}F$'%(m-19), color=color)
      else: plt.errorbar(x=isotopeDataFrame.loc[m,idx[:,estimationMethodLabels[i],'mean']],y=i*np.ones_like(initCenterEsts), xerr=isotopeDataFrame.loc[m,idx[:,estimationMethodLabels[i],'range']],fmt="o", color=color)
  plt.figure('IsotopeDataFrame in plot form')
  plt.gcf().set_size_inches(20, 12)
  plt.title("Investigating Transition Frequency dependence on isotope number in RaF\n(3 methods for peak identification considered)")
  plt.xlabel(r'Wavenumber $(cm^{-1})$')
  plt.yticks(ticks=range(len(estimationMethodLabels)),labels=estimationMethodLabels)
  plt.legend(loc='best')
  plt.savefig('./FitResults/IsotopeDataFrameSummaryPlot')
  plt.close()

  """Now converting to shift data:"""
  referenceFrame = isotopeDataFrame.loc[245,idx[:,:,['mean','range']]].copy()#hehe "reference frame". Was not intentional lol
  isoShiftsFrame=pd.DataFrame(index=massList, columns = pd.MultiIndex.from_product([transitionLabels,estimationMethodLabels, ['shift','error']], names=['Shifts','Methods','Stats']) )
  for m in massList:
    isoShiftsFrame.loc[m,idx[:,:,'shift']]=(isotopeDataFrame.loc[m,idx[:,:,'mean']] - referenceFrame.loc[idx[:,:,"mean"]]).values
    errorSquaror = np.square(isotopeDataFrame.loc[m,idx[:,:,'range']]) + np.square(referenceFrame.loc[idx[:,:,"range"]])
    isoShiftsFrame.loc[m,idx[:,:,'error']]=np.sqrt(errorSquaror.astype(np.float64)).values
  with pd.option_context('display.max_rows', 100, 'display.max_columns', 30):print("Isotope Shifts:\n",isoShiftsFrame)
  
  """shift vs mass plots to compare methods"""
  for i in range(len(transitionLabels)):
    transition=transitionLabels[i]
    plt.figure(transition)
    plt.gcf().set_size_inches(20, 12)
    plt.title("Isotope Shift vs. Mass for the %s Transition in RaF\n(3 methods for peak identification considered)"%transition)
    for meth in [estimationMethodLabels[0],estimationMethodLabels[2]]:
      color = next(plt.gca()._get_lines.prop_cycler)['color']
      plt.errorbar(x=(massList-245), y=isoShiftsFrame.loc[:,idx[transition,meth,'shift']], yerr=isoShiftsFrame.loc[:,idx[transition,meth,'error']],fmt="o",label=meth, color=color, markersize=8)
    plt.title("Investigating Transition Frequency dependence on isotope number in RaF\n(3 methods for peak identification considered)")
    plt.xlabel('mass diffs (amu)')
    plt.ylabel(r'shift $(cm^{-1})$')
    plt.legend(loc='best')
    plt.savefig('./FitResults/IsotopeShiftDiffMethods_%s-%s.png'%(i,i))
    plt.close()

  """shift vs mass plots to compare transitions"""
  for meth in estimationMethodLabels:
    plt.figure(meth)
    plt.gcf().set_size_inches(20, 12)
    plt.title("Isotope Shift vs. Mass for %s Estimation method in RaF\n(4 transitions included)"%meth)
    for transition in transitionLabels:
      color = next(plt.gca()._get_lines.prop_cycler)['color']
      plt.errorbar(x=(massList-245), y=isoShiftsFrame.loc[:,idx[transition,meth,'shift']], yerr=isoShiftsFrame.loc[:,idx[transition,meth,'error']],fmt="o",label=transition, color=color, markersize=8)
    plt.title("Investigating Transition Frequency dependence on isotope number in RaF\n(3 methods for peak identification considered)")
    plt.xlabel('mass diffs (amu)')
    plt.ylabel(r'shift $(cm^{-1})$')
    plt.legend(loc='best')
    plt.savefig('./FitResults/IsotopeShiftDiffTransitions_%s.png'%meth)
    plt.close()    
  #TODO: Isotope summary plots. Take differences
'''4. "Make a table of "isotope shifts", comparing differences between different isotopes and using the same electronic transition"'''
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import LoadingAndMungingData as lmd
import newTask4_IsotopeShiftsLetsGoooo as FCUK
pd.options.mode.chained_assignment = None  # default='warn'
rewrite=False
ltrim=13256.5; rtrim=13287
sameSigma=True
idx = pd.IndexSlice
allScansBigDic = {}
for m in [241,242,243,244,245,247]: allScansBigDic[m] = lmd.makeScanToWavemeterDic(m, redo=False, verbose=False)
massList=[242,243,244,245, 247]
colorDict={242:'red', 243:'orange',244:'green',245:'blue',247:'purple'}
massScanDic={}
massScanDic[242]=[2312, 2313] #These are good individually and combined!
massScanDic[243]=[2302,2303,2308]#2300? #2283(from a different time, when signals weren't as strong. use in future), 2301("wavemeter stopped working in the middle of a peak" + relatively weak signal)
massScanDic[244]=[2304,2305,2306]#,2307] #possibly remove 2307?
massScanDic[245]=[2309,2310,2320]#, [2341(75mW),2349],2350("re-tuned TiSa overlap upstairs --> additional 50% improvement")]#,[2346] is a pdl scan though. gross...#2178 is Ti:Sa, but actually gross af#
massScanDic[247]=[2311,2322]#,[2188,2190](also decent, but from diff era with different rates)
initCenterEsts=[13284.75,13278.62,13272.50,13266.5]#,13260.35]
transitionLabels = np.array(["%d->%d"%(i,i) for i in range(len(initCenterEsts))])
estimationMethodLabels = np.array(['SkewedMu','LocMaxDat','LocMaxFit'])
estimationStatisticsLabels = np.array(['mean','stderr','range'])
sigmaEst=.55
gammaEst=1.77
skew0=-2
resolutionList=[.01,.02,.05,.1,.2,.5]
isotopeDataFrame = pd.DataFrame(index=massList, columns = pd.MultiIndex.from_product([transitionLabels,estimationMethodLabels, estimationStatisticsLabels], names=['Transitions','Methods','Stats']) )
isotopeDataFrame=isotopeDataFrame.sort_index()
for m in massList:
  s=massScanDic[m]
  peakList=initCenterEsts
  finalfitResults = FCUK.Scanalyzer(m,s,rewrite=rewrite,peakList=peakList,peakSigmas=sigmaEst*np.ones_like(peakList),sameSigma=sameSigma,initGamma=gammaEst,resList=resolutionList, ltrim=ltrim, rtrim=rtrim, makePlots=True,sameSkew=True, useWeights=True, skew0=-4)
  (fce,fdl,ffl)=(finalfitResults['fce'],finalfitResults['fdl'],finalfitResults['ffl'])
  for p in range(len(peakList)):
    isotopeDataFrame.loc[m,(transitionLabels[p],'SkewedMu')]=fce[p,:]
    isotopeDataFrame.loc[m,(transitionLabels[p],'LocMaxDat')]=fdl[p,:]
    isotopeDataFrame.loc[m,(transitionLabels[p],'LocMaxFit')]=ffl[p,:]
  with pd.option_context('display.max_rows', 100, 'display.max_columns', 60): print("newest isotope entry in dataframe:\n",isotopeDataFrame.loc[m,(idx[:,:,'mean'])])
    
with pd.option_context('display.max_rows', 100, 'display.max_columns', 100):print("ayy, are we done? isotopeDataFrame:\n",isotopeDataFrame)
#if not os.path.exists('./FitResults/OutputFiles/'): os.makedirs('./FitResults/OutputFiles/')
#isotopeDataFrame.to_csv(path_or_buf='./FitResults/OutputFiles/IsotopeDataFrame.csv')
"""

