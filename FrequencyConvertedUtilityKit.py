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


def backgroundEstimator(yArray):
  l = len(yArray)
  orderedByHeight = np.sort(yArray)
  beegee = np.mean(orderedByHeight[0:int(l/8)])
  return(beegee)

def fitPlottingSubRoutine(datFrame, mass, scan, r, n, fitRes, redchi=-1):
  #fullDatArray=datDic[x]
  #datArray=dataRebinner(fullDatArray,r) #inputting full dataset so that fit can be plotted at higher resolution for large-r datasets
  print("making res=%.3f, n=%d plot for mass%d scan %d"%(r,n,mass,scan))
  fitDic = fitRes.best_values
  fitPlotFig = plt.figure()
  plt.gcf().set_size_inches(20, 12)
  xDat = np.array(datFrame.loc[:,'wavenumber_mean']); yDat = np.array(datFrame.loc[:,'signal_value']); ySigDat = np.array(datFrame.loc[:,'signal_uncertainty'])
  #xdat = datFrame.loc[:,'wavenumber_mean']; ydat = datFrame.loc[:,'signal_value']; sigydat = datFrame.loc[:,'signal_uncertainty']
  xFull = np.arange(np.min(xDat)-.05,np.max(xDat)+.05,.01)
  #plt.plot(xdat, ydat, 'b.')
  plt.errorbar(xDat, yDat, yerr=ySigDat, fmt='b.-',ecolor='k', alpha=.5)
  plt.fill_between(xDat, yDat,color='blue', alpha=.25)
  plt.xlabel(r'Wavenumber (cm$^{-1}$)', fontsize=18)
  plt.ylabel('rate (counts/s)', fontsize=18)
  plt.title(r'$^{%d}$Ra$^{19}$F Spectrum; Scan %d; Resolution = %.2f, Fitting to %d peaks' %(mass-19, scan, r, n), fontsize=24)
  plt.plot(xFull, fitRes.eval(x=xFull,params=fitRes.init_params), 'k--', label="init_Fit")
  plt.plot(xFull, fitRes.eval(x=xFull,params=fitRes.params), 'r-', label="best_fit",linewidth=5)
  comps = fitRes.eval_components()
  bg = backgroundEstimator(yDat)
  plt.plot(xDat, comps['l0_'],'--', label="linear term")
  for i in range(n):
    plt.plot(xDat, comps['sv'+str(i)+'_'], '--', label="peak "+str(i))
    xpos=fitDic['sv'+str(i)+'_center']
    plt.axvline(xpos, 0,1, color='red', linestyle='dashed')
    plt.annotate(s=r'$\mu_%d = %.3f$'%(i, xpos), xy=(xpos, bg/2-i*bg/(4*n)), fontsize=14, ha='center', xycoords=('data','data'))
  plt.legend(loc=2, fontsize=18)
  if redchi==-1: pass
  else: plt.annotate(s=r'$\chi_{red}^2 = %f$'%redchi, xy=(.25,.2), fontsize=14, ha='center', xycoords=('figure fraction','figure fraction'))
  if not os.path.exists('./FitResults/mass%dFits/Scan%dFits'%(mass,scan)): os.makedirs('./FitResults/mass%dFits/Scan%dFits'%(mass,scan))
  plt.gcf().savefig("FitResults/mass%dFits/Scan%dFits/Mass%d_Scan%d_%dbins_%dpeaksFit.png"%(mass,scan,mass,scan,len(datFrame.index),n))
  plt.close()

def fitNPeaks(datFrame, peaksList, peakSigmas=np.array([]), method='leastsq', useWeights=False, sameSkew=True, skewList=np.array([]), skew0="NaN", initGamma=1): #fit scan data with rebinning to spectrum with pre-guessed peaks
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

def fitScanX(scanFrame, mass, scan, peaksList, peakSigmas=np.array([]),initGamma=1,resList=[.01,.02,.05,.1,.2,.5], method='leastsq', makePlots=True, sameSkew=True, useWeights=True, lcrop=-1, rcrop=-1, skewList=np.array([]), skew0='NaN'):
  #fit scan x data with rebinning R to spectrum with N peaks #TODO:minR, maxN
  #For later: scanFrame = lmd.rawDatPrep(mass,scan, wavenumber(!?!), other opts)
  #Also for later, impose crop regions in wavenumber
  print("now running fitScanX for mass=%d; scan=%d"%(mass,scan))
  if len(peakSigmas) == 0:
    peakSigmas = 2*np.ones_like(peaksList)
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
  compiledResults = pd.DataFrame(columns=["GoF", 'fitResults', 'PeakEsts'], index=resList)#,type=['float',])
  """Above this line is just prepwork"""
  #for r in resList: #range(R, minR-1, -1):
  for i in range(len(resList)):
    r=resList[i]
    print("resolution=", r)
    datFrame = lmd.makeUseable(scanFrame, resolution=r, cropSparseEnds=True, noNaNsense=True)#TODO allow for range cropping
    (fitRes, warningStatus) = fitNPeaks(datFrame, peaksList, peakSigmas=peakSigmas, initGamma=initGamma, useWeights=useWeights, sameSkew=sameSkew)#add other opts?
    if warningStatus == -1:
      print("That's it for this scan, boys. Don't. push. these peaks. They're. close. to. the. eeeedge. (One of the peaks is leaking out of the scan window at rebin setting%d)"%r)
      return(compiledGoFs[:i], compiledCenterEsts[:i])
    if not os.path.exists('./FitResults/Mass%dFits/Scan%dFits'%(mass,scan)): os.makedirs('./FitResults/Mass%dFits/Scan%dFits'%(mass,scan))
    fitReportFile = open('./FitResults/Mass%dFits/Scan%dFits/Mass%dScan%d_%dbins_FitReport.txt'%(mass,scan,mass,scan,len(datFrame.index)),'w+')
    fitReportFile.write(fitRes.fit_report(min_correl=0.25)); fitReportFile.close()

    if makePlots == True: fitPlottingSubRoutine(datFrame, mass, scan, r, N, fitRes, redchi=fitRes.redchi)
    compiledFitResults[r] = fitRes.best_values
    #print("test1: type(fitRes.best_values) = ", type(fitRes.best_values) )
    compiledResults.loc[r,"FitResults"] = [fitRes.best_values]
    fitCenterEsts = np.zeros([N,2])
    parmCenterNames = ['sv'+str(j)+'_center' for j in range(N)].append(['l0_slope', 'l0_intercept'])
    kwargs = {'p_names':parmCenterNames}
    print("fitRes.errorbars", fitRes.errorbars)
    if fitRes.errorbars:
      #fitRes.conf_interval(**kwargs)#/Try adding some Try/Except shenannigans?
      for p in range(N):
        fitCenterEsts[p, 0] = fitRes.best_values['sv'+str(p)+'_center']; fitCenterEsts[p,1] = fitRes.params['sv'+str(p)+'_center'].stderr
    else:
      for p in range(N):
        fitCenterEsts[p, 0] = fitRes.best_values['sv'+str(p)+'_center']; fitCenterEsts[p,1] = 4
      #TODO: reincorporate jack-knife type thing?
      """if r==1:
        for p in range(N):
          fitCenterEsts[p, 0] = fitRes.best_values['sv'+str(p)+'_center']; fitCenterEsts[p,1] = 2*byEyeSorted[p,1]
      else:
        print("#Cry. lmfit won't give me errorbars, so I'll just have to redo the fit with my bin cutoffs shifted halfway over, then compare the results from both fits")
        datArray2 = dataRebinner(datDic[x][int(r/2):], r) #Cry. If lmfit won't give me errorbars, I'll just have to redo the fit with my bin cutoffs shifted halfway over, then compare the results from both fits
        (fitRes2, warningStatus2) = fitNPeaks(datArray2, byEyeSorted[:,0], peakSigmas=byEyeSorted[:,1], useWeights=useWeights, sameSkew=sameSkew)
        if warningStatus2 == -1:
          print("That's it for this scan, boys. Don't. push. these peaks. They're. close. to. the. eeeedge. (One of the peaks is leaking out of the scan window at rebin setting%d)"%r)
          return(compiledGoFs[:r-minR], compiledCenterEsts[:r-minR])
        for p in range(N):
          fitCenterEsts[p, 0] = (fitRes.best_values['sv'+str(p)+'_center'] + fitRes2.best_values['sv'+str(p)+'_center'])/2
          fitCenterEsts[p,1] = max((fitRes.best_values['sv'+str(p)+'_center'] - fitRes2.best_values['sv'+str(p)+'_center'])*5, byEyeSorted[p,1]) #Cry... the *5 is to be conservative, since this 'jackknife' is quite jank.
          #That or my by-eye estimate of the uncertainty, which will probably be larger..."""
    compiledCenterEsts[i,:,:] = fitCenterEsts
    #print("test2: type(fitCenterEsts) = ", type(fitCenterEsts) )
    compiledResults.loc[r,'PeakEsts'] = [fitCenterEsts]
    print("Just ran fitScanX routine for: scan %d; res = %.3f; N=%d, and found redchi = %f"%(scan,r,N,fitRes.redchi), "fitCenterEsts:\n", fitCenterEsts[:,:])
    compiledGoFs[i] = fitRes.redchi
    compiledResults.loc[r,'GoF'] = fitRes.redchi
  #print("report from last fit performed:\n",fitRes.fit_report())
  compiledResultsFile = open('./FitResults/Mass%dFits/Scan%dFits/AllFitResultsCompiledMass%dScan%d-%dPeaks.txt'%(mass,scan,mass,scan,N),'w+')
  compiledResultsFile.write('Mass: %d ; Scan: %d\nInitial peak center estimates:'%(mass,scan)+str(peaksList)); compiledResultsFile.write('\nInitial peak width estimates:'+ str(peakSigmas));
  compiledResultsFile.write("\nCompilation of Results:\n");
  with pd.option_context('display.max_rows', 100, 'display.max_columns', 200): compiledResultsFile.write(str(compiledResults));
  compiledResultsFile.close()

  with open('./FitResults/Mass%dFits/Scan%dFits/ResultsCompiledForInput_Mass%d_Scan%d_%dPeaks-cfrx.txt'%(mass,scan,mass,scan,N),'wb') as outputFile:
    outputFile.write(json.dumps(compiledGoFs.tolist()).encode("utf-8"));
    outputFile.close()
  with open('./FitResults/Mass%dFits/Scan%dFits/ResultsCompiledForInput_Mass%d_Scan%d_%dPeaks-ccex.txt'%(mass,scan,mass,scan,N),'wb') as outputFile:
    outputFile.write(json.dumps(compiledCenterEsts.tolist()).encode("utf-8"));
    outputFile.close()
  return(compiledGoFs, compiledCenterEsts) #TODO?: get peak estimates, apply them to next fit. Carry out next fit(s) -> Nahhh

def MultiBinSpreadPlotter(mass, scan, ccex, resList):
  finScanEsts = np.c_[np.mean(ccex[:,:,0], axis=0), np.std(ccex[:,:,0], axis=0), np.max(ccex[:,:,0], axis=0)-np.min(ccex[:,:,0], axis=0)]
  fig = plt.figure(2)
  plt.clf()
  fig.set_size_inches(20, 12)
  colorCoding=['red','orange','yellow','green','blue','purple']
  yVals = np.arange(len(ccex[:,0,0]))
  for p in range(len(ccex[0,:,0])):
    plt.gca().add_patch(Rectangle((finScanEsts[p,0]-3*finScanEsts[p,1], yVals[0]-1), 6*finScanEsts[p,1], yVals[-1]+1, color=colorCoding[p], alpha=.25))
    plt.axvline(finScanEsts[p,0], ymin=0, ymax=1, color=colorCoding[p], linestyle='--')
    print("test: len(ccex[:,p,0])=%d ; len(yVals)=%d"%(len(ccex[:,p,0]), len(yVals)) )
    plt.errorbar(ccex[:,p,0], yVals, xerr=ccex[:,p,1], fmt='o', color=colorCoding[p], markeredgecolor='black', markersize=7, label="Peak %d"%(p+1))
    #TODO: plt.errorbar()
    #plt.annotate(r'$\chi^2_{red} = %.2f$'%cfrx[r-1,p-1], xy=(ccex[r,p][-1], r-.3), fontsize=8, ha='center', xycoords=('data','data'))
    plt.title(r'Center Parameter Estimates vs. Resolution Setting Fits of $^{%d}$Ra$^{19}$F Scan %d'%(mass-19,scan), fontsize=18)
    plt.xlabel(r'$\nu_{ex} (cm)^{-1}$', fontsize=16)
    plt.ylabel("Resolution Setting", fontsize=16)
    #TODO:yaxis tick labels

  plt.ylim([-.5, yVals[-1]+.5])
  plt.xlim(plt.gca().get_xlim())

  #box = plt.gca().get_position()
  #plt.gca().set_position([box.x0, box.y0, box.width * 0.95, box.height])
  lgd=plt.legend(loc="center right", bbox_to_anchor=(1.05,.5), fontsize=12)
  #vLinePlotter(vlineArrayPI12, r'$^2\Sigma^{+}_{1/2} \rightarrow ^2\Pi_{1/2}$')
  #vLinePlotter(vlineArrayDELTA32, r'$^2\Delta_{3/2}$')
  #vLinePlotter(vlineArrayDELTA52, r'$^2\Delta_{5/2}$')

  if not os.path.exists('./FitResults/Mass%dFits/Scan%dFits/'%(mass,scan)): os.makedirs('./FitResults/Mass%dFits/Scan%dFits'%(mass,scan))
  plt.savefig('./FitResults/Mass%dFits/Scan%dFits/CenterParametersEstimateSpread_Mass%dScan%d_%dPeaks.png'%(mass,scan,mass,scan,len(ccex[0,:,0])), bbox_extra_artists=(lgd,), bbox_inches='tight')
  if not os.path.exists('./FitResults/RebinDependencePlots/'): os.makedirs('./FitResults/RebinDependencePlots/')
  plt.savefig('./FitResults/RebinDependencePlots/CenterParametersEstimateSpread_Mass%dScan%d_%dPeaks.png'%(mass,scan,len(ccex[0,:,0])), bbox_extra_artists=(lgd,), bbox_inches='tight')
  plt.close(2)

def weightedStatistics(dataVals, dataSigs, transitionLabel="'this'"):
  if len(dataVals)==1:
    print("yo... There was only one data point. What are you averaging, dawg?")
    return((dataVals[0], dataSigs[0]))
  dataVarians = np.square(dataSigs)
  weightedMean = (1/np.sum(1/dataVarians))*np.sum(dataVals/dataVarians)
  statistVar = 1/np.sum(1/dataVarians)
  scatterVar = statistVar*(1/(len(dataVals)-1))*np.sum( np.square(dataVals - weightedMean)/dataVarians )
  if statistVar>scatterVar:
    print("Statistical variance larger than scattering variance for "+transitionLabel+" transition. Will use statistVar to report final uncertainty.")
    weightedError=math.sqrt(statistVar)
  elif statistVar<scatterVar:
    print("Scatter variance larger than scattering variance for "+transitionLabel+" transition. Will use scatterVar to report final uncertainty.")
    weightedError=math.sqrt(statistVar)
  else:
    print("wtf, what are the odds!? statistVar==scatterVar = ", statistVar==scatterVar)
    weightedError=math.sqrt(statistVar)
  return((weightedMean, weightedError))

def Scanalyzer(m, s, rewrite=False, peakList=[13285,13278.8,13272.8,13266.57], peakSigmas=np.array([]),initGamma=1, resList=[.01,.02,.05,.1,.2,.5], method="leastsq", fitPlots=True, binSpreadPlot=True, sameSkew=False, useWeights=True,skew0=float('NaN')):
  print("Running Scanalyzer. scan =",s)
  initCenterEsts=[13285,13278.8,13272.8,13266.57]#,13260]
  initWidthEsts=2*np.ones_like(initCenterEsts)
  resolutionList=[.01,.02,.05,.1,.2,.5]
  N = len(initCenterEsts)
  if (os.path.exists('./FitResults/Mass%dFits/Scan%dFits/ResultsCompiledForInput_Mass%d_Scan%d_%dPeaks-ccex.txt'%(m,s,m,s,N)) and (rewrite==False)):
    print("Success!")
    with open('./FitResults/Mass%dFits/Scan%dFits/ResultsCompiledForInput_Mass%d_Scan%d_%dPeaks-ccex.txt'%(m,s,m,s,N),'r') as ccexFile:
      ccex = np.array(json.load(ccexFile))
  else:
    mfba = lmd.rawDatPrep(m,s)
    (cfrx,ccex) = fitScanX(mfba,m,s, peakList, peakSigmas=peakSigmas, initGamma=initGamma, resList=resList, method=method, makePlots=fitPlots, useWeights=useWeights, sameSkew=sameSkew, skew0=skew0)
  finalScanEstimates = np.c_[np.mean(ccex[:,:,0], axis=0), np.std(ccex[:,:,0], axis=0), np.max(ccex[:,:,0], axis=0)-np.min(ccex[:,:,0], axis=0)]
  #print("finalScanEstimates.shape=",finalScanEstimates.shape,"\nfinalScanEstimates:\n",finalScanEstimates)
  #finalScanEsts = np.copy(finalScanEstimates)
  for p in range(len(ccex[0,:,0])):
    weightStats = weightedStatistics(ccex[:,p,0], ccex[:,p,1])
    #print("test: weightStats = ", weightStats)
    finalScanEstimates[p,0] = weightStats[0]; finalScanEstimates[p,1] = weightStats[1]
  print("Also test: finalScanEstimates.shape=",finalScanEstimates.shape," finalScanEstimates:\n",finalScanEstimates)
  if not os.path.exists('./FitResults/OutputFiles/mass%d/'%(m)): os.makedirs('./FitResults/OutputFiles/mass%d/'%(m))
  xFile=open("./FitResults/OutputFiles/Mass%d/Mass%dScan%dCompiledFitEstimates.txt"%(m,m,s),'w+')
  xFile.write("#Compiled Fit Center Estimates:\n"+str(ccex))
  xFile.write("\n#Scanalyzer Final Estimates:\n"+str(finalScanEstimates[:,0])+"\n#1Sigma:\n"+str(finalScanEstimates[:,1])+"\n#Range:\n"+str(finalScanEstimates[:,2]))
  xFile.close()
  if binSpreadPlot==True:
    MultiBinSpreadPlotter(m,s,ccex,resList)
  return(finalScanEstimates)
'''
def initializeIDedPeakDictionary(targetDict, vibrationalLabelList):
  for vibeLabel in vibrationalLabelList:
      targetDict[vibeLabel]=[]

def NearTheory(targetDict, peakEntry, predicArray, vibrationalLabelList, electricLabel="", plotResults=True, predicUncerts=0):
  meanK=peakEntry[0]; sigmaK=peakEntry[1]; rangeK=peakEntry[2]
  if np.any(abs(meanK-predicArray)<=math.sqrt(rangeK**2+predicUncerts**2)):
    idLevelIndex = np.argmin(abs(meanK-predicArray)); vibeLabel = vibrationalLabelList[idLevelIndex]
    targetDict[vibeLabel].append(peakEntry)
    return(True)
  elif np.any(abs(meanK-predicArray)<=math.sqrt((5*sigmaK)**2+predicUncerts**2)):
    idLevelIndex = np.argmin(abs(meanK-predicArray)); vibeLabel = vibrationalLabelList[idLevelIndex]
    targetDict[vibeLabel].append(peakEntry)
    print("rangeK did not suffice, but the fitted peak was within 5 sigma of a theory peak.\nPeak Entry:", peakEntry, "\nPredic Entry:",electricLabel,vibeLabel)
    return(True)
  return(False)
'''

if __name__ == '__main__':
  allScansBigDic = {}
  for m in [241,242,243,244,245,247]:
    allScansBigDic[m] = lmd.makeScanToWavemeterDic(m, redo=False, verbose=False)
