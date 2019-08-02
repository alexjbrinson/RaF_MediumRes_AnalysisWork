import numpy as np
#import scipy as sp
#from scipy import interpolate
#import scipy.signal as sig
import matplotlib.pyplot as plt
import math
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
#from matplotlib.widgets import TextBox
#from matplotlib.widgets import Button
import os.path
import lmfit
from lmfit import Model, Parameter
from lmfit.models import SkewedVoigtModel, LinearModel, GaussianModel, LorentzianModel
import emcee
#import csv
import time
import numdifftools



def vLinePlotter(vlineArray, transitionLabel, reflections=False, numericLabels=False):
  for i in range(len(vlineArray)):
    plt.axvline(vlineArray[i],0,1, color='k', linestyle='dashed', linewidth=1, alpha=.75)
    if numericLabels:
      plt.annotate(s=vlabelArray[i]+"\n"+str(vlineArray[i]), xy=(vlineArray[i],.8), fontsize=8, ha='center', xycoords=('data','figure fraction'))
    else:
      plt.annotate(s=vlabelArray[i], xy=(vlineArray[i],.7+(i%6)/50), fontsize=8, ha='center', xycoords=('data','figure fraction'))
    if reflections:
      plt.axvline(vlineArray[i]-2*beta*gamma*vlineArray[i],0,1, color='r', linestyle='dashed', linewidth=.25, alpha=.75)
      if numericLabels:
        plt.annotate(s=vlabelArray[i]+"\nAnticolinear\nReflection\n%.2f"%(vlineArray[i]-2*beta*gamma*vlineArray[i]), xy=(vlineArray[i]-2*beta*gamma*vlineArray[i],.13), fontsize=8, ha='center', xycoords=('data','figure fraction'))
      else:
        plt.annotate(s=vlabelArray[i]+"\nReflection", xy=(vlineArray[i]-2*beta*gamma*vlineArray[i],.13), fontsize=8, ha='center', xycoords=('data','figure fraction'))
  plt.annotate(s=transitionLabel+', P', xy=(np.mean(vlineArray[0:6]), .85), fontsize=14, ha='center', xycoords=('data','figure fraction'))
  plt.annotate(s=transitionLabel+', Q', xy=(np.mean(vlineArray[6:12]), .8), fontsize=14, ha='center', xycoords=('data','figure fraction'))
  plt.annotate(s=transitionLabel+', R', xy=(np.mean(vlineArray[12:18]), .85), fontsize=14, ha='center', xycoords=('data','figure fraction'))

def ImportPeakEstimateDics(targetDictNames, fname):
  with open(fname, 'r') as f:
      for line in f:
        for i in range(len(targetDictNames)):
          if line.startswith(targetDictNames[i]):
            exec(line, globals())
          else: pass

def cleanDataSet(dRay):
  """crops data in v-space to remove sparse, pure-noise portions of scans"""
  meanSpacing = np.mean(dRay[1:,2]-dRay[:-1,2])
  i = 0
  while (dRay[i+1,2]-dRay[i,2]>3*meanSpacing) or (dRay[i+2,2]-dRay[i+1,2]>3*meanSpacing) or (dRay[i+3,2]-dRay[i+2,2]>3*meanSpacing):
  # If the any of the next 3 v-spacings are greater than 3 times the average spacing, increment the index at which to start cropping.
    i+=1
  len1 = len(dRay[:,2])
  len2 = len(dRay[i:,2])
  #print("test5: len1 = %d, len2 = %d"%(len1,len2))
  return dRay[i:,:]

def errorBinner(errs,binSize):
  datOldLength = len(errs)
  remainder = len(errs)%binSize
  if remainder !=0:
    trimmedErrs = errs[:-remainder]
  else:
    trimmedErrs = errs
  newLength = len(trimmedErrs)
  rebinnedErrs = np.zeros(int(newLength/binSize))
  #newLength=(len(errs)/binSize)
  #rebinnedErrs = np.zeros(newLength)
  for i in range(binSize):
    rebinnedErrs = rebinnedErrs + np.square(trimmedErrs[i::binSize])
  rebinnedErrs = np.sqrt(rebinnedErrs)
  return rebinnedErrs

def freqBinner(freqs,binSize):
  oldLength = len(freqs)
  remainder = len(freqs)%binSize
  if remainder !=0:
    trimmedFreqs = freqs[:-remainder]
  else:
    trimmedFreqs = freqs
  newLength = len(trimmedFreqs)
  rebinnedFreqs = np.zeros(int(newLength/binSize))
  for i in range(binSize):
    rebinnedFreqs = rebinnedFreqs + trimmedFreqs[i::binSize]
  rebinnedFreqs = rebinnedFreqs/binSize
  return rebinnedFreqs

def rateBinner(rates,binSize):
  datOldLength = len(rates)
  remainder = len(rates)%binSize
  if remainder !=0:
    trimmedRates = rates[:-remainder]
  else:
    trimmedRates = rates
  newLength = len(trimmedRates)
  rebinnedRates = np.zeros(int(newLength/binSize))
  for i in range(binSize):
    rebinnedRates = rebinnedRates + trimmedRates[i::binSize]
  return rebinnedRates

def dataRebinner(datRay, binSize):
  oldErrs= datRay[:,1]
  oldFreqs=datRay[:,2]
  oldRates=datRay[:,3]
  assert(np.all(oldFreqs == np.sort(oldFreqs)))
  remainder = len(oldErrs)%binSize
  if remainder !=0:
    trimmedDatRay = datRay[:-remainder,:]
  else:
    trimmedDatRay = datRay
  newLength = (len(oldErrs)-remainder)/binSize
  newInds = np.arange(newLength)
  newErrs = errorBinner(trimmedDatRay[:,1], binSize)
  newFreqs = freqBinner(trimmedDatRay[:,2], binSize)
  newRates = rateBinner(trimmedDatRay[:,3], binSize)
  #print("test4: lengths = %d, %d, %d, %d" %(len(newInds),len(newErrs), len(newFreqs), len(newRates) ))
  rebinnedData = np.c_[newInds, newErrs, newFreqs, newRates]
  return rebinnedData

def backgroundEstimator(yArray):
  l = len(yArray)
  orderedByHeight = np.sort(yArray)
  beegee = np.mean(orderedByHeight[0:int(l/8)])
  return(beegee)

def fitPlottingSubRoutine(x, r, n, fitRes, redchi=-1):
  fullDatArray=datDic[x]
  datArray=dataRebinner(fullDatArray,r) #inputting full dataset so that fit can be plotted at higher resolution for large-r datasets
  print("making r=%d, n=%d plot for scan %d"%(r,n,x))
  fitDic = fitRes.best_values
  plt.figure(1)
  plt.clf()
  fig= plt.gcf()
  fig.set_size_inches(20, 12)
  plt.plot(datArray[:,2], datArray[:,3], 'b.')
  plt.errorbar(datArray[:,2], datArray[:,3], yerr=datArray[:,1], fmt='b-', alpha=.5)
  plt.fill_between(datArray[:,2], datArray[:,3],color='blue', alpha=.5)
  plt.xlabel(r'Wavenumber (cm$^{-1}$)', fontsize=18)
  plt.ylabel('rate', fontsize=18)
  plt.title(r'$^{226}$Ra$^{19}$F Spectrum Scan %d; rebin = %d, fitting to %d peaks' %(x, r, n), fontsize=24)
  plt.plot(fullDatArray[:,2], fitRes.eval(x=fullDatArray[:,2],params=fitRes.init_params), 'k--', label="init_Fit")
  plt.plot(fullDatArray[:,2], fitRes.eval(x=fullDatArray[:,2],params=fitRes.params), 'r-', label="best_fit",linewidth=5)
  comps = fitRes.eval_components()
  bg = backgroundEstimator(datArray[:,3])
  plt.plot(datArray[:,2], comps['l0_'],'--', label="linear term")
  for i in range(n):
    plt.plot(datArray[:,2], comps['sv'+str(i)+'_'], '--', label="peak "+str(i))
    xpos=fitDic['sv'+str(i)+'_center']
    plt.axvline(xpos, 0,1, color='red', linestyle='dashed')
    plt.annotate(s=r'$\mu_%d = %.3f$'%(i, xpos), xy=(xpos, bg/2-i*bg/(4*n)), fontsize=12, ha='center', xycoords=('data','data'))
  plt.legend(loc=2, fontsize=10)
  if redchi==-1: pass
  else: plt.annotate(s=r'$\chi_{red}^2 = %f$'%redchi, xy=(.35,.66), fontsize=12, ha='center', xycoords=('figure fraction','figure fraction'))
  if not os.path.exists('FitResults/Scan%dFits'%x):
    os.mkdir('FitResults/Scan%dFits'%x)
  fig.savefig("FitResults/Scan%dFits/Scan%d_rebin%d_%dpeaksFit.png"%(x,x,r,n))
  plt.close(1)

def fitNPeaks(data, peaksList, peakRanges=np.array([]), method='leastsq', useWeights=False, sameSkew=False): #fit scan data with rebinning to spectrum with pre-guessed peaks
  if len(peakRanges) == 0:
    peakRanges = 5*np.ones_like(peaksList)
  else:
    assert(peakRanges.shape == peaksList.shape)
    assert(np.all(peakRanges>0))
  N = len(peaksList)
  xDat = data[:,2]; yDat = data[:,3]; ySigDat = data[:,1]
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
    k = peaksList[i]; sigmaK = peakRanges[i] 
    ind1 = np.argmin(abs(xDat-k))
    if ind1 <= 3 or len(xDat)-ind1<=3:
      print("WARNING: Peak occurs too closely to edge of dataset. A lower rebin setting is recommended.")
      warningStatus=-1
      return(False, warningStatus)
    ind2 = np.argmax(yDat[ind1-2:ind1+3])+ind1-2
    estimHeight = yDat[ind2] - bg
    svmod = SkewedVoigtModel(prefix="sv"+str(i)+"_")
    svmod.set_param_hint('center', value=peaksList[i], min=max(peaksList[i]-2*peakRanges[i], xDat[3]), max=min(peaksList[i]+2*peakRanges[i],xDat[-3]))
    svmod.set_param_hint('sigma', value=peakRanges[i], min=0.1, max=2*peakRanges[i])
    svmod.set_param_hint('amplitude', value=estimHeight*(peakRanges[i]/(1*.45)), min=3*np.mean(ySigDat))
    if N>2: svmod.set_param_hint('skew', value = -4, min=-12,max=12,)
    elif N<=2: svmod.set_param_hint('skew', value = 0, min=-1,max=1, vary=True)
    peakModelsArray.append(svmod)
    params += svmod.make_params()#svmod.guess(yDat, x=xDat)#
    if i == 0: params['sv'+str(i)+'_gamma']= Parameter(value=1, min=0, max = 3*peakRanges[i], vary=True)
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

#def startFromPrevFit(data, additPeak, additPeakSig, fitRes, method='leastsq', useWeights=False, sameSkew=False): #fit scan data with rebinning to spectrum with pre-guessed peaks
  #Hmm, this would risk building off of a trash fit and never recovering though...

def fitScanX(x, R, method='leastsq', makePlots=True, sameSkew=False, minR=1, maxN=10, useWeights=True): #fit scan x data with rebinning R to spectrum with N peaks #TODO:minR, maxN
  datArray = datDic[x]
  xDat = datArray[:,2]; yDat = datArray[:,3]; ySigDat = datArray[:,1]
  peaksList = peakEsts[x]; peakRanges = peakUncerts[x]
  N = len(peaksList)
  if N>maxN:
    print("N=%d was larger than maxN=%d, will only fit to first %d peaks."%(N,maxN,maxN))
    N=maxN
  byEyeHeights = []
  byEyeData = [[]]
  for i in range(len(peaksList)):
    k = peaksList[i]; sigmaK = peakRanges[i] 
    ind1 = np.argmin(abs(xDat-k))
    ind2 = np.argmax(yDat[ind1-2:ind1+2])+ind1-2
    estimHeight = yDat[ind2]
    byEyeHeights.append(estimHeight)
  byEyeData = np.c_[peaksList, peakRanges, byEyeHeights]
  #byEyeSorted = np.array(sorted(byEyeData, reverse=True, key= lambda ray: ray[2]))[:maxN] #sorts (by-eye) peak list from tallest to shortest
  byEyeSorted = byEyeData[:maxN]
  compiledGoFs = -1*np.ones(R-minR+1)
  compiledFitResults = {}
  compiledCenterEsts = -1*np.ones([R-minR+1, N, 2])
  """Above this line is just prepwork"""
  for r in range(minR, R+1, 1): #range(R, minR-1, -1):
    print("r=", r)
    datArray = dataRebinner(datDic[x], r)
    (fitRes, warningStatus) = fitNPeaks(datArray, byEyeSorted[:,0], peakRanges=byEyeSorted[:,1], useWeights=useWeights, sameSkew=sameSkew)
    if warningStatus == -1:
      print("That's it for this scan, boys. Don't. push. these peaks. They're. close. to. the. eeeedge. (One of the peaks is leaking out of the scan window at rebin setting%d)"%r)
      return(compiledGoFs[:r-minR], compiledCenterEsts[:r-minR])
    if not os.path.exists('FitResults/Scan%dFits'%x): os.mkdir('FitResults/Scan%dFits'%x)
    fitReportFile = open("FitResults/Scan%dFits/Scan%d_Rebin%d_FitReport.txt"%(x,x,r),'w+')
    fitReportFile.write(fitRes.fit_report(min_correl=0.25)); fitReportFile.close()

    if makePlots == True: fitPlottingSubRoutine(x,r,N,fitRes, redchi=fitRes.redchi)
    compiledFitResults[r] = fitRes.best_values
    fitCenterEsts = np.zeros([N,2])
    parmCenterNames = ['sv'+str(j)+'_center' for j in range(N)].append(['l0_slope', 'l0_intercept'])
    kwargs = {'p_names':parmCenterNames}
    print("fitRes.errorbars", fitRes.errorbars)
    if fitRes.errorbars:
      #fitRes.conf_interval(**kwargs)#/Try adding some Try/Except shenannigans?
      for p in range(N):
        fitCenterEsts[p, 0] = fitRes.best_values['sv'+str(p)+'_center']; fitCenterEsts[p,1] = fitRes.params['sv'+str(p)+'_center'].stderr
    else:
      if r==1:
        for p in range(N):
          fitCenterEsts[p, 0] = fitRes.best_values['sv'+str(p)+'_center']; fitCenterEsts[p,1] = 2*byEyeSorted[p,1]
      else:
        print("#Cry. lmfit won't give me errorbars, so I'll just have to redo the fit with my bin cutoffs shifted halfway over, then compare the results from both fits")
        datArray2 = dataRebinner(datDic[x][int(r/2):], r) #Cry. If lmfit won't give me errorbars, I'll just have to redo the fit with my bin cutoffs shifted halfway over, then compare the results from both fits
        (fitRes2, warningStatus2) = fitNPeaks(datArray2, byEyeSorted[:,0], peakRanges=byEyeSorted[:,1], useWeights=useWeights, sameSkew=sameSkew)
        if warningStatus2 == -1:
          print("That's it for this scan, boys. Don't. push. these peaks. They're. close. to. the. eeeedge. (One of the peaks is leaking out of the scan window at rebin setting%d)"%r)
          return(compiledGoFs[:r-minR], compiledCenterEsts[:r-minR])
        for p in range(N):
          fitCenterEsts[p, 0] = (fitRes.best_values['sv'+str(p)+'_center'] + fitRes2.best_values['sv'+str(p)+'_center'])/2
          fitCenterEsts[p,1] = max((fitRes.best_values['sv'+str(p)+'_center'] - fitRes2.best_values['sv'+str(p)+'_center'])*5, byEyeSorted[p,1]) #Cry... the *5 is to be conservative, since this 'jackknife' is quite jank.
          #That or my by-eye estimate of the uncertainty, which will probably be larger...
    compiledCenterEsts[r-minR+1-1,:,:] = fitCenterEsts
    print("Just ran fitScanX routine for:x=%d; r=%d; N=%d, and found redchi = %f"%(x,r,N,fitRes.redchi), "fitCenterEsts:\n", fitCenterEsts[:,:])
    compiledGoFs[r-minR] = fitRes.redchi
  #print("report from last fit performed:\n",fitRes.fit_report())
  return(compiledGoFs, compiledCenterEsts) #TODO?: get peak estimates, apply them to next fit. Carry out next fit(s) -> Nahhh

def MultiBinSpreadPlotter(x, ccex, minR=1, maxR=20):
  finScanEsts = np.c_[np.mean(ccex[:,:,0], axis=0), np.std(ccex[:,:,0], axis=0), np.max(ccex[:,:,0], axis=0)-np.min(ccex[:,:,0], axis=0)]
  fig = plt.figure(2)
  plt.clf()
  fig.set_size_inches(20, 12)
  colorCoding=['red','orange','yellow','green','blue','purple']
  for p in range(len(ccex[0,:,0])):
    plt.gca().add_patch(Rectangle((finScanEsts[p,0]-3*finScanEsts[p,1], minR-1), 6*finScanEsts[p,1], maxR+2, color=colorCoding[p], alpha=.25))
    plt.axvline(finScanEsts[p,0], ymin=0, ymax=1, color=colorCoding[p], linestyle='--')
    print("test: len(ccex[:,p,0])=%d ; len(np.arange(minR, minR+len(ccex[:,p,0]), 1))=%d"%( len(ccex[:,p,0]), len(np.arange(minR, minR+len(ccex[:,p,0]), 1)) ))
    plt.errorbar(ccex[:,p,0], np.arange(minR, minR+len(ccex[:,p,0]), 1), xerr=ccex[:,p,1], fmt='o', color=colorCoding[p], markeredgecolor='black', markersize=7, label="Peak %d"%(p+1))
    #TODO: plt.errorbar()
    #plt.annotate(r'$\chi^2_{red} = %.2f$'%cfrx[r-1,p-1], xy=(ccex[r,p][-1], r-.3), fontsize=8, ha='center', xycoords=('data','data'))
    plt.title("Center Parameter Estimates vs. Rebin Setting Fits of Scan %d"%x, fontsize=18)
    plt.xlabel(r'$\nu_{ex} (cm)^{-1}$', fontsize=16)
    plt.ylabel("Rebin Setting", fontsize=16)

  plt.ylim([minR-.5, minR+len(ccex[:,p,0])+.5])
  plt.xlim(plt.gca().get_xlim())
  """for i in range(len(ccex[0,:,0])):
    plt.plot(-1, -1, '.',color=colorCoding[i], label="Peak %d"%(i+1))"""
  box = plt.gca().get_position()
  plt.gca().set_position([box.x0, box.y0, box.width * 0.95, box.height])
  plt.legend(loc="center right", bbox_to_anchor=(1.15,.5), fontsize=12)
  vLinePlotter(vlineArrayPI12, r'$^2\Sigma^{+}_{1/2} \rightarrow ^2\Pi_{1/2}$')
  vLinePlotter(vlineArrayDELTA32, r'$^2\Delta_{3/2}$')
  vLinePlotter(vlineArrayDELTA52, r'$^2\Delta_{5/2}$')

  if not os.path.exists('FitResults/Scan%dFits'%x):
    os.mkdir('FitResults/Scan%dFits'%x)
  plt.savefig("FitResults/Scan%dFits/CenterParametersEstimateSpread_Scan%d.png"%(x,x))
  if not os.path.exists('FitResults/RebinDependencePlots'):
    os.mkdir('FitResults/RebinDependencePlots')
  plt.savefig("FitResults/RebinDependencePlots/CenterParametersEstimateSpread_Scan%d.png"%x)
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

def Scanalyzer(x, minR=1, maxR=20, maxN=10, method="emcee", fitPlots=True, binSpreadPlot=True, sameSkew=False, useWeights=True):
  rbin = maxR #initial bin settings
  print("Running Scanalyzer. x =",x)
  (cfrx,ccex) = fitScanX(x, rbin, method=method, minR=minR, maxN=maxN, useWeights=useWeights, sameSkew= (sameSkew))# and ((scanCount>9)) ) #After scan 10-1, mirror removed, so we can add assumption about peak shapes similarities(?)
  #print("fitScanX: compiled fits reduced chi^2 vals:\n", cfrx, "\nccex:", ccex)
  """print("test1:\n",np.mean(ccex[:,:,0], axis=0))
  print("test2:\n", np.std(ccex[:,:,0], axis=0))
  print("test3:\n", np.max(ccex[:,:,0], axis=0)-np.min(ccex[minR-1:,:,0], axis=0))"""
  finalScanEstimates = np.c_[np.mean(ccex[:,:,0], axis=0), np.std(ccex[:,:,0], axis=0), np.max(ccex[:,:,0], axis=0)-np.min(ccex[:,:,0], axis=0)]
  #print("finalScanEstimates.shape=",finalScanEstimates.shape,"\nfinalScanEstimates:\n",finalScanEstimates)
  #finalScanEsts = np.copy(finalScanEstimates)
  for p in range(len(ccex[0,:,0])):
    weightStats = weightedStatistics(ccex[:,p,0], ccex[:,p,1])
    #print("test: weightStats = ", weightStats)
    finalScanEstimates[p,0] = weightStats[0]; finalScanEstimates[p,1] = weightStats[1]
  print("Also test: finalScanEstimates.shape=",finalScanEstimates.shape," finalScanEstimates:\n",finalScanEstimates)

  if not os.path.exists('FitResults/Scan%dFits/OutputFiles'%x):
    os.mkdir('FitResults/Scan%dFits/OutputFiles'%x)
  np.savetxt("FitResults/Scan"+str(x)+"Fits/OutputFiles/CompiledFitRedChis.csv", cfrx, delimiter=",")
  np.savetxt("FitResults/Scan"+str(x)+"Fits/OutputFiles/ScanalyzerOutput.csv", cfrx, delimiter=",")
  xFile=open("FitResults/Scan"+str(x)+"Fits/OutputFiles/CompiledFitEstimates.txt",'w+')
  xFile.write("#Compiled Fit Center Estimates:\n"+str(ccex))
  xFile.write("\n#Scanalyzer Final Estimates:\n"+str(finalScanEstimates[:,0])+"\n#1Sigma:\n"+str(finalScanEstimates[:,1])+"\n#Range:\n"+str(finalScanEstimates[:,2]))
  xFile.close()
  if binSpreadPlot==True:
    MultiBinSpreadPlotter(x, ccex, minR=minR, maxR=maxR)
  return(finalScanEstimates)

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

class Index(object):
  #Controls button features of scanPlotter.
    ind = 0 #index of scan to be plotted.
    rbs = 1 #rebin setting

    def next(self, event):
      if self.ind == len(datDic)-1:
        self.ind = 0
      else:
        self.ind += 1
      scanCount = self.ind
      x = dickeys[scanCount]
      rebin = self.rbs
      scanPlotter(x, scanCount, ax, rebin, showLinePredictions=slp)
      plt.draw()

    def prev(self, event):
      if self.ind == 0:
        self.ind = len(datDic)-1
      else:
        self.ind -= 1
      scanCount = self.ind
      x = dickeys[scanCount]
      rebin = self.rbs
      scanPlotter(x, scanCount, ax, rebin, showLinePredictions=slp)
      plt.draw()

    def binInc(self, event):
      if self.rbs == 20:
        self.rbs = 20
      else:
        self.rbs += 1
      scanCount = self.ind
      x = dickeys[scanCount]
      rebin = self.rbs
      scanPlotter(x, scanCount, ax, rebin, showLinePredictions=slp)
      plt.draw()

    def binDec(self, event):
      if self.rbs == 1:
        self.rbs = 1
      else:
        self.rbs -= 1
      scanCount = self.ind
      x = dickeys[scanCount]
      rebin = self.rbs
      scanPlotter(x, scanCount, ax, rebin, showLinePredictions=slp)
      plt.draw()

def scanPlotter(x, axis, binSize, showLinePredictions=False):
  plt.sca(axis)
  plt.cla()
  datArray = datDic[x]
  assert(np.all(datArray[:,2] == np.sort(datArray[:,2])))
  stablePeakIds = identifyPeaks(datArray,1,8)
  stablePeakMeans = [np.mean(stablePeakIds[i]) for i in range(len(stablePeakIds))]
  print("test7: len(stablePeakIds)=%d, stablePeakMeans:\n"%len(stablePeakIds), stablePeakMeans)
  #print("stablePeakIds:\n", stablePeakIds)
  peaksByEye = peakEsts[x]
  peakSigmas = peakUncerts[x]
  datArray = dataRebinner(datArray, binSize)
  assert(np.all(datArray[:,2] == np.sort(datArray[:,2])))
  plt.plot(datArray[:,2], datArray[:,3], 'bo-', alpha=.5)
  plt.errorbar(datArray[:,2], datArray[:,3], yerr = datArray[:,1], fmt='k.', alpha=.25)
  plt.fill_between(datArray[:,2], datArray[:,3],color='blue', alpha=.5)
  plt.xlabel(r'Wavenumber (cm$^{-1}$)', fontsize=18)
  plt.ylabel('rate', fontsize=18)
  plt.title(r'$^{226}$Ra$^{19}$F Scan [%d]; rebin = %d' %(x, binSize), fontsize=24)
  #peakInds = sig.find_peaks(datArray[:,3],prominence=6*np.mean(datArray[:,1]),width=2)[0]
  peakInds = sig.find_peaks(datArray[:,3],prominence=6*np.mean(datArray[:,1]),width=math.ceil(len(datArray[:,0])/250))[0]
  rebinPeakFreqs = datArray[peakInds,2]
  plt.plot(datArray[peakInds,2],datArray[peakInds,3]+20,'yo')

  for k in stablePeakMeans:
    if k <0: break
    freqString = "%.2f" % k
    ind1 = np.argmin(abs(datArray[:,2]-k))
    ind2 = np.argmax(datArray[ind1-1:ind1+1,3])+ind1-1
    estimHeight = datArray[ind2,3]
    plt.plot(k, estimHeight,'ro')
    #plt.text(k,estimHeight*1.02, freqString,ha='center')

  for i in range(len(peaksByEye)):
    #if k <0: break
    k = peaksByEye[i]; sigmaK = peakSigmas[i] 
    print("tempTest: k=%d"%k)
    freqString = "%.2f" % k
    ind1 = np.argmin(abs(datArray[:,2]-k))
    print("tempTest: ind1=%d"%ind1)
    ind2 = np.argmax(datArray[ind1-2:ind1+2,3])+ind1-2
    estimHeight = datArray[ind2,3]
    #plt.plot(k, estimHeight,'go')
    plt.errorbar(k, estimHeight, xerr=sigmaK, fmt='go')
    plt.text(k,estimHeight*1.02, freqString,ha='center')

  if showLinePredictions==True:
    vLinePlotter(vlineArrayPI12, r'$^2\Sigma^{+}_{1/2} \rightarrow ^2\Pi_{1/2}$')
    vLinePlotter(vlineArrayDELTA32, r'$^2\Delta_{3/2}$')
    vLinePlotter(vlineArrayDELTA52, r'$^2\Delta_{5/2}$')
  plt.xlim(np.min(datArray[:,2])-2, np.max(datArray[:,2])+2)
  plt.ylim(.95*np.min(datArray[:,3]),1.05*np.max(datArray[:,3]))

def startTogglePlot(rebin=1, slp=True, scanCount=0):
  x = dickeys[scanCount] #ID number of scan initially plotted
  callback = Index()
  fig, ax = plt.subplots(figsize=(18,9))
  scanPlotter(x, 0, ax, rebin, showLinePredictions=slp)  
  """Buttons controlling which scan is plotted"""
  axprev = plt.axes([0.025, 0.8, 0.05, 0.075])
  axnext = plt.axes([0.925, 0.8, 0.05, 0.075])
  bnext = Button(axnext, 'Next')
  bnext.on_clicked(callback.next)
  bprev = Button(axprev, 'Previous')
  bprev.on_clicked(callback.prev)
  """Buttons controlling bin setting for current scan plot (TODO)"""
  axBinDec = plt.axes([0.025, 0.7, 0.05, 0.075])
  axBinInc = plt.axes([0.925, 0.7, 0.05, 0.075])
  bBinDec = Button(axBinDec, 'Decrease\nBin Size')
  bBinDec.on_clicked(callback.binDec)
  bBinInc = Button(axBinInc, 'Increase\nBin Size')
  bBinInc.on_clicked(callback.binInc)

if __name__ == '__main__':

  startTime = time.clock()
  """Loading in data from scans"""
  dirlist=os.listdir('scans245')
  scanInds = []
  print("test1. os.listdir('scans245'):\n",dirlist)
  for i in range(len(dirlist)):
    if (dirlist[i].endswith('.csv') and dirlist[i].startswith('245RaF_LR_')):
      scanInds.append( int(dirlist[i].replace('.csv',"").replace('245RaF_LR_',"")) )
  print("test2. scanInds:\n",scanInds)

  datDic = {}

  for i in range(len(scanInds)):
    datArray = np.loadtxt('scans245/245RaF_LR_'+str(scanInds[i])+'.csv', dtype=float, skiprows=1, delimiter=',')
    if datArray.ndim != 2:
      print("Junk dataset from scan "+str(scanInds[i])+". Will throw out.")
    elif len(datArray[:,2])<20:
      print("Few datapoints in scan "+str(scanInds[i])+". Will throw out.")
    else:
      if np.any(datArray[:,2]<0):
        mask = datArray[:,2]>0
        datArray = np.array(datArray[[mask==True]])
        print("Scan "+str(scanInds[i])+" contained negative wavenumbers...", str(len(mask)-len(datArray[:,2])) + " data point(s) have been removed. Updated array shape =", datArray.shape)
      datDic[scanInds[i]] = cleanDataSet(datArray)
      if scanInds[i]==2137:
        print("All of the signal in Scan 2137 occurs in the first 6th of the dataset. Will crop the rest so it doesn't dominate the fits.")
        datArray=datDic[2137]
        rCutoff = np.argmin(np.abs(datArray[:,2]-13325))
        datDic[2137] = datArray[:rCutoff]
      elif scanInds[i]==2138:
        print("All of the signal in Scan 2138 occurs in the last 6th of the dataset. Will crop the rest so it doesn't dominate the fits.")
        datArray=datDic[2138]
        lCutoff = np.argmin(np.abs(datArray[:,2]-13225))
        datDic[2138] = datArray[lCutoff:]
      elif scanInds[i]==2178:
        print("There's some fishy business going on int the first quarter of Scan 2178. Will crop so it doesn't screw up my fits.")
        datArray=datDic[2178]
        lCutoff = np.argmin(np.abs(datArray[:,2]-13256))
        datDic[2178] = datArray[lCutoff:]
      elif scanInds[i]==2368:
        print("Scan 2368 has garbage at the very end. Will crop so it doesn't screw up my fits.")
        datArray=datDic[2368]
        datDic[2368] = datArray[:-2]

  print("datDic.keys()", list(datDic.keys()))

  beta = 0.0005920684 #v_bunch/c
  gamma = 1.00000017527255 #1/sqrt(1-beta^2)
  dyeInds = [2130, 2131,2132,2135,2136,2137,2138,2139,2323,2324,2325,2346,2360,2364,2365,2368,2375,2376]
  tiSapInds = [2164,2165,2178,2309,2310,2317,2319,2320,2340,2341,2349,2350]

  vlabelArray = np.array(["6->5","5->4","4->3","3->2","2->1","1->0",
            "5->5","4->4","3->3","2->2","1->1","0->0",
            "5->6","4->5","3->4","2->3","1->2","0->1"])

  vlineArrayPI12 = [12833.3, 12835.6, 12838., 12840.6, 12843.2, 12846.,   #P-band, \Delta v =-1
                    13254.6, 13260.3, 13266.2, 13272.2, 13278.3, 13284.5, #Q-band, \Delta v =-0
                    13670.2, 13679.3, 13688.5,13697.8, 13707.2, 13716.8]  #R-band, \Delta v =+1

  vlineArrayDELTA32 = [14674.9, 14680.2, 14685.8, 14691.7, 14697.7, 14704.,
                       15096.2, 15105., 15114., 15123.3, 15132.8, 15142.5, 
                       15508.9, 15520.9, 15533.2, 15545.6, 15558.3, 15571.3]

  vlineArrayDELTA52 = [15699., 15706.4, 15713.7, 15721.2, 15728.8, 15736.5,
                       16120.3, 16131.1, 16141.9, 16152.8, 16163.9, 16175.,
                       16531., 16545.1, 16559.3, 16573.5, 16587.9, 16602.4]

  vlineArrayPI32 = [-1,-1,-1,-1,-1,-1, 
                    -1,-1,-1,15308, 15325, 15342,
                    -1,-1,-1,-1,-1,-1,]

  vlineArrayPI12Reflex = vlineArrayPI12 - 2*beta*gamma*np.array(vlineArrayPI12)

  predictedKs = np.union1d(np.union1d(vlineArrayPI12, vlineArrayPI12), np.union1d(vlineArrayDELTA32, vlineArrayDELTA52))

  """(Temporarily?) Global Variables:"""
  peakEsts = {}; peakUncerts = {};
  ImportPeakEstimateDics(["peakEsts", "peakUncerts"], "IdentifyingPeaksInScansByEye_Take2.txt")
  print("test: len(list(peakEsts.keys())) = ", len(list(peakEsts.keys())))
  rebinMinDic = {};rebinMaxDic = {}
  ImportPeakEstimateDics(["rebinMinDic","rebinMaxDic"], "rebinRangeSettingsDictionaries.txt")
  #print("test: rebinMaxDic = ", rebinMaxDic)

  PI12Dic={}; DELTA32Dic={}; DELTA52Dic={}; PI32Dic={}; PI12ReflexDic = {}
  initializeIDedPeakDictionary(PI12Dic, vlabelArray); initializeIDedPeakDictionary(DELTA32Dic, vlabelArray); initializeIDedPeakDictionary(DELTA52Dic, vlabelArray)
  initializeIDedPeakDictionary(PI32Dic, vlabelArray); initializeIDedPeakDictionary(PI12ReflexDic, vlabelArray)
  idDicList = [PI12Dic,DELTA32Dic,DELTA52Dic,PI32Dic, PI12ReflexDic] #List of dictionaries where I'll store Scanalyzer results consistent with theory
  predictionsList = [vlineArrayPI12,vlineArrayDELTA32,vlineArrayDELTA52,vlineArrayPI32, vlineArrayPI12Reflex]
  predictionsListUncertainties = [.5, 1, .5, 5, .5]
  electronicLabelList = ["PI_1/2","DELTA_3/2","DELTA_5/2","PI_32", "PI_1/2(Ref)"]
  theoryList=[]; mysteryList=[]

  showFinalPlot = True
  dickeys = np.sort(np.array(list(datDic.keys())))
  scanCountRange = range(len(dickeys))
  rebin=20 #maximum rebin setting, unless specified to be lower by rebinMaxDic
  minnerBinner = 1 #minimum rebin setting, unless specified to be higher by rebinMinDic
  for scanCount in scanCountRange:
    x=dickeys[scanCount]
    print("running the main event! x =%d"%x)
    print("test: rebinMaxDic[%d]=%d"%(x,rebinMaxDic[x]))
    minRebin = max(minnerBinner,rebinMinDic[x])
    maxRebin = min(rebin, rebinMaxDic[x])
    finScanEstsX = Scanalyzer(x, minR=minRebin, maxR=maxRebin, maxN=6, method="emcee", sameSkew=True, useWeights=True)#"leastsq")
    if maxRebin==minnerBinner: #if only one rebin setting exists, we have to add uncertainty by hand, since there's no data spread to draw from.
      finScanEstsX = np.c_[finScanEstsX[:,0], .5*np.ones_like(finScanEstsX[:,0]), np.ones_like(finScanEstsX[:,0])]
    print("Scanalyzer finished scanning for x=%d.\nFinal Scan Estimates:"%x,finScanEstsX[:,0],"\nNow comparing to theory predictions:")
    for j in range(len(finScanEstsX[:,0])):
      pkEntry = finScanEstsX[j]
      homeFound=False
      print("x=%d; j=%d"%(x,j))
      if scanCount<=10: #if we're trying to ID peaks in an early scan, we should first check if it's a "reflected" peak.
        if NearTheory(PI12ReflexDic, pkEntry, vlineArrayPI12Reflex, vlabelArray, electricLabel="PI12_REFLEX", predicUncerts=.5):
          homeFound=True
      if homeFound==False: #Only bother comparing to other predictions if we've already established that it wasn't a "reflected" peak.
        for i in range(len(idDicList)):
          #print("x=%d; j=%d; i=%d"%(x,j,i))
          if NearTheory(idDicList[i], pkEntry, predictionsList[i], vlabelArray, electricLabel=electronicLabelList[i], predicUncerts=predictionsListUncertainties[i]):
            print("Ayyy, peak identified! peak value = %.2f; electricLabel = "%pkEntry[0], electronicLabelList[i])
            homeFound=True
            break
      if homeFound==True:
        theoryList.append([scanCount, pkEntry[0], pkEntry[2]])
      else:
        print("This little peaky had no home :(", pkEntry)
        mysteryList.append([scanCount, pkEntry[0], pkEntry[2]])
  averagesFile1 = open("FitResults/AveragesFile_FromSigmas.txt","w+")
  averagesFile2 = open("FitResults/AveragesFile_FromSpreads.txt","w+")
  finalOutputFile1 = open("FitResults/FinalOutputFile_FromSigmas.txt","w+")
  finalOutputFile2 = open("FitResults/FinalOutputFile_FromSpreads.txt","w+")
  anyTransitionsIdentified = False
  observedTransitionsDic = {}
  finalOutputFile1.write("Transition\n"+10*"\t"+"v''->v'\t\tExp.(cm^-1)\n"+"-"*50+"\n")
  finalOutputFile2.write("Transition\n"+10*"\t"+"v''->v'\t\tExp.(cm^-1)\n"+"-"*50+"\n")

  for i in range(len(idDicList)):
        print(electronicLabelList[i]+" results:\n", idDicList[i])
        finalOutputFile1.write("-"*50+"\nSIGMA->"+electronicLabelList[i]+"\n")
        finalOutputFile2.write("-"*50+"\nSIGMA->"+electronicLabelList[i]+"\n")
        for transition in list(idDicList[i].keys()):
          if not(idDicList[i][transition]==[]):
            anyTransitionsIdentified = True
            transitionData = np.array(idDicList[i][transition])
            print("JulyTest1: i=%d, transition = ",transition)
            print("test: transitionData.shape = ", transitionData.shape)
            print("transitionData:\n", transitionData)
            #wStats = weightedStatistics(transitionData[:,0], transitionData[:,1], transitionLabel=electronicLabelList[i]+"_"+transition)
            #^ Previous line modified 19/July/2019 to replace traditional scat/stat uncerts with ranges of peak ests over diff rebin settings
            wStats1 = weightedStatistics(transitionData[:,0], transitionData[:,1], transitionLabel=electronicLabelList[i]+"_"+transition)
            observedTransitionsDic[electronicLabelList[i]+"_"+transition]  = wStats1
            averagesFile1.write("Transition: "+electronicLabelList[i]+" "+transition+"  Average Value: %.2f +/- %.5f"%wStats1)
            averagesFile1.write("\nConsistent Observations:\n")
            averagesFile1.write(np.array2string(np.array(idDicList[i][transition]), formatter={'float_kind':lambda k: "%.4f" % k}))
            averagesFile1.write("\n\n")
            finalOutputFile1.write("\t"*10+transition+"\t\t\t%.2f +/- %.5f\n"%wStats1)

            wStats2 = weightedStatistics(transitionData[:,0], transitionData[:,2], transitionLabel=electronicLabelList[i]+"_"+transition)
            observedTransitionsDic[electronicLabelList[i]+"_"+transition]  = wStats2
            averagesFile2.write("Transition: "+electronicLabelList[i]+" "+transition+"  Average Value: %.2f +/- %.5f"%wStats2)
            averagesFile2.write("\nConsistent Observations:\n")
            averagesFile2.write(np.array2string(np.array(idDicList[i][transition]), formatter={'float_kind':lambda k: "%.4f" % k}))
            averagesFile2.write("\n\n")
            finalOutputFile2.write("\t"*10+transition+"\t\t\t%.2f +/- %.5f\n"%wStats2)

  theoryList=np.array(theoryList); mysteryList=np.array(mysteryList)
  print("tests. theoryList:\n",theoryList,"\nmysteryList:\n",mysteryList)
  print("testy: observedTransitionsDic:\n", observedTransitionsDic)
  averagesFile1.write("\nObservations that weren't assigned to known transitions:\n")
  averagesFile1.write(np.array2string(mysteryList, formatter={'float_kind':lambda k: "%.2f" % k}))
  averagesFile1.close()
  finalOutputFile1.close()
  averagesFile2.write("\nObservations that weren't assigned to known transitions:\n")
  averagesFile2.write(np.array2string(mysteryList, formatter={'float_kind':lambda k: "%.2f" % k}))
  averagesFile2.close()
  finalOutputFile2.close()

  fig = plt.figure(3)
  fig.set_size_inches(20, 12)
  if theoryList.shape[0]>0:
    plt.errorbar(theoryList[:,1], theoryList[:,0], xerr=theoryList[:,2], fmt='o', color="green", label="Peaks that agree with prediction", markersize=6)
  if mysteryList.shape[0]>0:
    plt.errorbar(mysteryList[:,1], mysteryList[:,0], xerr=mysteryList[:,2], fmt='o', color="red", label="Peaks with no home :(", markersize=6)
  vLinePlotter(vlineArrayPI12, r'$^2\Sigma^{+}_{1/2} \rightarrow ^2\Pi_{1/2}$', reflections=True)
  vLinePlotter(vlineArrayDELTA32, r'$^2\Delta_{3/2}$')
  vLinePlotter(vlineArrayDELTA52, r'$^2\Delta_{5/2}$')
  if anyTransitionsIdentified:
    for transition in observedTransitionsDic:
      kPos=observedTransitionsDic[transition][0]; kSig=observedTransitionsDic[transition][1]
      print("kPos =",kPos,"; kSig =",kSig)
      plt.gca().add_patch(Rectangle((kPos-1*kSig, scanCountRange[0]-1), 2*kSig, scanCountRange[-1]-scanCountRange[0]+2, color="blue", alpha=.25))
      plt.axvline(x=kPos, ymin=0, ymax=1, color="blue", linestyle='-', linewidth=.5)
      plt.annotate("observedTransition\n"+transition, xy=(kPos, -.1), fontsize=14, ha='center', xycoords=('data','figure fraction'),color="blue")

  plt.xlim(plt.gca().get_xlim())
  box = plt.gca().get_position()
  plt.gca().set_position([box.x0, box.y0, box.width * 0.9, box.height])
  plt.legend(loc="center right", bbox_to_anchor=(1.25,.5), fontsize=10)
  plt.title("Compiling all Scanalyzer Results", fontsize=18)
  plt.xlabel(r'Wavenumber (cm$^{-1}$)', fontsize=18)
  plt.ylabel("Scan Index", fontsize=18)
  plt.yticks(ticks=scanCountRange, labels=dickeys[scanCountRange] )
  if not os.path.exists('FitResults'):
    os.mkdir('FitResults')
  plt.savefig("FitResults/CompiledScanalyzerResultsPlot.png")

  try:
    endTime = time.clock()
    print("Holy fudge... This actually took %f s to run :("%(endTime-startTime))
  except (ValueError, TypeError):
    print("Holy fudge... This actually took so long to run that %(endTime-startTime) doesn't even register as a number anymore...")

  if showFinalPlot:
    plt.show()