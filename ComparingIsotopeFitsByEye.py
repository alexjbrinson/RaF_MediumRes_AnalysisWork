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
import FrequencyConvertedUtilityKitAlternate as fcuk

def makePlot1(datFrame, mass, s, r, n, fitRes, color1, color2, redchi=-1):
  scan = str(s)
  #print("making res=%.3f, n=%d plot for mass%d scan %s\nrandom subsample # %d"%(r,n,mass,scan,k))
  fitDic = fitRes.best_values
  #fitPlotFig = plt.figure()
  plt.gcf().set_size_inches(20, 12)
  ax = plt.gca()
  sigTot = np.sum(datFrame.loc[:,'signal_value'])
  print("test: sigTot = ",sigTot)
  xDat = np.array(datFrame.loc[:,'wavenumber_mean']); yDat = np.array(datFrame.loc[:,'signal_value'])/sigTot; ySigDat = np.array(datFrame.loc[:,'signal_uncertainty']/sigTot)
  xFull = np.arange(np.min(xDat)-.05,np.max(xDat)+.05,.01)
  plt.errorbar(xDat, yDat, yerr=ySigDat, fmt='.-',ecolor='k',color=color1, alpha=.3, label='A='+str(mass-19))
  plt.fill_between(xDat, yDat,color=color1, alpha=.25)
  plt.xlabel(r'Wavenumber (cm$^{-1}$)', fontsize=18)
  plt.ylabel('rate (counts/s)', fontsize=18)
  plt.title(r'$^{A}$Ra$^{19}$F Spectrum; Resolution = %.2f' %r, fontsize=24)
  #plt.plot(xFull, fitRes.eval(x=xFull,params=fitRes.init_params), 'k--', label="init_Fit", linewidth=2)
  plt.plot(xFull, fitRes.eval(x=xFull,params=fitRes.init_params)/sigTot, '--', color=color2, label="init_Fit", linewidth=2)
  plt.plot(xFull, fitRes.eval(x=xFull,params=fitRes.params)/sigTot, '-', color=color2, label="best_fit",linewidth=4)

def makePlot2(datFrame, mass, s, r, n, polyCompsDat, color1, color2):
  scan = str(s)
  #print("making res=%.3f, n=%d plot for mass%d scan %s\nrandom subsample # %d"%(r,n,mass,scan,k))
  #fitDic = fitRes.best_values
  #fitPlotFig = plt.figure()
  plt.gcf().set_size_inches(20, 12)
  ax = plt.gca()
  sigTot = np.sum(datFrame.loc[:,'signal_value'])
  xDat = np.array(datFrame.loc[:,'wavenumber_mean']); yDat = np.array(datFrame.loc[:,'signal_value'])/sigTot; ySigDat = np.array(datFrame.loc[:,'signal_uncertainty'])/sigTot
  plt.errorbar(xDat, yDat, yerr=ySigDat, fmt='.-',ecolor='k',color=color1, alpha=.3, label='A='+str(mass-19))
  plt.fill_between(xDat, yDat,color=color1, alpha=.25)
  plt.xlabel(r'Wavenumber (cm$^{-1}$)', fontsize=18)
  plt.ylabel('rate (counts/s)', fontsize=18)
  plt.title(r'$^{A}$Ra$^{19}$F Spectrum; Resolution = %.2f' %r, fontsize=24)
  #plt.plot(xFull, fitRes.eval(x=xFull,params=fitRes.init_params), 'k--', label="init_Fit", linewidth=2)

  for i in range(len(polyCompsDat)):
    polyData=polyCompsDat[i]; xPolyVals = polyData[:,0] ; yPolyVals=polyData[:,1]/sigTot
    arg_max=np.argmax(yPolyVals); y_max=yPolyVals[arg_max] ; x_max = xPolyVals[arg_max]
    if i==0: plt.plot(xPolyVals, yPolyVals, '-', color=color2, linewidth=3, label = 'piecewise polynomial fit results')
    else: plt.plot(xPolyVals, yPolyVals, '-', color=color2, linewidth=3)
    plt.plot(x_max,y_max, "o", color=color2, markersize=10)
    plt.annotate(s=r'max$_{poly}\nu_%d = %.2f$'%(i, x_max), xy=(x_max, y_max*.95), fontsize=14, ha='center', va='top', xycoords=('data','data'),color=color2)
    datMaxEsts = fcuk.findLocalMax(datFrame, x_max, .5, uncertIndex=3)
    plt.errorbar(datMaxEsts[0,0], datMaxEsts[1,0]/sigTot, xerr=datMaxEsts[0,1], yerr=datMaxEsts[1,1]/sigTot, fmt='o', color=color1, ecolor='k', alpha=.5)
    plt.annotate(s=r'max$_{dat}\nu_%d = %.2f$'%(i, datMaxEsts[0,0]), xy=(datMaxEsts[0,0], datMaxEsts[1,0]*1.05/sigTot), fontsize=14, ha='center', va='bottom', xycoords=('data','data'),color=color1)

def makePlot3(datFrame, mass, s, r, n, fitRes, polyOrdersList, peaksList):
  scan = str(s)
  #print("making res=%.3f, n=%d plot for mass%d scan %s\nrandom subsample # %d"%(r,n,mass,scan,k))
  #fitDic = fitRes.best_values
  #fitPlotFig = plt.figure()
  plt.gcf().set_size_inches(20, 12)
  ax = plt.gca()
  sigTot = np.sum(datFrame.loc[:,'signal_value'])
  xDat = np.array(datFrame.loc[:,'wavenumber_mean']); yDat = np.array(datFrame.loc[:,'signal_value'])/sigTot; ySigDat = np.array(datFrame.loc[:,'signal_uncertainty'])/sigTot
  xFull = np.arange(np.min(xDat)-.05,np.max(xDat)+.05,.01)
  plt.errorbar(xDat, yDat, yerr=ySigDat, fmt='.-',ecolor='k',color="blue", alpha=.3, label='Data')
  plt.fill_between(xDat, yDat,color="blue", alpha=.25)
  plt.xlabel(r'Wavenumber (cm$^{-1}$)', fontsize=18)
  plt.ylabel('rate (counts/s)', fontsize=18)
  plt.title(r'$^{A}$Ra$^{19}$F Spectrum; Resolution = %.2f' %r, fontsize=24)
  #plt.plot(xFull, fitRes.eval(x=xFull,params=fitRes.init_params), 'k--', label="init_Fit", linewidth=2)
  #plt.plot(xFull, fitRes.eval(x=xFull,params=fitRes.init_params)/sigTot, 'k--', label="init_Fit", linewidth=2)
  plt.plot(xFull, fitRes.eval(x=xFull,params=fitRes.params)/sigTot, 'k-', label="lmfit best_fit",linewidth=4)
  count=0
  for pOrd in polyOrdersList:
    count+=1
    color = next(ax._get_lines.prop_cycler)['color']
    for i in range(n):
      datMaxEsts = fcuk.findLocalMax(datFrame, peaksList[i], .5, uncertIndex=3)
      plt.errorbar(datMaxEsts[0,0], datMaxEsts[1,0]/sigTot, xerr=datMaxEsts[0,1], yerr=datMaxEsts[1,1]/sigTot, fmt='o', color="blue", ecolor='k', alpha=.5)
      plt.annotate(s=r'max$_{dat}\nu_%d = %.2f$'%(i, datMaxEsts[0,0]), xy=(datMaxEsts[0,0], datMaxEsts[1,0]*1.05/sigTot), fontsize=14, ha='center', va='bottom', xycoords=('data','data'),color=color)
      polyFitDat = fcuk.findPolyMax(datFrame, datMaxEsts[0,0]+np.array([-4,+1]), datMaxEsts[0,0]+np.array([-1.5,+.75]), polyOrder=pOrd)
      polyData=polyFitDat[1]; xPolyVals = polyData[:,0] ; yPolyVals=polyData[:,1]/sigTot
      arg_max=np.argmax(yPolyVals); y_max=yPolyVals[arg_max] ; x_max = xPolyVals[arg_max]
      if i==0: plt.plot(xPolyVals, yPolyVals, '-', color=color, linewidth=3, label = '%dth order'%pOrd)
      else: plt.plot(xPolyVals, yPolyVals, '-', color=color, linewidth=3)
      plt.plot(x_max,y_max, "o", color=color, markersize=10)
      plt.annotate(s=r'max$_{poly}\nu_%d = %.2f$'%(i, x_max), xy=(x_max, y_max*.95), fontsize=14, ha='center', va='top', xycoords=('data','data'),color=color)
      plt.annotate(s=r'\chi^2_{\text{red}} = '+'%.3f'%polyFitDat[2], xy=(x_max, datMaxEsts[1,0]*(9-count)/10), fontsize=14, ha='center', va='top', xycoords=('data','data'),color=color)
      pollyFunc=polyFitDat[3] ; g = datMaxEsts[0,0]+(.75-1.5)/2; xSemiFull=datMaxEsts[0,0]+np.arange(-4, 1,.01)-g; ySemiFull = pollyFunc(xSemiFull)/sigTot; xSemiFull+=g
      plt.plot(xSemiFull, ySemiFull, '--', color=color)

def makePlot4(datFrame, mass, s, r, n, fitRes, polyOrdersList, peaksList):
  scan = str(s)
  #print("making res=%.3f, n=%d plot for mass%d scan %s\nrandom subsample # %d"%(r,n,mass,scan,k))
  #fitDic = fitRes.best_values
  #fitPlotFig = plt.figure()
  plt.gcf().set_size_inches(20, 12)
  ax = plt.gca()
  sigTot = np.sum(datFrame.loc[:,'signal_value'])
  xDat = np.array(datFrame.loc[:,'wavenumber_mean']); yDat = np.array(datFrame.loc[:,'signal_value'])/sigTot; ySigDat = np.array(datFrame.loc[:,'signal_uncertainty'])/sigTot
  xFull = np.arange(np.min(xDat)-.05,np.max(xDat)+.05,.01)
  plt.errorbar(xDat, yDat, yerr=ySigDat, fmt='.-',ecolor='k',color="blue", alpha=.3, label='Data')
  plt.fill_between(xDat, yDat,color="blue", alpha=.25)
  plt.xlabel(r'Wavenumber (cm$^{-1}$)', fontsize=18)
  plt.ylabel('rate (counts/s)', fontsize=18)
  plt.title(r'$^{A}$Ra$^{19}$F Spectrum; Resolution = %.2f' %r, fontsize=24)
  #plt.plot(xFull, fitRes.eval(x=xFull,params=fitRes.init_params), 'k--', label="init_Fit", linewidth=2)
  #plt.plot(xFull, fitRes.eval(x=xFull,params=fitRes.init_params)/sigTot, 'k--', label="init_Fit", linewidth=2)
  plt.plot(xFull, fitRes.eval(x=xFull,params=fitRes.params)/sigTot, 'k-', label="lmfit best_fit",linewidth=4)
  count=0
  for pOrd in polyOrdersList:
    count+=1
    color = next(ax._get_lines.prop_cycler)['color']
    for i in range(n):
      datMaxEsts = fcuk.findLocalMax(datFrame, peaksList[i], .5, uncertIndex=3)
      plt.errorbar(datMaxEsts[0,0], datMaxEsts[1,0]/sigTot, xerr=datMaxEsts[0,1], yerr=datMaxEsts[1,1]/sigTot, fmt='o', color="blue", ecolor='k', alpha=.5)
      plt.annotate(s=r'max$_{dat}\nu_%d = %.2f$'%(i, datMaxEsts[0,0]), xy=(datMaxEsts[0,0], datMaxEsts[1,0]*1.05/sigTot), fontsize=14, ha='center', va='bottom', xycoords=('data','data'),color=color)
      polyFitDat = fcuk.findPolyMax(datFrame, datMaxEsts[0,0]+np.array([-4,+1]), datMaxEsts[0,0]+np.array([-1.5,+.75]), polyOrder=pOrd)
      polyData=polyFitDat[1]; xPolyVals = polyData[:,0] ; yPolyVals=polyData[:,1]/sigTot
      arg_max=np.argmax(yPolyVals); y_max=yPolyVals[arg_max] ; x_max = xPolyVals[arg_max]
      if i==0: plt.plot(xPolyVals, yPolyVals, '-', color=color, linewidth=3, label = '%dth order'%pOrd)
      else: plt.plot(xPolyVals, yPolyVals, '-', color=color, linewidth=3)
      plt.plot(x_max,y_max, "o", color=color, markersize=10)
      plt.annotate(s=r'max$_{poly}\nu_%d = %.2f$'%(i, x_max), xy=(x_max, y_max*.95), fontsize=14, ha='center', va='top', xycoords=('data','data'),color=color)
      #plt.annotate(s=r'\chi^2_{\text{red}} = '+'%.3f'%polyFitDat[2], xy=(x_max, datMaxEsts[1,0]*(9-count)/10), fontsize=14, ha='center', va='top', xycoords=('data','data'),color=color)
      #pollyFunc=polyFitDat[3] ; g = datMaxEsts[0,0]+(.75-1.5)/2; xSemiFull=datMaxEsts[0,0]+np.arange(-4, 1,.01)-g; ySemiFull = pollyFunc(xSemiFull)/sigTot; xSemiFull+=g
      #plt.plot(xSemiFull, ySemiFull, '--', color=color)

      polyFitDat2 = fcuk.findPolyMax(datFrame, datMaxEsts[0,0]+np.array([-4,+1]), datMaxEsts[0,0]+np.array([-1.5,+.75]), weights=False, polyOrder=pOrd)
      polyData2=polyFitDat2[1]; xPolyVals2 = polyData2[:,0] ; yPolyVals2=polyData2[:,1]/sigTot
      arg_max2=np.argmax(yPolyVals2); y_max2=yPolyVals2[arg_max2] ; x_max2 = xPolyVals2[arg_max2]
      if i==0: plt.plot(xPolyVals2, yPolyVals2, '--', color=color, linewidth=3, label = '%dth order, no weights'%pOrd)
      else: plt.plot(xPolyVals2, yPolyVals2, '--', color=color, linewidth=3)
      plt.plot(x_max2,y_max2, "o", color=color, markersize=10, alpha=.25)
      plt.annotate(s=r'max$_{poly, n.w.}\nu_%d = %.2f$'%(i, x_max2), xy=(x_max2, y_max2*.95), fontsize=14, ha='center', va='top', xycoords=('data','data'),color=color)
      #plt.annotate(s=r'\chi^2_{\text{red}} = '+'%.3f'%polyFitDat[2], xy=(x_max, datMaxEsts[1,0]*(9-count)/10), fontsize=14, ha='center', va='top', xycoords=('data','data'),color=color)

if __name__ == '__main__':
  pd.options.mode.chained_assignment = None  # default='warn' (Pandas keep harassing me and I'm doing nothing wrong!)
  
  massList=np.array([245]) #np.array([242,243,244,245, 247])
  rewrite=True; ltrim=13256.5; rtrim=13287; smoothingWidth=0; pOrder=16; polyOrdersList=[5,7,10,16,20]
  sameSigma=False; sameSkew=False; sameGamma=False
  sigmaEst=.5; gammaEst=2; skew0=-2.6; skewList=[-2.5,-2,-2,-2]
  resolutionList= [.03] #[.03,.05,.07,.1,.2,.3] #  struggling a bit with low count statistics for ^{224,225}RaF
  fitMethod="leastsq"

  allScansBigDic = {}
  for m in massList: allScansBigDic[m] = lmd.makeScanToWavemeterDic(m, redo=False, verbose=False)
  colorDict1={242:'red', 243:'orange',244:'green',245:'blue',247:'purple'}
  colorDict2={242:'peru', 243:'gold',244:'lime',245:'cyan',247:'magenta'}
  massScanDic={}
  massScanDic[242]=[[2312, 2313]] #These are good individually and combined!
  massScanDic[243]=[[2302,2303,2308]]#,2283]#2300? #2283(from a different time, when signals weren't as strong. use in future), 2301("wavemeter stopped working in the middle of a peak" + relatively weak signal)
  massScanDic[244]=[[2304,2305,2306]]#,2307] #possibly remove 2307?
  massScanDic[245]=[[2309,2310,2320]]#,[2341,2349],2350]#, [2341(75mW),2349],2350("re-tuned TiSa overlap upstairs --> additional 50% improvement")]#,[2346] is a pdl scan though. gross...#2178 is Ti:Sa, but actually gross af#
  massScanDic[247]=[[2311,2322]]#,[2188,2190]]#(also decent, but from diff era with different rates)
  initCenterEsts={}
  initCenterEsts[242]=[13285,13278.86,13272.76,13266.76]#,13260.35]
  initCenterEsts[243]=[13284.9,13278.75,13272.61,13266.61]#,13260.35]
  initCenterEsts[244]=[13284.88,13278.9,13272.9,13266.7]#,13260.35]  #[13284.84,13278.68,13272.59,13266.43]
  initCenterEsts[245]=[13284.73,13278.57,13272.46,13266.45]#,13260.35]
  initCenterEsts[247]=[13284.63,13278.5,13272.4,13266.1]#,13260.35]#[13284.53,13278.41,13272.24,13266.05]
  
  for m in massList:
    mass = m
    scanListsList = massScanDic[m]
    s=scanListsList[0]
    peaksList = initCenterEsts[m]
    N = len(peaksList)
    for r in resolutionList:
      print("m=%d, scans:%s, peaksList:%s"%(m,str(s),str(peaksList)))
      scanFrame = lmd.mergeDatRaw(m, s)
      datFrame=lmd.smoother(lmd.makeUseable(scanFrame, resolution=r,cropSparseEnds=True, noNaNsense=True, ltrim=ltrim, rtrim=rtrim, verbose=False), smoothWidth=smoothingWidth)#, normalizedOn="Integral"
      (fitRes, warningStatus, numPeaksUsed) = fcuk.fitNPeaks(datFrame, peaksList, peakSigmas=sigmaEst, initGamma=gammaEst, useWeights=True, sameSkew=sameSkew, skewList=skewList,skew0=skew0, sameSigma=sameSigma, sameGamma=sameGamma,linearTerm=False,method=fitMethod)#add other opts?
      if warningStatus == -1:
        print("That's it for this scan, boys. Don't. push. these peaks. They're. close. to. the. eeeedge. (One of the peaks is leaking out of the scan window at rebin setting%d)"%r)
      #fitReportFile = open(resolutionPath+'randSamp%d_FitReport.txt'%k,'w+')
      #fitReportFile.write("Fit Report for: Mass = %d, Scan = %s, Resolution = %.3f, randSamp#%d\n"%(mass,scan,r,k)); fitReportFile.write(fitRes.fit_report(min_correl=0.25)); fitReportFile.close()
      #compiledFitResults[r,k] = fitRes.best_values
      #compiledResults.loc[(r,k),"fitResults"] = [fitRes.best_values]
      print("Just ran fitScanX routine for: scan %s; res = %.3f; N=%d, and found redchi = %f"%(s,r,numPeaksUsed,fitRes.redchi))
      fitCenterEsts = np.full((N,2),np.nan); fitLocalMaxEsts = np.full((N,2),np.nan);
      datLocalMaxEsts = np.full((N,2),np.nan); polyFitMaxEsts = np.full((N,2),np.nan);
      parmCenterNames = ['sv'+str(j)+'_center' for j in range(N)].append(['l0_slope', 'l0_intercept'])
      kwargs = {'p_names':parmCenterNames}
      #print("fitRes.errorbars", fitRes.errorbars)
      xDat = np.array(datFrame.loc[:,'wavenumber_mean']); xFull = np.arange(np.min(xDat)-.05,np.max(xDat)+.05,.001)
      comps = fitRes.eval_components(x=xFull)
      redχsqMatrix=-1*np.ones((N,len(polyOrdersList)))
      polyMaxWeightMatrix=-1*np.ones((N,len(polyOrdersList)))
      polyMaxUnWeightMatrix=-1*np.ones((N,len(polyOrdersList)))
      polyCompsDat=[]
      for p in range(N):
        if p in range(numPeaksUsed):
          mu_p = fitRes.best_values['sv'+str(p)+'_center'];
          sigma_p = fitRes.best_values['sv'+str(p)+'_sigma']; gamma_p = fitRes.best_values['sv'+str(p)+'_gamma'];
          mu_pUncert = fitRes.params['sv'+str(p)+'_center'].stderr if fitRes.errorbars else max(sigma_p, gamma_p)
          sigma_p = fitRes.best_values['sv'+str(p)+'_sigma'];
          gamma_p = fitRes.best_values['sv'+str(p)+'_gamma'];
          skew_p = fitRes.best_values['sv'+str(p)+'_skew'];
          xSimp = np.arange(mu_p-(sigma_p+gamma_p),mu_p+(sigma_p+gamma_p),.01); ySimp=fcuk.skewedVoigt(xSimp,1,mu_p,sigma_p,gamma_p,skew_p)
          fitLocMax_p = xSimp[np.argmax(ySimp)]; #WTF? Is my skewedVoigt function not good enough for you, bitch!? (p.s. it looks like no, it's not...)
          fitLocMax_p=xFull[np.argmax(comps['sv'+str(p)+'_'])];
          datLocalEst_p = fcuk.findLocalMax(datFrame, fitLocMax_p, 0.5, uncertIndex=3)
          for j in range(len(polyOrdersList)):
            datPolyEst_p = fcuk.findPolyMax(datFrame, [peaksList[p]-4,peaksList[p]+1], [fitLocMax_p-1.5,fitLocMax_p+.75], polyOrder=polyOrdersList[j])
            #fitCenterEsts[p, 0] = mu_p; fitCenterEsts[p,1] = mu_pUncert
            #fitLocalMaxEsts[p,0] = fitLocMax_p; fitLocalMaxEsts[p,1] = mu_pUncert
            #datLocalMaxEsts[p,0] = datLocalEst_p[0,0] ; datLocalMaxEsts[p,1] = datLocalEst_p[0,1]
            #polyFitMaxEsts[p,0] = datPolyEst_p[0][0,0] ; polyFitMaxEsts[p,1] = datPolyEst_p[0][0,1]
            polyCompsDat.append(datPolyEst_p[1])
            redχsqMatrix[p,j]=datPolyEst_p[2]
            polyMaxWeightMatrix[p,j]=datPolyEst_p[0][0,0]
            datPolyEst_p2 = fcuk.findPolyMax(datFrame, [peaksList[p]-4,peaksList[p]+1], [fitLocMax_p-1.5,fitLocMax_p+.75], polyOrder=polyOrdersList[j], weights=False)
            polyMaxUnWeightMatrix[p,j]=datPolyEst_p2[0][0,0]
            #print("mass %d, peak %d, polyOrder %d, reduced Chi squared = "%(m,p,polyOrdersList[j]),datPolyEst_p[2])
        else:
          datLocalEst_p = fcuk.findLocalMax(datFrame, peaksList[p], 0.75, uncertIndex=3)
          for j in range(len(polyOrdersList)):
            datPolyEst_p = fcuk.findPolyMax(datFrame, [peaksList[p]-4,peaksList[p]+1], [datLocalEst_p[0,0]-1.5,datLocalEst_p[0,0]+.75], polyOrder=polyOrdersList[j], verbose=False)
            #datLocalMaxEsts[p,0] = datLocalEst_p[0,0] ; datLocalMaxEsts[p,1] = datLocalEst_p[0,1]
            #polyFitMaxEsts[p,0] = datPolyEst_p[0][0,0] ; polyFitMaxEsts[p,1] = datPolyEst_p[0][0,1]
            polyCompsDat.append(datPolyEst_p[1])
            redχsqMatrix[p,j]=datPolyEst_p[2]
            polyMaxWeightMatrix[p,j]=datPolyEst_p[0][0,0]
            datPolyEst_p2 = fcuk.findPolyMax(datFrame, [peaksList[p]-4,peaksList[p]+1], [datLocalEst_p[0,0]-1.5,datLocalEst_p[0,0]+.75], polyOrder=polyOrdersList[j], weights=False)
            polyMaxUnWeightMatrix[p,j]=datPolyEst_p2[0][0,0]
            #print("mass %d, peak %d, polyOrder %d, reduced Chi squared = "%(m,p,polyOrdersList[j]),datPolyEst_p[2])

      redχsqFrame=pd.DataFrame(data=redχsqMatrix,index=["0->0","1->1","2->2","3->3"], columns=polyOrdersList)
      polyMaxWeightFrame=pd.DataFrame(data=polyMaxWeightMatrix,index=["0->0","1->1","2->2","3->3"], columns=polyOrdersList)
      polyMaxUnWeightFrame=pd.DataFrame(data=polyMaxUnWeightMatrix,index=["0->0","1->1","2->2","3->3"], columns=polyOrdersList)
      polyMaxDifferenceFrame=pd.DataFrame(data=polyMaxWeightMatrix-polyMaxUnWeightMatrix,index=["0->0","1->1","2->2","3->3"], columns=polyOrdersList)
      print("m = %d, reduced chi squared matrix:\n"%m, redχsqFrame)
      print("m = %d, local max matrix w/ weighting:\n"%m, polyMaxWeightFrame)
      #print("m = %d, local max matrix w/o weighting:\n"%m, polyMaxUnWeightFrame)
      print("m = %d, local max difference matrix, w - w/o weighting:\n"%m, polyMaxDifferenceFrame)
      #plt.figure(1)
      #makePlot1(datFrame, m, s, r, numPeaksUsed, fitRes, colorDict1[m], colorDict2[m], redchi=fitRes.redchi)
      #plt.figure(2)
      #makePlot2(datFrame, m, s, r, numPeaksUsed, polyCompsDat, colorDict1[m], colorDict2[m])
      if m == 245:
        #plt.figure(1)
        #makePlot3(datFrame, m, s, r, N, fitRes, polyOrdersList, peaksList)
        plt.figure(1)
        makePlot4(datFrame, m, s, r, N, fitRes, polyOrdersList, peaksList)
  plt.figure(1)
  plt.legend()
  plt.show()

