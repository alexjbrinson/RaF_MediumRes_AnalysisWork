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
from datetime import date
import numdifftools
import LoadingAndMungingData as lmd
import FrequencyConvertedUtilityKit as FCUK

if __name__ == '__main__':
  pd.options.mode.chained_assignment = None  # default='warn'
  rewrite=False; importDataFrames=True
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
  sigmaEst=.55; gammaEst=1.77; skew0=-2
  resolutionList=[.01,.02,.05,.1,.2,.5]
  isotopeDataFrame = pd.DataFrame(index=massList, columns = pd.MultiIndex.from_product([transitionLabels,estimationMethodLabels, estimationStatisticsLabels], names=['Transitions','Methods','Stats']) )
  #isotopeDataFrame=isotopeDataFrame.sort_index()
  if os.path.exists('./FitResults/OutputFiles/isotopeDataFrame.csv') and (importDataFrames==True):# and os.path.exists('./FitResults/OutputFiles/isoShiftsFrame.csv')
    print("ayyy! No computation necessary lol")
    isotopeDataFrame = pd.DataFrame(data=np.loadtxt('./FitResults/OutputFiles/isotopeDataFrame.csv'), index=massList, columns = pd.MultiIndex.from_product([transitionLabels,estimationMethodLabels, estimationStatisticsLabels], names=['Transitions','Methods','Stats']) )
  else:
    isotopeDataFrame = pd.DataFrame(index=massList, columns = pd.MultiIndex.from_product([transitionLabels,estimationMethodLabels, estimationStatisticsLabels], names=['Transitions','Methods','Stats']) )
    for m in massList:
      s=massScanDic[m]
      peakList=initCenterEsts
      finalfitResults = FCUK.Scanalyzer(m,s,rewrite=rewrite,peakList=peakList,peakSigmas=sigmaEst*np.ones_like(peakList),sameSigma=sameSigma,initGamma=gammaEst,resList=resolutionList, ltrim=ltrim, rtrim=rtrim, makePlots=True,sameSkew=True, useWeights=True, skew0=-4)
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
    np.savetxt('./FitResults/OutputFiles/isotopeDataFrame.csv',isotopeDataFrame.values)
    with open('./FitResults/OutputFiles/Isotope Dataframe .txt', 'w+') as isotopeTextFile:
      with pd.option_context('display.max_rows', 100, 'display.max_columns', 100): isotopeTextFile.write(str(isotopeDataFrame))
      isotopeTextFile.close()
  
  """Transition(S) frequencies plot to compare different isotopes and methods"""
  plt.figure('IsotopeDataFrame in plot form')
  plt.xlabel(r'Wavenumber $(cm^{-1})$', fontsize=18)
  for j in range(len(massList)):
    m=massList[j]
    offset = (j-math.floor(len(massList)/2))/(4*len(massList)) 
    color = next(plt.gca()._get_lines.prop_cycler)['color']
    for i in range(len(estimationMethodLabels)):
      if i==0: plt.errorbar(x=isotopeDataFrame.loc[m,idx[:,estimationMethodLabels[i],'mean']],y=i*np.ones_like(initCenterEsts)+offset, xerr=isotopeDataFrame.loc[m,idx[:,estimationMethodLabels[i],'range']],fmt="o",label=r'$^{%d}Ra^{19}F$'%(m-19), color=color)
      else: plt.errorbar(x=isotopeDataFrame.loc[m,idx[:,estimationMethodLabels[i],'mean']],y=i*np.ones_like(initCenterEsts)+offset, xerr=isotopeDataFrame.loc[m,idx[:,estimationMethodLabels[i],'range']],fmt="o", color=color)
  plt.gcf().set_size_inches(20, 12)
  plt.title("Investigating Transition Frequency dependence on isotope number in RaF\n(3 methods for peak identification considered)", fontsize=24)
  plt.yticks(ticks=range(len(estimationMethodLabels)),labels=estimationMethodLabels, fontsize=18)
  plt.legend(loc='best', fontsize=18)
  plt.savefig('./FitResults/IsotopeDataFrameSummaryPlot')
  plt.close()

  """Now converting to shift data:"""
  referenceFrame = isotopeDataFrame.loc[245,idx[:,:,['mean','range']]].copy()#hehe "reference frame". Was not intentional lol
  isoShiftsFrame=pd.DataFrame(index=massList, columns = pd.MultiIndex.from_product([transitionLabels,estimationMethodLabels, ['shift','error']], names=['Shifts','Methods','Stats']) )
  for m in massList:
    isoShiftsFrame.loc[m,idx[:,:,'shift']]=(isotopeDataFrame.loc[m,idx[:,:,'mean']] - referenceFrame.loc[idx[:,:,"mean"]]).values
    errorSquaror = np.square(isotopeDataFrame.loc[m,idx[:,:,'range']]) + np.square(referenceFrame.loc[idx[:,:,"range"]])
    isoShiftsFrame.loc[m,idx[:,:,'error']]=np.sqrt(errorSquaror.astype(np.float64)).values
  np.savetxt('./FitResults/OutputFiles/isoShiftsFrame.csv', isoShiftsFrame.values)
  with pd.option_context('display.max_rows', 100, 'display.max_columns', 30):print("Isotope Shifts:\n",isoShiftsFrame)
  with open('./FitResults/OutputFiles/Isotope Shifts Dataframe.txt', 'w+') as isoShiftsTextFile:
    with pd.option_context('display.max_rows', 100, 'display.max_columns', 100): isoShiftsTextFile.write(str(isoShiftsFrame))
    isoShiftsTextFile.close()
  
  """shift vs mass plots to compare methods"""
  for i in range(len(transitionLabels)):
    transition=transitionLabels[i]
    plt.figure(transition)
    plt.gcf().set_size_inches(20, 12)
    plt.title(r"Isotope Shift vs. Mass for the %d'' $\rightarrow$ %d' Transition in RaF"%(i, i)+"\n(%d methods for peak identification considered)"%len(estimationMethodLabels), fontsize=24)
    for j in range(len(estimationMethodLabels)):
      meth = estimationMethodLabels[j]
      color = next(plt.gca()._get_lines.prop_cycler)['color']
      offset = (j-math.floor(len(estimationMethodLabels)/2))/(4*len(estimationMethodLabels)) 
      plt.errorbar(x=(massList-245)+offset, y=isoShiftsFrame.loc[:,idx[transition,meth,'shift']], yerr=isoShiftsFrame.loc[:,idx[transition,meth,'error']],fmt="o",label=meth, color=color, markersize=8)
    plt.xlabel('mass diffs (amu)', fontsize=18)
    plt.ylabel(r'shift $(cm^{-1})$', fontsize=18)
    plt.legend(loc=3, fontsize=18, ncol=len(estimationMethodLabels))
    plt.savefig('./FitResults/IsotopeShiftDiffMethods_%s-%s.png'%(i,i))
    plt.close()

  """shift vs mass plots to compare transitions"""
  for meth in estimationMethodLabels:
    plt.figure(meth)
    plt.gcf().set_size_inches(20, 12)
    plt.title("Isotope Shift vs. Mass for %s Estimation method in RaF\n(4 transitions included)"%meth, fontsize=24)
    for j in range(len(transitionLabels)):
      transition = transitionLabels[j]
      color = next(plt.gca()._get_lines.prop_cycler)['color']
      offset = (j-math.floor(len(transitionLabels)/2))/(4*len(transitionLabels)) 
      plt.errorbar(x=(massList-245)+offset, y=isoShiftsFrame.loc[:,idx[transition,meth,'shift']], yerr=isoShiftsFrame.loc[:,idx[transition,meth,'error']],fmt="o",label=r"$%d \rightarrow %d$"%(j,j), color=color, markersize=8)
    plt.xlabel('mass diffs (amu)', fontsize=18)
    plt.ylabel(r'shift $(cm^{-1})$', fontsize=18)
    plt.legend(loc=3, fontsize=18, ncol=len(transitionLabels))
    plt.savefig('./FitResults/IsotopeShiftDiffTransitions_%s.png'%meth)
    plt.close()
  importTestFrame=pd.DataFrame(data=np.loadtxt('./FitResults/OutputFiles/isoShiftsFrame.csv'), index=massList, columns = pd.MultiIndex.from_product([transitionLabels,estimationMethodLabels, ['shift','error']], names=['Shifts','Methods','Stats']) )
  print("Testing data frame importability. importTestFrame==isoShiftsFrame: ", np.all((importTestFrame-isoShiftsFrame).values==0))

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

