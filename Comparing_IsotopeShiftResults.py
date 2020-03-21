import math
import numpy as np
import pandas as pd
import os.path
import json
import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches
#from matplotlib.patches import Rectangle
#import numdifftools
import time
from datetime import date
import LoadingAndMungingData as lmd
import FrequencyConvertedUtilityKit as FCUK

def makeShiftFrame(isoDopeDF, refMass=245, errorsFrom='stderr', fileWrite=True, extraMeths=[]):
  defaultMeths = isoDopeDF.columns.unique(level='Methods').values
  if type(extraMeths)==str: methLabs=np.append(defaultMeths, extraMeths)
  elif type(extraMeths)==list and len(extraMeths)==0: methLabs = defaultMeths
  tLabs = isoDopeDF.columns.unique(level='Transitions').values;
  isoShiftsFrame=pd.DataFrame(index=isoDopeDF.index, columns = pd.MultiIndex.from_product([tLabs, methLabs, np.array(['shift','uncertainty'])], names=['Transitions','Methods','Stats']) )
  referenceFrame = isoDopeDF.loc[refMass,idx[:,:,['mean',errorsFrom]]].copy()#hehe "reference frame". Was not intentional lol
  for m in massList:
    isoShiftsFrame.loc[m,idx[:,defaultMeths,'shift']]=(isoDopeDF.loc[m,idx[:,:,'mean']] - referenceFrame.loc[idx[:,:,"mean"]]).values
    errorSquaror = np.square(isoDopeDF.loc[m,idx[:,:,errorsFrom]]) + np.square(referenceFrame.loc[idx[:,:,errorsFrom]])
    if m==245: isoShiftsFrame.loc[m,idx[:,defaultMeths,'uncertainty']]=np.zeros_like(errorSquaror.astype(np.float64))
    else: isoShiftsFrame.loc[m,idx[:,defaultMeths,'uncertainty']]=np.sqrt(errorSquaror.astype(np.float64)).values
  if fileWrite:  
    np.savetxt('./FitResults/OutputFiles/isoShiftsFrame.csv', isoShiftsFrame.values)
    with open('./FitResults/OutputFiles/Isotope Shifts Dataframe.txt', 'w+') as isoShiftsTextFile:
      with pd.option_context('display.max_rows', 100, 'display.max_columns', 310): isoShiftsTextFile.write(str(isoShiftsFrame))
      isoShiftsTextFile.close()
  return(isoShiftsFrame)

def shiftVsMassPlotter_OneTransitionSlice(shiftsFrame, i, saveFig=True, closeFig=True):
  """shift vs mass plots to compare methods"""
  mList=np.array(shiftsFrame.index.values); tLabs=shiftsFrame.columns.unique(level='Transitions').values; methLabs=shiftsFrame.columns.unique(level='Methods').values
  transition=tLabs[i]
  plt.figure(transition)
  plt.gcf().set_size_inches(20, 12)
  plt.title(r"Isotope Shift vs. Mass for the %d'' $\rightarrow$ %d' Transition in RaF"%(i, i)+"\n(%d methods for peak identification considered)"%len(methLabs), fontsize=24)
  for j in range(len(methLabs)): #if it weren't for the damn offset, I could've written this loop as "for meth in methLabs:" sigh...
    meth = methLabs[j]
    color = next(plt.gca()._get_lines.prop_cycler)['color']
    offset = (j-math.floor(len(methLabs)/2))/(4*len(methLabs)) 
    plt.errorbar(x=(mList-245)+offset, y=shiftsFrame.loc[:,idx[transition,meth,'shift']], yerr=shiftsFrame.loc[:,idx[transition,meth,'uncertainty']],fmt="o",label=meth, color=color, markersize=8)
  plt.xlabel('Mass Difference (amu)', fontsize=18)
  plt.ylabel(r'shift $(cm^{-1})$', fontsize=18)
  plt.legend(loc=3, fontsize=24, ncol=len(methLabs))
  if saveFig==True: plt.savefig('./FitResults/IsotopeShiftDiffMethods_%s-%s.png'%(i,i))
  if closeFig: plt.close()

def shiftVsMassPlotter_OneMethodSlice(shiftsFrame, i, saveFig=True, closeFig=True):
  """shift vs mass plots to compare methods"""
  mList=np.array(shiftsFrame.index.values); tLabs=shiftsFrame.columns.unique(level='Transitions').values; methLabs=shiftsFrame.columns.unique(level='Methods').values
  meth=methLabs[i]
  plt.figure(meth)
  plt.gcf().set_size_inches(20, 12)
  plt.title("Isotope Shift vs. Mass for %s Estimation method in RaF\n(4 transitions included)"%meth, fontsize=24)
  for j in range(len(tLabs)): #if it weren't for the damn offset, I could've written this loop as "for meth in methLabs:" sigh...
    transition = tLabs[j]
    color = next(plt.gca()._get_lines.prop_cycler)['color']
    offset = (j-math.floor(len(tLabs)/2))/(4*len(tLabs)) 
    plt.errorbar(x=(mList-245)+offset, y=shiftsFrame.loc[:,idx[transition,meth,'shift']], yerr=shiftsFrame.loc[:,idx[transition,meth,'uncertainty']],fmt="o",label=r"$%d \rightarrow %d$"%(j,j), color=color, markersize=8)
  plt.xlabel('Mass Difference (amu)', fontsize=18)
  plt.ylabel(r'shift $(cm^{-1})$', fontsize=18)
  plt.legend(loc=3, fontsize=24, ncol=len(tLabs))
  if saveFig==True: plt.savefig('./FitResults/IsotopeShiftDiffTransitions_%s.png'%meth)
  if closeFig: plt.close()


'''4. "Make a table of "isotope shifts", comparing differences between different isotopes and using the same electronic transition"'''
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
  print("testss:\n",isotopeDataFrame.index,isotopeDataFrame.columns.unique('Transitions').values)
  #isotopeDataFrame=isotopeDataFrame.sort_index()
  if os.path.exists('./FitResults/OutputFiles/isotopeDataFrame.csv') and (importDataFrames==True):# and os.path.exists('./FitResults/OutputFiles/isoShiftsFrame.csv')
    print("ayyy! No computation necessary lol")
    isotopeDataFrame = pd.DataFrame(data=np.loadtxt('./FitResults/OutputFiles/isotopeDataFrame.csv'), index=massList, columns = pd.MultiIndex.from_product([transitionLabels,estimationMethodLabels, estimationStatisticsLabels], names=['Transitions','Methods','Stats']) )
    with pd.option_context('display.max_rows', 10, 'display.max_columns', 10):print("isotopeDataFrame:\n",isotopeDataFrame)
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
 
  isoShiftsFrame = makeShiftFrame(isotopeDataFrame, refMass=245, errorsFrom='stderr', fileWrite=True)
  with pd.option_context('display.max_rows', 10, 'display.max_columns', 10):print("Isotope Shifts:\n",isoShiftsFrame)
  
  dcr2LYNCH = {242:1.1708, 243:1.2680, 244:1.4041, 245:1.4858, 247:1.5871} #dictionary with δ<r^2> measurements due to K.M.LYNCH (2018) for Radium isotopes (with masses shifted to acount for 19Fluorine)
  theoryShiftPerfm = {243:-0.78549, 245:0, 247:-0.78549}#{243:-1.56667, 245:0, 247:-1.7596} #dictionary with scaling constants, i.e. if isotopeShift_i = A_i + B_i*δ<r^2>, these are the B_i (and we expect A_i to be p negligible here) - from email thread due to Timur and Berger
  #isoShiftsFrame.loc[:,idx[:,'maybeTheory?']] = np.zeros_like(isoShiftsFrame.loc[:,idx[:,'SkewedMu']])
  #isoShiftsFrame.insert(columns='maybeTheory?',level='Methods')
  isoShiftsFrame2 = makeShiftFrame(isotopeDataFrame, refMass=245, errorsFrom='stderr', fileWrite=False, extraMeths='MaybeTheory?') #this is kind of a wack way to have to do this shit imo
  for m in list(theoryShiftPerfm.keys()): 
    print("test? m =%d, prediction=%.3f"%(m,theoryShiftPerfm[m]*(dcr2LYNCH[m]-dcr2LYNCH[245]) ))
    isoShiftsFrame2.loc[m,idx[:,'MaybeTheory?','shift']]= theoryShiftPerfm[m]*(dcr2LYNCH[m]-dcr2LYNCH[245])#theoryShiftPerfm[m]*(dcr2LYNCH[m]-dcr2LYNCH[m])
    isoShiftsFrame2.loc[m,idx[:,'MaybeTheory?','uncertainty']]=theoryShiftPerfm[m]*(math.sqrt(2)*0.0002)+0.1*(dcr2LYNCH[m]-dcr2LYNCH[245])

  '''shift vs mass plots'''
  #for i in range(len(transitionLabels)):   shiftVsMassPlotter_OneTransitionSlice(isoShiftsFrame2,i,saveFig=True, closeFig=not(i==0)) #shift vs mass plots to compare methods
  #for i in range(len(estimationMethodLabels)): shiftVsMassPlotter_OneMethodSlice(isoShiftsFrame,i,saveFig=True, closeFig=True) #shift vs mass plots to compare transitions
    
  '''importTestFrame=pd.DataFrame(data=np.loadtxt('./FitResults/OutputFiles/isoShiftsFrame.csv'), index=massList, columns = pd.MultiIndex.from_product([transitionLabels,estimationMethodLabels, ['shift','uncertainty']], names=['Shifts','Methods','Stats']) )
  print("Testing data frame importability. importTestFrame==isoShiftsFrame: ", np.all((importTestFrame-isoShiftsFrame).values==0))'''

  '''--- 3/20/20 IDK WHAT I'M DOING'''
  print("test:\n", isoShiftsFrame.loc[:,idx[:,'SkewedMu',:]])
  mList=np.array(isoShiftsFrame.index.values); tLabs=isoShiftsFrame.columns.unique(level='Transitions').values; methLabs=isoShiftsFrame.columns.unique(level='Methods').values
  meth=methLabs[0]
  plt.figure("Comparison")
  plt.gcf().set_size_inches(20, 12)
  plt.title("Isotope Shift vs. Mass for %s Estimation method in RaF\n(4 transitions included)\n Comparing w Silviu"%meth, fontsize=24)
  for j in range(len(tLabs)): #if it weren't for the damn offset, I could've written this loop as "for meth in methLabs:" sigh...
    transition = tLabs[j]
    color = next(plt.gca()._get_lines.prop_cycler)['color']
    offset = (j-math.floor(len(tLabs)/2))/(4*len(tLabs)) 
    plt.errorbar(x=(mList-245)+offset, y=isoShiftsFrame.loc[:,idx[transition,meth,'shift']], yerr=isoShiftsFrame.loc[:,idx[transition,meth,'uncertainty']],fmt="o",label=r"Alex$_{%d \rightarrow %d}$"%(j,j), color=color, markersize=8,alpha=0.25)
    silviuTempDat = np.genfromtxt("./SilviuData/isotope_shift_%d-%d.csv"%(j,j), comments='#', delimiter=",")
    print("testssts:\n", silviuTempDat[:,0])
    plt.errorbar(x=(mList-245)+offset, y= -1*silviuTempDat[:,0], yerr=silviuTempDat[:,1], fmt="x",label=r"Silviu$_{%d \rightarrow %d}$"%(j,j), color=color, markersize=8)
  plt.xlabel('Mass Difference (amu)', fontsize=18)
  plt.ylabel(r'shift $(cm^{-1})$', fontsize=18)
  plt.legend(loc=3, fontsize=24, ncol=len(tLabs))

  silviuDat = np.zeros((4,5,2))
  '''for i in range(4):
    print("SilviuData/isotope_shift_%d-%d"%(i,i))
    silviuTempDat = np.genfromtxt("./SilviuData/isotope_shift_%d-%d.csv"%(i,i), comments='#', delimiter=",")
    print("testssts:\n", silviuTempDat[:,0])'''
  plt.show()