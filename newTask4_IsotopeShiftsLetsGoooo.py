import math
import numpy as np
import scipy as sp
import scipy.odr as spodr
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
import lmfit
from lmfit import Model, Parameter
from lmfit.models import LinearModel

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
  '''for m in massList:
    isoShiftsFrame.loc[m,'deltaRSq']=deltaRsq(m, dRsqDic, refMass=245)'''
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
  for j in range(len(tLabs)):
    transition = tLabs[j]
    color = next(plt.gca()._get_lines.prop_cycler)['color']
    offset = (j-math.floor(len(tLabs)/2))/(4*len(tLabs)) 
    plt.errorbar(x=(mList-245)+offset, y=shiftsFrame.loc[:,idx[transition,meth,'shift']], yerr=shiftsFrame.loc[:,idx[transition,meth,'uncertainty']],fmt="o",label=r"$%d \rightarrow %d$"%(j,j), color=color, markersize=8)
  plt.xlabel('Mass Difference (amu)', fontsize=18)
  plt.ylabel(r'shift $(cm^{-1})$', fontsize=18)
  plt.legend(loc=3, fontsize=24, ncol=len(tLabs))
  if saveFig==True: plt.savefig('./FitResults/IsotopeShiftDiffTransitions_%s.png'%meth)
  if closeFig: plt.close()

def shiftVsChargeRadPlotter_OneMethodSliceOld(shiftsFrame, i, deltaRSqDic,sigmaRSqDic,refMass=0, saveFig=True, closeFig=True,forceOrigin=True):
  """shift vs mass plots to compare methods"""
  mList=np.array(shiftsFrame.index.values);
  deltList=np.array([deltaRsq(m, deltaRSqDic, refMass=refMass) for m in mList])
  nonRefMasses=[]
  for m in mList:
    if m!=refMass: nonRefMasses.append(m)
  xDat=np.array([deltaRsq(m, deltaRSqDic, refMass=refMass) for m in nonRefMasses])
  xErr=np.sqrt(sigmaRSqDic[refMass]**2+np.array([sigmaRSqDic[m] for m in nonRefMasses]))
  print("tests.xDat:\n",xDat,"xErr",xErr)
  tLabs=shiftsFrame.columns.unique(level='Transitions').values; methLabs=shiftsFrame.columns.unique(level='Methods').values
  meth=methLabs[i]
  plt.figure(meth)
  plt.gcf().set_size_inches(20, 12)
  plt.title(r'Isotope Shift vs. $\delta\langle r^2 \rangle$ for %s Estimation method in RaF'%meth+'\n(4 transitions included, forceOrigin=%s)'%str(forceOrigin), fontsize=24)
  allFits=np.empty((len(tLabs),2))
  for j in range(len(tLabs)):
    transition = tLabs[j]
    color = next(plt.gca()._get_lines.prop_cycler)['color']
    yDat=np.array([shiftsFrame.loc[m,idx[transition,meth,'shift']] for m in nonRefMasses])
    yErr=np.array([shiftsFrame.loc[m,idx[transition,meth,'uncertainty']] for m in nonRefMasses])
    lmod = LinearModel(prefix='l0_')
    params = lmod.make_params()
    if forceOrigin: params['l0_intercept']= Parameter('l0_intercept', value=0, min=-0.01, max = 0.01, vary=False)
    fitResult=lmod.fit(yDat, params, x=xDat, weights=1/np.square(yErr) )
    print("test: j=%d; best_values="%j,fitResult.best_values)
    allFits[j,:]=np.array([fitResult.best_values['l0_intercept'],fitResult.best_values['l0_slope']])
    if forceOrigin: fitReportFile = open('./FitResults/isoShiftsTransition%d_FitReport_OriginFixed.txt'%j,'w+')
    else: fitReportFile = open('./FitResults/isoShiftsTransition%d_FitReport'%j,'w+')
    fitReportFile.write("Isotope Shift Fit Report for: %d -> %d Transition:"%(j,j));
    fitReportFile.write(fitResult.fit_report(min_correl=0.25)); fitReportFile.close()
    offset = 0#(j-math.floor(len(tLabs)/2))/(4*len(tLabs))
    if forceOrigin:
      plt.errorbar(x=deltList+offset, y=shiftsFrame.loc[:,idx[transition,meth,'shift']], yerr=shiftsFrame.loc[:,idx[transition,meth,'uncertainty']],fmt="o",
      label=r"$%d \rightarrow %d$"%(j,j)+'\n'+r'$\delta\nu=%.2f\, \delta\langle r^2\rangle$'%allFits[j,1], color=color, markersize=8)
    else:  
      plt.errorbar(x=deltList+offset, y=shiftsFrame.loc[:,idx[transition,meth,'shift']], yerr=shiftsFrame.loc[:,idx[transition,meth,'uncertainty']],fmt="o",
      label=r"$%d \rightarrow %d$"%(j,j)+'\n'+r'$\delta\nu=%.2f,%.2f\, \delta\langle r^2\rangle$'%(allFits[j,0],allFits[j,1]), color=color, markersize=8)
    plt.plot(deltList, fitResult.eval(x=deltList,params=fitResult.params), linestyle='dashed', color=color)
  plt.xlabel(r'$\delta\langle r^2 \rangle$ (fm$^2$)', fontsize=18)
  plt.ylabel(r'shift $(cm^{-1})$', fontsize=18)
  plt.legend(loc=3, fontsize=14, ncol=len(tLabs))
  if saveFig==True: plt.savefig('./FitResults/IsotopeShiftVsChargeRadTransitions_%s_forceOrig%s.png'%(meth,str(forceOrigin)))
  if closeFig: plt.close()

def shiftVsChargeRadPlotter_OneMethodSlice(shiftsFrame, i, deltaRSqDic,sigmaRSqDic,refMass=0, saveFig=True, closeFig=True,forceOrigin=True,verbose=False):
  """shift vs mass plots to compare methods"""
  mList=np.array(shiftsFrame.index.values);
  deltList=np.array([deltaRsq(m, deltaRSqDic, refMass=refMass) for m in mList])
  nonRefMasses=[]
  for m in mList:
    if m!=refMass: nonRefMasses.append(m)
  xDat=np.array([deltaRsq(m, deltaRSqDic, refMass=refMass) for m in nonRefMasses])
  xErr=np.sqrt(sigmaRSqDic[refMass]**2+np.square(np.array([sigmaRSqDic[m] for m in nonRefMasses])))
  tLabs=shiftsFrame.columns.unique(level='Transitions').values; methLabs=shiftsFrame.columns.unique(level='Methods').values
  meth=methLabs[i]
  plt.figure(meth)
  plt.gcf().set_size_inches(20, 12)
  plt.title(r'Isotope Shift vs. $\delta\langle r^2 \rangle$ for %s Estimation method in RaF'%meth+'\n(4 transitions included, forceOrigin=%s)'%str(forceOrigin), fontsize=24)
  allFits=np.empty((len(tLabs),2,2))
  for j in range(len(tLabs)):
    transition = tLabs[j]
    color = next(plt.gca()._get_lines.prop_cycler)['color']
    yDat=np.array([shiftsFrame.loc[m,idx[transition,meth,'shift']] for m in nonRefMasses])
    yErr=np.array([shiftsFrame.loc[m,idx[transition,meth,'uncertainty']] for m in nonRefMasses])
    
    linear = spodr.Model(f)
    mydata = spodr.RealData(xDat,y=yDat,sx=xErr,sy=yErr)
    if forceOrigin: myodr = spodr.ODR(mydata, linear, beta0=[-1])
    else: myodr = spodr.ODR(mydata, linear, beta0=[-1,0])
    output=myodr.run()
    if verbose and j==0: 
      output.pprint()
    if forceOrigin:
      #allFits[j,0,:]=np.array([output.beta[0],output.sd_beta[0]])
      allFits[j,0,:]=np.array([output.beta[0],math.sqrt(output.cov_beta[0])])
      allFits[j,1,:]=np.zeros(2)
      fitReportFile = open('./FitResults/isoShiftsTransition%d_FitReport_OriginFixed.txt'%j,'w+')
      plt.errorbar(x=0, y=0, fmt="o", color=color, markersize=6)
      plt.errorbar(x=xDat, y=yDat, xerr=xErr, yerr=yErr, fmt="o",
      label=r"$%d \rightarrow %d$"%(j,j)+'\n'+r'$\delta\nu=(%.2f\,\pm %.3f)\, \delta\langle r^2\rangle$'%(allFits[j,0,0],allFits[j,0,1]), color=color, markersize=8)
    else:
      #allFits[j,0,:]=np.array([output.beta[0],output.sd_beta[0]])
      #allFits[j,1,:]=np.array([output.beta[1],output.sd_beta[1]])
      allFits[j,0,:]=np.array([output.beta[0],math.sqrt(output.cov_beta[0,0])])
      allFits[j,1,:]=np.array([output.beta[1],math.sqrt(output.cov_beta[1,1])])
      fitReportFile = open('./FitResults/isoShiftsTransition%d_FitReport'%j,'w+')
      plt.errorbar(x=0, y=0, fmt="o", color=color, markersize=6)
      plt.errorbar(x=xDat, y=yDat, xerr=xErr, yerr=yErr, fmt="o",
      label=r"$%d \rightarrow %d$"%(j,j)+'\n'+r'$\delta\nu=%.2f,%.2f\, \delta\langle r^2\rangle$'%(allFits[j,1,0],allFits[j,0,0]), color=color, markersize=8)
    
    fitReportFile.write("Isotope Shift Fit Report for: %d -> %d Transition:"%(j,j));
    fitReportFile.write("Beta: " + str(output.beta) + "\nBeta Std Error: " + str(output.sd_beta) + "\nBeta Covariance" + str(output.cov_beta))
    #fitReportFile.write("Residual Variance: ", output.res_var, "\nInfo: ", output.info, "\nStop Reason: ", output.stop)
    fitReportFile.close()  
    plt.plot(deltList, f(output.beta, deltList), linestyle='dashed', color=color)
    if forceOrigin: 
      plt.fill_between(deltList, f(output.beta+output.sd_beta, deltList), y2=f(output.beta-output.sd_beta, deltList), color=color, alpha=0.25 )

  plt.xlabel(r'$\delta\langle r^2 \rangle$ (fm$^2$)', fontsize=18)
  plt.ylabel(r'shift $(cm^{-1})$', fontsize=18)
  plt.legend(loc=3, fontsize=14, ncol=len(tLabs))
  if saveFig==True: plt.savefig('./FitResults/IsotopeShiftVsChargeRadTransitions_%s_forceOrig%s.png'%(meth,str(forceOrigin)))
  if closeFig: plt.close()

def deltaRsq(massNumber, drsqDic, refMass=0):
  deltaRef=drsqDic[refMass] if refMass>0 else 0#-19, since masses quoted included ^{19}F
  return(drsqDic[massNumber]-deltaRef)

def f(B, x):
  '''Linear function y = m*x + b'''
    # B is a vector of the parameters.
    # x is an array of the current x values.
    # x is in the same format as the x passed to Data or RealData.
    #
    # Return an array in the same format as y passed to Data or RealData.
  if len(B)==2:
    return B[0]*x + B[1]
  return B[0]*x



'''4. "Make a table of "isotope shifts", comparing differences between different isotopes and using the same electronic transition"'''
if __name__ == '__main__':
  pd.options.mode.chained_assignment = None  # default='warn'
  rewrite=False; importDataFrames=False
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
  resolutionList=[.01,.02,.03,.05,.07,.1,.2,.3]
  isotopeDataFrame = pd.DataFrame(index=massList, columns = pd.MultiIndex.from_product([transitionLabels,estimationMethodLabels, estimationStatisticsLabels], names=['Transitions','Methods','Stats']) )
  print("testss:\n",isotopeDataFrame.index,isotopeDataFrame.columns.unique('Transitions').values)
  #isotopeDataFrame=isotopeDataFrame.sort_index()
  dRsqDic = {242:1.1708,243:1.2680,244:1.4041,245:1.4858,247:1.6980} #given in terms of RaF isotope masses, which is hopefully less confusing?
  dRsqSigmaDic = {242:0.0002,243:0.0002,244:0.0002,245:0.0002,247:0.0002} #once again given in terms of Ra/f isotope masses
  if os.path.exists('./FitResults/OutputFiles/isotopeDataFrame.csv') and (importDataFrames==True):# and os.path.exists('./FitResults/OutputFiles/isoShiftsFrame.csv')
    print("ayyy! No computation necessary lol")
    isotopeDataFrame = pd.DataFrame(data=np.loadtxt('./FitResults/OutputFiles/isotopeDataFrame.csv'), index=massList, columns = pd.MultiIndex.from_product([transitionLabels,estimationMethodLabels, estimationStatisticsLabels], names=['Transitions','Methods','Stats']) )
    with pd.option_context('display.max_rows', 10, 'display.max_columns', 10):print("isotopeDataFrame:\n",isotopeDataFrame)
  else:
    isotopeDataFrame = pd.DataFrame(index=massList, columns = pd.MultiIndex.from_product([transitionLabels,estimationMethodLabels, estimationStatisticsLabels], names=['Transitions','Methods','Stats']) )
    for m in massList:
      s=massScanDic[m]
      peakList=initCenterEsts
      finalfitResults = FCUK.Scanalyzer(m,s,rewrite=rewrite,peaksList=peakList,peakSigmas=sigmaEst*np.ones_like(peakList),sameSigma=sameSigma,initGamma=gammaEst,resList=resolutionList, ltrim=ltrim, rtrim=rtrim, makePlots=True,sameSkew=True, useWeights=True, skew0=-4)
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

  #dcr2LYNCH = {242:1.1708, 243:1.2680, 244:1.4041, 245:1.4858, 247:1.5871} #dictionary with δ<r^2> measurements due to K.M.LYNCH (2018) for Radium isotopes (with masses shifted to acount for 19Fluorine)
  #theoryShiftPerfm = {243:-0.78549, 245:0, 247:-0.78549}#{243:-1.56667, 245:0, 247:-1.7596} #dictionary with scaling constants, i.e. if isotopeShift_i = A_i + B_i*δ<r^2>, these are the B_i (and we expect A_i to be p negligible here) - from email thread due to Timur and Berger
  #isoShiftsFrame.loc[:,idx[:,'maybeTheory?']] = np.zeros_like(isoShiftsFrame.loc[:,idx[:,'SkewedMu']])
  #isoShiftsFrame.insert(columns='maybeTheory?',level='Methods')
  '''isoShiftsFrame2 = makeShiftFrame(isotopeDataFrame, refMass=245, errorsFrom='stderr', fileWrite=False, extraMeths='MaybeTheory?') #this is kind of a wack way to have to do this shit imo
  for m in list(theoryShiftPerfm.keys()): 
    print("test? m =%d, prediction=%.3f"%(m,theoryShiftPerfm[m]*(dcr2LYNCH[m]-dcr2LYNCH[245]) ))
    isoShiftsFrame2.loc[m,idx[:,'MaybeTheory?','shift']]= theoryShiftPerfm[m]*(dcr2LYNCH[m]-dcr2LYNCH[245])#theoryShiftPerfm[m]*(dcr2LYNCH[m]-dcr2LYNCH[m])
    isoShiftsFrame2.loc[m,idx[:,'MaybeTheory?','uncertainty']]=theoryShiftPerfm[m]*(math.sqrt(2)*0.0002)+0.1*(dcr2LYNCH[m]-dcr2LYNCH[245])'''

  '''shift vs mass plots'''
  for i in range(len(transitionLabels)):   shiftVsMassPlotter_OneTransitionSlice(isoShiftsFrame,i,saveFig=True, closeFig=True) #shift vs mass plots to compare methods
  for i in range(len(estimationMethodLabels)): shiftVsChargeRadPlotter_OneMethodSlice(isoShiftsFrame,i,dRsqDic,dRsqSigmaDic,refMass=245,saveFig=True,forceOrigin=False, closeFig=True) #shift vs δ charge radius plots to compare transitions
  for i in range(len(estimationMethodLabels)): shiftVsChargeRadPlotter_OneMethodSlice(isoShiftsFrame,i,dRsqDic,dRsqSigmaDic,refMass=245,saveFig=True,forceOrigin=True, closeFig=not(i==0),verbose=(i==0)) #this time forcing fits to pass through origin
  '''importTestFrame=pd.DataFrame(data=np.loadtxt('./FitResults/OutputFiles/isoShiftsFrame.csv'), index=massList, columns = pd.MultiIndex.from_product([transitionLabels,estimationMethodLabels, ['shift','uncertainty']], names=['Shifts','Methods','Stats']) )
  print("Testing data frame importability. importTestFrame==isoShiftsFrame: ", np.all((importTestFrame-isoShiftsFrame).values==0))'''
  plt.show()