import math
import numpy as np
import pandas as pd
import os.path
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import LoadingAndMungingData as lmd
import FrequencyConvertedUtilityKit as FCUK

'''3. "Analyse each scan individually and extract an average "peak position" for each electronic transition "'''


if __name__ == '__main__':

  allScansBigDic = {}
  for m in [241,242,243,244,245,247]:
    allScansBigDic[m] = lmd.makeScanToWavemeterDic(m, redo=False, verbose=True)

  rewrite=False
  
  
  massList=[242,243,244,245, 247]
  colorDict={242:'red', 243:'orange',244:'green',245:'blue',247:'purple'}
  massScanDic={}
  massScanDic[242]=[2312, 2313]
  massScanDic[243]=[2283,2301,2302,2303,2308]#2300?
  massScanDic[244]=[2304,2305,2306,2307]
  massScanDic[245]=[2309,2310,2320]#,2341,2349,2350]#,2346]#2346 is a pdl scan though. gross...
  massScanDic[247]=[2188,2190,2311]
  initCenterEsts=[13285,13278.8,13272.8,13266.57]#,13260]
  sigmaEst=.8
  gammaEst=1.5
  resolutionList=[.01,.02,.05,.1,.2,.5]
  resolution = .1
  normalization="MaxValue" #Integral

  for m in massList:
    isoFrames = []

    for s in massScanDic[m]:
      print("m=%d, s=%d")
      """
      if (s in [2312,2313,2283]): peakList = initCenterEsts
      elif (s in [2301,2302,2303]): peakList = [13284.7,13278.2,13272.8]
      elif s==2308: peakList=[13272.5,13266.6]
      elif (s in [2304,2305,2306]): peakList = initCenterEsts
      elif s == 2307: peakList = [13272.5,13266.67,13261]
      elif (s in [2188,2190]): [13284.7,13278.2,13272.8]
      elif s==2311: peakList = initCenterEsts
      else: peakList=initCenterEsts
      FCUK.Scanalyzer(m,s,rewrite=False,peakList=peakList,peakSigmas=sigmaEst*np.ones_like(peakList),initGamma=gammaEst,resList=resolutionList,method="leastsq", fitPlots=True, binSpreadPlot=True, sameSkew=True, useWeights=True, skew0=-2)
      """
      isoFrames.append(lmd.rawDatPrep(m,s))
      
    isotopeData=pd.concat(isoFrames)
    #isoDope=lmd.makeUseable(isotopeData, resolution=.1, normalize=True)
    isoDope=lmd.makeUseable(isotopeData, resolution=resolution, normalizedOn=normalization,ltrim=13256.5,rtrim=13285.5)
    plt.errorbar(x=isoDope.loc[:,'wavenumber_mean'], y=isoDope.loc[:,'signal_value']-np.min(isoDope.loc[:,'signal_value']), yerr=isoDope.loc[:,'signal_uncertainty'], fmt=".-",color=colorDict[m],ecolor='k', alpha=.5, label=r'$^{%d}$Ra$^{19}$F'%(m-19))
  plt.gcf().set_size_inches(20, 12)
  plt.xlabel(r'wavenumber ($cm^{-1}$)')
  plt.ylabel('rate (counts/s)')
  plt.title("Q-Band Spectra For Different RaF Isotopes, Resolution=%.2f, Normalized on "%resolution +str(normalization))
  plt.legend(loc='best')
  plt.show()
'''4. "Make a table of "isotope shifts", comparing differences between different isotopes and using the same electronic transition"'''