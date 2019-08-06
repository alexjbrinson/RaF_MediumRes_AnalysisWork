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
  massScanDic[245]=[2309,2310,2320]#, [2341(75mW),2349,2350("re-tuned TiSa overlap upstairs --> additional 50% improvement")]#,[2346] is a pdl scan though. gross...#2178 is Ti:Sa, but actually gross af
  massScanDic[247]=[2311,2322]#,2188,2190]
  initCenterEsts=[13284.75,13278.62,13272.50,13266.5]#,13260.35]
  sigmaEst=.55
  gammaEst=1.77
  skew0
  resolutionList=[.01,.02,.05,.1,.2,.5]

  for m in massList:
    for s in massScanDic[m]:
      print("m=%d, s=%d")
      if (s in [2312,2313,2283]): peakList = initCenterEsts
      elif (s in [2301,2302,2303]): peakList = [13284.7,13278.2,13272.8]
      elif s==2308: peakList=[13272.5,13266.6]
      elif (s in [2304,2305,2306]): peakList = initCenterEsts
      elif s == 2307: peakList = [13272.5,13266.67,13261]
      elif (s in [2188,2190]): [13284.7,13278.2,13272.8]
      elif s==2311: peakList = initCenterEsts
      else: peakList=initCenterEsts
      FCUK.Scanalyzer(m,s,rewrite=True,peakList=peakList,peakSigmas=sigmaEst*np.ones_like(peakList),initGamma=gammaEst,resList=resolutionList,method="leastsq", fitPlots=True, binSpreadPlot=True, sameSkew=True, useWeights=True, skew0=-4)
    (cfrx,ccex) = FCUK.fitScanX(lmd.mergeDatRaw(m,massScanDic[m]),m,0, initCenterEsts, peakSigmas=sigmaEst*np.ones_like(initCenterEsts), initGamma=gammaEst, resList=resolutionList, makePlots=True, useWeights=True, sameSkew=True, skew0=-4)
    FCUK.MultiBinSpreadPlotter(m,0,ccex,resolutionList)
    finalScanEstimates = np.c_[np.mean(ccex[:,:,0], axis=0), np.std(ccex[:,:,0], axis=0), np.max(ccex[:,:,0], axis=0)-np.min(ccex[:,:,0], axis=0)]
    #print("finalScanEstimates.shape=",finalScanEstimates.shape,"\nfinalScanEstimates:\n",finalScanEstimates)
    #finalScanEsts = np.copy(finalScanEstimates)
    for p in range(len(ccex[0,:,0])):
      weightStats = FCUK.weightedStatistics(ccex[:,p,0], ccex[:,p,1])
      #print("test: weightStats = ", weightStats)
      finalScanEstimates[p,0] = weightStats[0]; finalScanEstimates[p,1] = weightStats[1]
    print("mass%d, combined scans "%m, massScanDic[m]," finalScanEstimates:\n", finalScanEstimates[:,0])
'''4. "Make a table of "isotope shifts", comparing differences between different isotopes and using the same electronic transition"'''