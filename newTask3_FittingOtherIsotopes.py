import numpy as np
import matplotlib.pyplot as plt
import math
import os.path

import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import json
import lmfit
from lmfit import Model, Parameter
from lmfit.models import SkewedVoigtModel, LinearModel, GaussianModel, LorentzianModel
import emcee
import LoadingAndMungingData as lmd
import FrequencyConvertedUtilityKit as FCUK

'''3. "Analyse each scan individually and extract an average "peak position" for each electronic transition "'''


if __name__ == '__main__':

  allScansBigDic = {}
  for m in [241,242,243,244,245,247]:
    allScansBigDic[m] = lmd.makeScanToWavemeterDic(m, redo=False, verbose=True)

  rewrite=False
  
  massList=[242,243,244,245, 247]
  massScanDic={}
  massScanDic[242]=[2312, 2313]
  massScanDic[243]=[2283,2301,2302,2303,2308]#2300?
  massScanDic[244]=[2304,2305,2306,2307]
  massScanDic[246]=[2309,2310,2320]
  massScanDic[247]=[2188,2190,2311]
  initCenterEsts=[13285,13278.8,13272.8,13266.57]#,13260]
  initWidthEsts=2*np.ones_like(initCenterEsts)
  resolutionList=[.01,.02,.05,.1,.2,.5]
  
  for m in massList:
    for s in massScanDic[m]:
      FCUK.Scanalyzer(m,s,resolutionList,peakList=initCenterEsts,peakRanges=initWidthEsts,resList=resolutionList,method="leastsq", fitPlots=True, binSpreadPlot=True, sameSkew=True, useWeights=True, skew0=-2)

'''4. "Make a table of "isotope shifts", comparing differences between different isotopes and using the same electronic transition"'''