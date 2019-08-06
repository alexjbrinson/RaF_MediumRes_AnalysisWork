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


if __name__ == '__main__':

  allScansBigDic = {}
  for m in [241,242,243,244,245,247]:
    allScansBigDic[m] = lmd.makeScanToWavemeterDic(m, redo=False, verbose=True)
  
  massList=[242,243,244,245, 247]
  colorDict={242:'red', 243:'orange',244:'green',245:'blue',247:'purple'}
  massScanDic={}
  massScanDic[242]=[2312, 2313]
  massScanDic[243]=[2283,2301,2302,2303,2308]#2300?
  massScanDic[244]=[2304,2305,2306,2307]
  massScanDic[245]=[2178,2309,2310,2320,2341,2349,2350,2346]#2346 is a pdl scan though. gross...
  massScanDic[247]=[2188,2190,2311]

  for m in massList:
    isoFrames = []
    if not os.path.exists('./ToShare/RaF_PI12-Qband_Isotopes/Data/Mass%d/'%m):
      os.makedirs('./ToShare/RaF_PI12-Qband_Isotopes/Data/Mass%d/'%m)
    if not os.path.exists('./ToShare/RaF_PI12-Qband_Isotopes/SanityCheckPlots/Mass%d/'%m):
      os.makedirs('./ToShare/RaF_PI12-Qband_Isotopes/SanityCheckPlots/Mass%d/'%m)
    for s in massScanDic[m]:
      print("m=%d, s=%d")
      isoFrames.append(lmd.rawDatPrep(m,s))
      newDataFrame = lmd.makeUseable(isoFrames[-1], resolution=.01, noNaNsense=True, cropSparseEnds=True)
      if s == 2346:
        os.mkdir('./ToShare/RaF_PI12-Qband_Isotopes/Data/Mass%d/Dye/'%m)
        os.mkdir('./ToShare/RaF_PI12-Qband_Isotopes/SanityCheckPlots/Mass%d/Dye/'%m)
        lmd.fileWriter(newDataFrame, m, s, target='./ToShare/RaF_PI12-Qband_Isotopes/Data/Mass%d/Dye/%dRaF_LR_scan%d.csv'%(m,m,s))
      else: 
        lmd.fileWriter(newDataFrame, m, s, target='./ToShare/RaF_PI12-Qband_Isotopes/Data/Mass%d/%dRaF_LR_scan%d.csv'%(m,m,s))
      xdat = newDataFrame.loc[:,'wavenumber_mean']; ydat = newDataFrame.loc[:,'signal_value']; sigydat = newDataFrame.loc[:,'signal_uncertainty']
      plt.figure('output Plot, Mass: %d ; scan: %d'%(m,s))
      plt.gcf().set_size_inches(20, 12)
      plt.errorbar(xdat, ydat, yerr=sigydat, fmt='bo-', ecolor='k', alpha=.5, markersize=5)
      plt.fill_between(xdat, ydat, color='blue', alpha=.3)
      plt.xlabel(r'wavenumber ($cm^{-1}$)', fontsize=18)
      plt.ylabel('rate (counts/s)', fontsize=18)
      del(newDataFrame)
      if s==2346:
        plt.title(r'$^{%d}$Ra$^{19}$F Scan %d - COBRA PDL \nCount Rate vs Wavenumber at '%(m-19,s)+r'$.01 cm^{-1}$ Resolution', fontsize=24)
        plt.savefig('./ToShare/RaF_PI12-Qband_Isotopes/SanityCheckPlots/Mass%d/Dye/%dRaF_LR_scan%d.png'%(m,m,s))
      else:
        plt.title(r'$^{%d}$Ra$^{19}$F Scan %d - Grating Ti:Sa \nCount Rate vs Wavenumber at '%(m-19,s)+r'$.01 cm^{-1}$ Resolution', fontsize=24)
        plt.savefig('./ToShare/RaF_PI12-Qband_Isotopes/SanityCheckPlots/Mass%d/%dRaF_LR_scan%d.png'%(m,m,s))
      plt.close()