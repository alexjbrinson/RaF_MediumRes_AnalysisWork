import numpy as np
import matplotlib.pyplot as plt
import math
import os.path
import LoadingAndMungingData as lmd
import FrequencyConvertedUtilityKit as fcuk
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import json
import lmfit
from lmfit import Model, Parameter
from lmfit.models import SkewedVoigtModel, LinearModel, GaussianModel, LorentzianModel
import emcee

'''2. "Do the same of the different isotopes 223-228Ra"'''
def makeScanToWavemeterDic(mass, verbose=False, redo=False):
  massDir= '../RaF_RawData/'+str(mass)+'/'
  dirlist=os.listdir(massDir)
  scanInds = []
  for scanFolder in dirlist:
    scanInds.append(int(scanFolder.lstrip('scan_')))
  if verbose: print("mass%d scanInds: "%mass, scanInds)
  colorDict={'pdl':"Red", 1:'Green', 2:'Blue', 3:'Purple', 4:'Orange'}
  wavemeterDic={}
  if os.path.exists('./wavemeterToUseDictionaries/RaF%dWavemeterDictionary.txt'%mass) and (redo==False):
    with open('./wavemeterToUseDictionaries/RaF%dWavemeterDictionary.txt'%mass,'r') as dicFile:
      wavemeterDic = json.load(dicFile)
  else:
    for s in np.sort(scanInds):
      wavemeterDic[str(s)]=str(lmd.whichWavemeter(mass,s))
    if not os.path.exists('./wavemeterToUseDictionaries/'):
      os.mkdir('./wavemeterToUseDictionaries/')
    with open('./wavemeterToUseDictionaries/RaF%dWavemeterDictionary.txt'%mass,'wb') as dicFile:
      dicFile.write(json.dumps(wavemeterDic,sort_keys=True).encode("utf-8"))
  if verbose: print("mass: %d  wavemeter Dictionary:\n"%mass, wavemeterDic)
  return(wavemeterDic)

"""fig1 = plt.figure("Wavenumber Ranges")
counter=0
unclearScansExist=False
for k in np.sort(list(datDic.keys())):
  datArray = datDic[k]
  if k in dyeInds:
    plt.plot(datArray[:,2], counter*np.ones_like(datArray[:,2])+0.0, "r-", alpha=.5, lw=16)
  elif k in tiSapInds:
    plt.plot(datArray[:,2], counter*np.ones_like(datArray[:,2]), "b-", alpha=.5, lw=16)
  else:
    plt.plot(datArray[:,2], counter*np.ones_like(datArray[:,2]), "g-", alpha=.5, lw=16)
    unclearScansExist=True
  plt.text(np.mean(datArray[:,2]), counter+.0, str(k), fontsize=16, horizontalalignment='center', verticalalignment='center')
  counter+=1
plt.xlabel(r'Wavenumber (cm$^{-1}$)', fontsize=18)
plt.ylabel('Index of scan', fontsize=18)
red_patch = mpatches.Patch(color='red', label="Dye Laser Scans")
blue_patch = mpatches.Patch(color='blue', label="TiSaph Laser Scans")
green_patch = mpatches.Patch(color='green', label="Unclear Scans")
if unclearScansExist: plt.legend(loc=4, handles=[red_patch, blue_patch, green_patch], fontsize=16)
else: plt.legend(loc=4, handles=[red_patch, blue_patch], fontsize=16)
plt.title(r'Wavenumber Ranges of $^{226}$Ra$^{19}$F LowRes Scans', fontsize=24)
plt.gcf().set_size_inches(20, 12)
plt.savefig("ScanWavenumberRanges.png")

for scindex in np.sort(list(datDic.keys())): dataFileComparator(scindex)"""

allScansBigDic = {}
for m in [241,242,243,244,245,247]:
  allScansBigDic[m] = makeScanToWavemeterDic(m, redo=False, verbose=True)

rewrite=True

for m in [241,242,243,244,247]:
  print("Doing new task. m = ",m)
  if not os.path.exists('./FrequencyConvertedDatasets/Plots/Mass%dPlots/'%m):
      os.mkdir('./FrequencyConvertedDatasets/Plots/Mass%dPlots/'%m)
  scanOverviewFigm = plt.figure('Mass: %d ; Scan Overview Figure'%m)
  plt.gcf().set_size_inches(20, 12)
  plt.title(r'Wavenumber Ranges of $^{%d}$Ra$^{19}$F Scans'%(m-19), fontsize=18)
  plt.xlabel(r'Wavenumber (cm$^{-1}$)', fontsize=18)
  plt.ylabel('Index of scan', fontsize=18)
  counter=0
  needPatches=[False,False,False,False]
  for s in np.sort(np.array(list(allScansBigDic[m].keys())).astype(int)):
    print("s=",s)
    wmToUse = allScansBigDic[m][str(s)]#I hate that everything has to be stored with strings in json...
    wmToUse = "pdl" if wmToUse == "pdl" else int(wmToUse)
    needWrite = (not os.path.exists('./FrequencyConvertedDatasets/%d/scan_%d'%(m,s))) or rewrite
    resolution = .01
    prdf = lmd.rawDatPrep(m, s, wmToUse, cleanWM=True, verbose=False)#preppedRawDataFrame. Everything but together, but not yet binned by wavenumber
    newDataFrame = lmd.makeUseable(prdf, resolution=resolution, noNaNsense=True, cropSparseEnds=True)
    #newDataFrame = lmd.doEverything(m, s, wmToUse, resolution=resolution, writeToFile=needWrite, makePlot=False, cleanWM=True, verbose=False)
    xdat = newDataFrame.loc[:,'wavenumber_mean']; ydat = newDataFrame.loc[:,'signal_value']; sigydat = newDataFrame.loc[:,'signal_uncertainty']
    if (len(xdat)<100 and np.mean(sigydat)<1):
      resolution=float(max(xdat)-min(xdat))/100.
      print("sigh. Redoing scan at %.3f resolution"%resolution)
      newDataFrame = lmd.makeUseable(prdf, resolution=resolution, noNaNsense=True, cropSparseEnds=True)
      xdat = newDataFrame.loc[:,'wavenumber_mean']; ydat = newDataFrame.loc[:,'signal_value']; sigydat = newDataFrame.loc[:,'signal_uncertainty']
    if needWrite: lmd.fileWriter(newDataFrame, m, s)
    plt.figure('output Plot, Mass: %d ; scan: %d wavemeter_'%(m,s) + str(wmToUse))
    plt.gcf().set_size_inches(20, 12)
    plt.errorbar(xdat, ydat, yerr=sigydat, fmt='bo-', ecolor='k', alpha=.5, markersize=5)
    plt.fill_between(xdat, ydat, color='blue', alpha=.3)
    plt.title(r'$^{%d}$Ra$^{19}$F Scan %d wavemeter_'%(m-19,s) + str(wmToUse)+ '\nCount Rate vs Wavenumber at '+r'$%.3f cm^{-1}$ Resolution'%resolution, fontsize=24)
    plt.xlabel(r'wavenumber ($cm^{-1}$)', fontsize=18)
    plt.ylabel('rate (counts/s)', fontsize=18)
    plt.legend(loc=2, fontsize=18)
    del(newDataFrame)
    plt.savefig('./FrequencyConvertedDatasets/Plots/Mass%dPlots/scan_%dPlot_%dBins.png'%(m,s,len(xdat)))
    plt.close()
    plt.figure(scanOverviewFigm.number)
    #if max(xdat)-min(xdat) < 20:
    #  plt.plot([np.mean(xdat)-10,np.mean(xdat)+10], [counter,counter], "-", color="grey", alpha=.25, lw=16)
    if wmToUse == "pdl":
      plt.plot(xdat, counter*np.ones_like(xdat)+0.0, "r-", alpha=.5, lw=16)
      needPatches[0]=True
    elif wmToUse == 1:
      plt.plot(xdat, counter*np.ones_like(xdat), "g-", alpha=.5, lw=16)
      needPatches[1]=True
    elif wmToUse == 2:
      plt.plot(xdat, counter*np.ones_like(xdat), "b-", alpha=.5, lw=16)
      needPatches[2]=True
    else:
      plt.plot(xdat, counter*np.ones_like(xdat), "y-", alpha=.5, lw=16)
      needPatches[3]=True
    plt.text(np.mean(xdat), counter+.0, str(s), fontsize=16, horizontalalignment='center', verticalalignment='center')
    counter+=1
  red_patch = mpatches.Patch(color='red', label="COBRA Scans")
  green_patch = mpatches.Patch(color='green', label="Injected Ti:Sa (HighRes)?")
  blue_patch = mpatches.Patch(color='blue', label="Grating Ti:Sa (LowRes)")
  yellow_patch = mpatches.Patch(color='yellow', label="laser source unclear")
  patchList=[red_patch,green_patch,blue_patch,yellow_patch]; handleList=[];
  for i in range(len(needPatches)):
    if needPatches[i]:handleList.append(patchList[i])
  lgd = plt.legend(loc="upper right", handles=handleList, fontsize=16, bbox_to_anchor=(1,1.0+.05*len(handleList)))
  plt.savefig('./OverviewFigures/%dScanOverviewFigure.png'%m, bbox_extra_artists=(lgd,), bbox_inches='tight')
  plt.close()

"""
plt.xlabel('Scan number', fontsize=18)
plt.ylabel('Molecule Mass (amu)', fontsize=18)
red_patch = mpatches.Patch(color='red', label="COBRA Dye Laser Scans")
green_patch = mpatches.Patch(color='green', label="Other Dy Laser Scans")
blue_patch = mpatches.Patch(color='blue', label="Ti:Sa Laser Scans")
purple_patch = mpatches.Patch(color='purple', label="wavemeter_3")
orange_patch = mpatches.Patch(color='orange', label="wavemeter_4")
#plt.legend(loc=4, handles=[red_patch, green_patch, blue_patch, purple_patch, orange_patch], fontsize=16)
plt.legend(loc=4, handles=[red_patch, green_patch, blue_patch], fontsize=16)
plt.title(r'Overview of All RaF Data', fontsize=24)
plt.gcf().set_size_inches(20, 12)
plt.savefig("RaFAllDataOverviewPlot.png")"""
'''3. "Analyse each scan individually and extract an average "peak position" for each electronic transition "'''
scans242=[2312, 2313]
scans243=[2283,2301,2302,2303,2308]#2300?
scans244=[2304,2305,2306,2307]
scans247=[2188,2190,2311]
'''4. "Make a table of "isotope shifts", comparing differences between different isotopes and using the same electronic transition"'''