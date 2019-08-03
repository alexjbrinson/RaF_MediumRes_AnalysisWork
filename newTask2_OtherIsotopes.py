import numpy as np
import matplotlib.pyplot as plt
import math
import os.path
import LoadingAndMungingData as lmd
import FrequencyConvertedUtilityKit as fcuk
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import json
"""import lmfit
from lmfit import Model, Parameter
from lmfit.models import SkewedVoigtModel, LinearModel, GaussianModel, LorentzianModel
import emcee"""

'''2. "Do the same of the different isotopes 223-228Ra"'''
def prepMassScans(mass, verbose=False, redo=False):
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

def dataFileComparator(scanInd):
  print("Now comparating scan_%d"%scanInd)
  oldLowResDatArray = datDic[scanInd]
  if scanInd in dyeInds: waveMeter = 'pdl'
  elif scanInd in tiSapInds: waveMeter = 2
  else: wavemeter = 'fsdaasdfads'
  binCount = len(oldLowResDatArray[:,2])
  needWrite = not os.path.exists('./FrequencyConvertedDatasets/245/scan_%d'%scanInd)
  newDataFrame = lmd.doEverything(245, scanInd, waveMeter, nBins=binCount, writeToFile=needWrite, makePlot=False, cleanWM=True, verbose=False)
  plt.figure("output Plot, mass: %d scan: "%245 +str(scanInd)+ " wavenumber: " +str(waveMeter)+ " numBins: %d"%binCount)
  plt.gcf().set_size_inches(20, 12)
  plt.errorbar(oldLowResDatArray[:,2], oldLowResDatArray[:,3], yerr = oldLowResDatArray[:,1], fmt='bo-', ecolor='k', alpha=.3, label='oldLowResDatArray')
  plt.fill_between(oldLowResDatArray[:,2], oldLowResDatArray[:,3],color='blue', alpha=.3)
  plt.errorbar(x=newDataFrame.loc[:,'wavenumber_mean'], y=newDataFrame.loc[:,'signal_value'], yerr=newDataFrame.loc[:,'signal_uncertainty'], fmt="ro-", ecolor='k', label='newDataFrame', markersize=3)
  plt.title('Mass: %d ; scan: '%245 +str(scanInd)+ ' wavemeter_' + str(waveMeter)+ '\ncount rate vs wavenumber for %d wavenumber bins'% len(newDataFrame.loc[:,'wavenumber_mean']), fontsize=24)
  plt.xlabel(r'wavenumber ($cm^{-1}$)', fontsize=18)
  plt.ylabel('rate (counts/s)', fontsize=18)
  plt.legend(loc=2, fontsize=18)
  if not os.path.exists('./FrequencyConvertedDatasets/245ComparatorPlots/'):
    os.mkdir('./FrequencyConvertedDatasets/245ComparatorPlots/')
  plt.savefig('./FrequencyConvertedDatasets/245ComparatorPlots/scan_%dComparisonPlot.png'%scanInd)
  plt.close()
  del(newDataFrame)

for scindex in np.sort(list(datDic.keys())): dataFileComparator(scindex)"""
allScansBigDic = {}
for m in [241,242,243,244,245,247]:
  allScansBigDic[m] = prepMassScans(m, redo=True, verbose=True)

print("test? allScansBigDic:\n",allScansBigDic)
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
'''4. "Make a table of "isotope shifts", comparing differences between different isotopes and using the same electronic transition"'''