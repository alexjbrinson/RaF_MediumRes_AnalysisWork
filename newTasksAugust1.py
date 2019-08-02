import numpy as np
import matplotlib.pyplot as plt
import math
import os.path
import LoadingAndMungingData as lmd
import FrequencyConvertedUtilityKit as fcuk
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
"""import lmfit
from lmfit import Model, Parameter
from lmfit.models import SkewedVoigtModel, LinearModel, GaussianModel, LorentzianModel
import emcee"""

'''1. "Check that the data files that I sent you before (frequency converted) agree with your frequency conversion"'''

dirlist=os.listdir('scans245')
scanInds245OLD = []
print("test1. os.listdir('scans245'):\n",dirlist)
for i in range(len(dirlist)):
  if (dirlist[i].endswith('.csv') and dirlist[i].startswith('245RaF_LR_')):
    scanInds245OLD.append( int(dirlist[i].replace('.csv',"").replace('245RaF_LR_',"")) )
print("test2. scanInds245OLD:\n",scanInds245OLD)

datDic = {}
for i in range(len(scanInds245OLD)):
  datArray = np.loadtxt('scans245/245RaF_LR_'+str(scanInds245OLD[i])+'.csv', dtype=float, skiprows=1, delimiter=',')
  if datArray.ndim != 2:
    print("Junk dataset from scan "+str(scanInds245OLD[i])+". Will throw out.")
  elif len(datArray[:,2])<20:
    print("Few datapoints in scan "+str(scanInds245OLD[i])+". Will throw out.")
  else:
    if np.any(datArray[:,2]<0):
      mask = datArray[:,2]>0
      datArray = np.array(datArray[[mask==True]])
      print("Scan "+str(scanInds245OLD[i])+" contained negative wavenumbers...", str(len(mask)-len(datArray[:,2])) + " data point(s) have been removed. Updated array shape =", datArray.shape)
    datDic[scanInds245OLD[i]] = fcuk.cleanDataSet(datArray)
    """if scanInds245OLD[i]==2137:
      print("All of the signal in Scan 2137 occurs in the first 6th of the dataset. Will crop the rest so it doesn't dominate the fits.")
      datArray=datDic[2137]
      rCutoff = np.argmin(np.abs(datArray[:,2]-13325))
      datDic[2137] = datArray[:rCutoff]
    elif scanInds245OLD[i]==2138:
      print("All of the signal in Scan 2138 occurs in the last 6th of the dataset. Will crop the rest so it doesn't dominate the fits.")
      datArray=datDic[2138]
      lCutoff = np.argmin(np.abs(datArray[:,2]-13225))
      datDic[2138] = datArray[lCutoff:]
    elif scanInds245OLD[i]==2178:
      print("There's some fishy business going on int the first quarter of Scan 2178. Will crop so it doesn't screw up my fits.")
      datArray=datDic[2178]
      lCutoff = np.argmin(np.abs(datArray[:,2]-13256))
      datDic[2178] = datArray[lCutoff:]
    elif scanInds245OLD[i]==2368:
      print("Scan 2368 has garbage at the very end. Will crop so it doesn't screw up my fits.")
      datArray=datDic[2368]
      datDic[2368] = datArray[:-2]""" #This commented out part was actually useful for trimming trashy data sets. But for now I'm just trying to use the lowRes files as a reference to compare my panda outputs with

print("datDic.keys()", list(datDic.keys()))

dyeInds = [2130,2131,2132,2323,2324,2325,2346,2360,2364,2365,2368,2375,2376]
tiSapInds = [2135,2136,2137,2138,2139,2164,2165,2178,2309,2310,2317,2319,2320,2340,2341,2349,2350]

fig1 = plt.figure("Wavenumber Ranges")
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
  newDataFrame = lmd.doEverything(245, scanInd, waveMeter, nBins=binCount, writeToFile=needWrite, makePlot=False, cleanWM=True, verbose=True)
  plt.figure("output Plot, mass: %d scan: "%245 +str(scanInd)+ " wavenumber: " +str(waveMeter)+ " numBins: %d"%binCount)
  plt.gcf().set_size_inches(20, 12)
  plt.errorbar(oldLowResDatArray[:,2], oldLowResDatArray[:,3], yerr = oldLowResDatArray[:,1], fmt='bo-', ecolor='k', alpha=.3, label='oldLowResDatArray')
  plt.fill_between(oldLowResDatArray[:,2], oldLowResDatArray[:,3],color='blue', alpha=.3)
  plt.errorbar(x=newDataFrame.loc[:,'wavenumber_mean'], y=newDataFrame.loc[:,'signal_value'], yerr=newDataFrame.loc[:,'signal_uncertainty'], fmt="ro-", ecolor='k', label='newDataFrame', markersize=3)
  plt.title('Mass: %d ; scan: '%245 +str(scanInd)+ ' wavemeter_' + str(waveMeter)+ '\ncount rate vs wavenumber for %d wavenumber bins'% len(newDataFrame.loc[:,'wavenumber_mean']), fontsize=18)
  plt.xlabel(r'wavenumber ($cm^{-1}$)', fontsize=16)
  plt.ylabel('rate (counts/s)', fontsize=16)
  plt.legend(loc=2, fontsize=16)
  if not os.path.exists('./FrequencyConvertedDatasets/245ComparatorPlots/'):
    os.mkdir('./FrequencyConvertedDatasets/245ComparatorPlots/')
  plt.savefig('./FrequencyConvertedDatasets/245ComparatorPlots/scan_%dComparisonPlot.png'%scanInd)
  plt.close()
  del(newDataFrame)

for scindex in np.sort(list(datDic.keys())): dataFileComparator(scindex)

'''2. "Do the same of the different isotopes 223-228Ra"'''
'''3. "Analyse each scan individually and extract an average "peak position" for each electronic transition "'''
'''4. "Make a table of "isotope shifts", comparing differences between different isotopes and using the same electronic transition"'''