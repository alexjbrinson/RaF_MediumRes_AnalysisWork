import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path
import LoadingAndMungingData as lmd

if __name__ == '__main__':
  mass=245
  scanList = [2317,2319,2340]; scanCombos=[[2317,2319],[2317,2340],[2319,2340],scanList]
  extraScans=[2185,2281,2316,2339,2347,2348]
  res=0.25
  colorList = ["yellow","red","blue"]; comboColors=["orange","green","purple","brown"]; extraColors=["orange","green","purple","brown","pink","gray"]
  singleScanFrames=[]
  for scan in scanList:
    exampleScan = 2340; # This is a really nice Q-band Sigma->Pi_1/2 scan. To see one of the new things that looks interesting, scan 2354 at resolution=.25
    singleScanFrames.append(lmd.rawDatPrep(mass,scan,cleanWM=True))#tBasedDframe = lmd.rawDatPrep(mass,scan,cleanWM=True)
  for i in range(len(singleScanFrames)):
    freakyFrame = lmd.makeUseable(singleScanFrames[i], resolution=res, noNaNsense=True, cropSparseEnds=True, verbose=False)#This is how you bin by frequency to get data like what you saw before
    plt.figure(1)
    plt.errorbar(freakyFrame["wavenumber_mean"],freakyFrame["signal_value"],yerr=freakyFrame['signal_uncertainty'],fmt='o-',alpha=.8,color=colorList[i],ecolor='k',markersize=6,label=str(scanList[i]))
    plt.figure(2)
    plt.errorbar(freakyFrame["wavenumber_mean"],freakyFrame["signal_value"],yerr=freakyFrame['signal_uncertainty'],fmt='o-',alpha=.8,color=colorList[i],ecolor='k',markersize=4,label=str(scanList[i]))
  
  print("yooo... I'm still running.")
  plt.figure(1)
  for i in range(len(scanCombos)):
    combo=scanCombos[i]
    freakyFrame = lmd.makeUseable(lmd.mergeDatRaw(mass,combo), resolution=res, noNaNsense=True, cropSparseEnds=True, verbose=False)#This is how you bin by frequency to get data like what you saw before
    plt.errorbar(freakyFrame["wavenumber_mean"],freakyFrame["signal_value"],yerr=freakyFrame['signal_uncertainty'],fmt='o-',alpha=.8,color=comboColors[i],ecolor='k',markersize=6,label=str(combo))
  plt.xlabel(r'Wavenumber (cm^{-1})'); plt.ylabel("Rate (counts/s)");
  plt.title(r'$Ra^{%d}F^{19}$,   $A^2\Pi_{1/2} \leftarrow X^2\Sigma^{+}$, $\Delta v=-1$'%(mass-19)+'\nScans %s,and all combinations thereof. Resolution=%.2f $cm^{-1}$'%(str(scanList), res))
  plt.legend(loc=3)

  print("yup, still running...")
  plt.figure(2) 
  for i in range(len(extraScans)):
    if extraScans[i]==2347: freakyFrame = lmd.makeUseable(lmd.rawDatPrep(mass,extraScans[i],cleanWM=True), resolution=res, noNaNsense=True, cropSparseEnds=True, verbose=True,ltrim=12800)
    else: freakyFrame = lmd.makeUseable(lmd.rawDatPrep(mass,extraScans[i],cleanWM=True), resolution=res, noNaNsense=True, cropSparseEnds=True, verbose=True)
    plt.errorbar(freakyFrame["wavenumber_mean"],freakyFrame["signal_value"],yerr=freakyFrame['signal_uncertainty'],fmt='o-',alpha=.5,color=extraColors[i], ecolor='k',markersize=4,label=str(extraScans[i]))
  plt.xlabel(r'Wavenumber (cm^{-1})'); plt.ylabel("Rate (counts/s)");
  plt.title(r'$Ra^{%d}F^{19}$,   $A^2\Pi_{1/2} \leftarrow X^2\Sigma^{+}$, $\Delta v=-1$'%(mass-19)+'\nScans: %s. Resolution=%.2f $cm^{-1}$'%(repr(np.append(scanList,extraScans)), res))
  plt.legend(loc=2)
  plt.show()