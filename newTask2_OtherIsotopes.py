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


def vLinePlotter(vlineArray, transitionLabel, vlabelArray, reflections=False, numericLabels=False):
  vlineArray=np.array(vlineArray);
  for i in range(len(vlineArray)):
    if i==-1: pass
    else:
      plt.axvline(vlineArray[i],0.05,.95, color='k', linestyle='dashed', linewidth=1, alpha=.75)
      if numericLabels:
        plt.annotate(s=vlabelArray[i]+"\n"+str(vlineArray[i]), xy=(vlineArray[i],.7+(i%6)/50), fontsize=6, ha='center', xycoords=('data','figure fraction'))
      else:
        plt.annotate(s=vlabelArray[i], xy=(vlineArray[i],.7+(i%6)/50), fontsize=6, ha='center', xycoords=('data','figure fraction'))
      if reflections:
        plt.axvline(vlineArray[i]-2*beta*gamma*vlineArray[i],0.05,.95, color='r', linestyle='dashed', linewidth=.25, alpha=.75)
        if numericLabels:
          plt.annotate(s=vlabelArray[i]+"\nAnticolinear\nReflection\n%.2f"%(vlineArray[i]-2*beta*gamma*vlineArray[i]), xy=(vlineArray[i]-2*beta*gamma*vlineArray[i],.13+(i%6)/50), fontsize=6, ha='center', xycoords=('data','figure fraction'))
        else:
          if i<3: plt.annotate(s=vlabelArray[i]+"\nReflection", xy=(vlineArray[i]-2*beta*gamma*vlineArray[i],.15+(i%6)/50), fontsize=6, ha='center', xycoords=('data','figure fraction'),color='red',alpha=.4)
  if not (np.all(vlineArray[0:6]==-1)): plt.annotate(s=r'$\Delta\mathit{v}=-1$', xy=(np.mean(np.ma.masked_where(vlineArray[0:6]==-1,vlineArray[0:6])), .05), fontsize=10, ha='center', xycoords=('data','figure fraction'))
  if not (np.all(vlineArray[6:12]==-1)): 
    plt.annotate(s=transitionLabel, xy=(np.mean(np.ma.masked_where(vlineArray[6:12]==-1,vlineArray[6:12])), .95), fontsize=16, ha='center', xycoords=('data','figure fraction'))
    plt.annotate(s=r'$\Delta\mathit{v}=0$', xy=(np.mean(np.ma.masked_where(vlineArray[6:12]==-1,vlineArray[6:12])), .05), fontsize=10, ha='center', xycoords=('data','figure fraction'))
  if not (np.all(vlineArray[12:18]==-1)): plt.annotate(s=r'$\Delta\mathit{v}=+1$', xy=(np.mean(np.ma.masked_where(vlineArray[12:18]==-1,vlineArray[12:18])), .05), fontsize=10, ha='center', xycoords=('data','figure fraction'))

'''2. "Do the same of the different isotopes 223-228Ra"'''
beta = 0.0005920684 #v_bunch/c
gamma = 1.00000017527255 #1/sqrt(1-beta^2)
vlabelArray = np.array(["6->5","5->4","4->3","3->2","2->1","1->0",
          "5->5","4->4","3->3","2->2","1->1","0->0",
          "5->6","4->5","3->4","2->3","1->2","0->1"])

'''All line positions observed/predicted from morse parameter analysis
vlineArrayPI12 = [12833.3, 12835.6, 12838., 12840.6, 12843.2, 12846.,   #P-band, \Delta v =-1
                  13254.6, 13260.3, 13266.2, 13272.2, 13278.3, 13284.5, #Q-band, \Delta v =-0
                  13670.2, 13679.3, 13688.5,13697.8, 13707.2, 13716.8]  #R-band, \Delta v =+1
vlineArrayDELTA32 = [14674.9, 14680.2, 14685.8, 14691.7, 14697.7, 14704.,
                     15096.2, 15105., 15114., 15123.3, 15132.8, 15142.5, 
                     15508.9, 15520.9, 15533.2, 15545.6, 15558.3, 15571.3]
vlineArrayDELTA52 = [15699., 15706.4, 15713.7, 15721.2, 15728.8, 15736.5,
                     16120.3, 16131.1, 16141.9, 16152.8, 16163.9, 16175.,
                     16531., 16545.1, 16559.3, 16573.5, 16587.9, 16602.4]
vlineArrayPI32 = [-1,-1,-1,-1,-1,-1, 
                  -1,-1,-1,15308, 15325, 15342,
                  -1,-1,-1,-1,-1,-1,]'''
# Will only include line in regions that were scanned
vlineArrayPI12 = [12833.3, 12835.6, 12838., 12840.6, 12843.2, 12846.,   #P-band, \Delta v =-1
                  13254.6, 13260.3, 13266.2, 13272.2, 13278.3, 13284.5, #Q-band, \Delta v =-0
                  13670.2, 13679.3, 13688.5,13697.8, 13707.2, 13716.8]  #R-band, \Delta v =+1
vlineArrayDELTA32 = [-1,-1,-1,-1,-1,-1,
                     15096.2, 15105., 15114., 15123.3, 15132.8, 15142.5, 
                     -1,-1,-1, -1, -1, -1]
vlineArrayDELTA52 = [-1,-1,-1,-1,-1,-1,
                     16120.3, 16131.1, 16141.9, 16152.8, 16163.9, 16175.,
                     -1,-1,-1,-1,-1,-1,]
vlineArrayPI32 = [-1,-1,-1,-1,-1,-1, 
                  -1,-1,-1,15308, 15325, 15342,
                  -1,-1,-1,-1,-1,-1,]

vlineArrayPI12Reflex = vlineArrayPI12 - 2*beta*gamma*np.array(vlineArrayPI12)

allScansBigDic = {}
for m in [241,242,243,244,245,247]:
  allScansBigDic[m] = lmd.makeScanToWavemeterDic(m, redo=False, verbose=False)

rewrite=True

"""for m in [241,242,243,244, 247]:
  scanInds = np.sort(np.array(list(allScansBigDic[m].keys())).astype(int))
  print("Doing new task. m = ",m)
  if not os.path.exists('./FrequencyConvertedDatasets/Plots/Mass%dPlots/'%m):
      os.mkdir('./FrequencyConvertedDatasets/Plots/Mass%dPlots/'%m)
  scanOverviewFigm = plt.figure('Mass: %d ; Scan Overview Figure'%m)
  plt.gcf().set_size_inches(20, 12)
  plt.title(r'Wavenumber Ranges of $^{%d}$Ra$^{19}$F Scans'%(m-19), fontsize=18)
  plt.xlabel(r'Wavenumber (cm$^{-1}$)', fontsize=18)
  plt.ylabel('Index of scan', fontsize=18)
  vLinePlotter(vlineArrayPI12, r'$X^2\Sigma^{+}_{1/2} \rightarrow A^2\Pi_{1/2}$', vlabelArray, reflections=True, numericLabels=False)
  vLinePlotter(vlineArrayPI32, r'$\rightarrow ^2\Pi_{3/2}$', vlabelArray, reflections=False, numericLabels=False)
  vLinePlotter(vlineArrayDELTA32, r'$\rightarrow ^2\Delta_{3/2}$', vlabelArray, reflections=False, numericLabels=False)
  vLinePlotter(vlineArrayDELTA52, r'$\rightarrow ^2\Delta_{5/2}$', vlabelArray, reflections=False, numericLabels=False)
  counter=0
  needPatches=[False,False,False,False]
  xMin = 20000; xMax=0
  for s in scanInds:
    print("s=",s)
    wmToUse = allScansBigDic[m][str(s)]#I hate that everything has to be stored with strings in json...
    wmToUse = "pdl" if wmToUse == "pdl" else int(wmToUse)
    needWrite = (not os.path.exists('./FrequencyConvertedDatasets/%d/scan_%d'%(m,s))) or rewrite
    resolution = .1
    prdf = lmd.rawDatPrep(m, s, wmToUse, cleanWM=True, verbose=False)#preppedRawDataFrame. Everything but together, but not yet binned by wavenumber
    newDataFrame = lmd.makeUseable(prdf, resolution=resolution, noNaNsense=True, cropSparseEnds=True)
    #newDataFrame = lmd.doEverything(m, s, wmToUse, resolution=resolution, writeToFile=needWrite, makePlot=False, cleanWM=True, verbose=False)
    xdat = newDataFrame.loc[:,'wavenumber_mean']; ydat = newDataFrame.loc[:,'signal_value']; sigydat = newDataFrame.loc[:,'signal_uncertainty']
    '''if (len(xdat)<100 and np.mean(sigydat)<1):
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
    plt.figure(scanOverviewFigm.number)'''
    xMin = min(xMin, np.min(xdat)); xMax = max(xMax, np.max(xdat))
    
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
  xRange=xMax-xMin; plt.xlim([xMin-.05*xRange, xMax+.05*xRange])
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


m=245
m245ScanstoInclude = [2130, 2131,2132,2323,2324,2325,2346,2360,2364,2365,2368,2375,2376,2135,2136,2137,2138,2139,2164,2165,2178,2309,2310,2317,2319,2320,2340,2341,2349,2350]
scanInds = np.sort(np.array(m245ScanstoInclude))
print("Doing new task. m = ",m)
if not os.path.exists('./FrequencyConvertedDatasets/Plots/Mass%dPlots/'%m):
  os.mkdir('./FrequencyConvertedDatasets/Plots/Mass%dPlots/'%m)
scanOverviewFigm = plt.figure('Mass: %d ; Scan Overview Figure'%m)
plt.gcf().set_size_inches(20, 12)
plt.title(r'Wavenumber Ranges of $^{%d}$Ra$^{19}$F Scans'%(m-19), fontsize=18)
plt.xlabel(r'Wavenumber (cm$^{-1}$)', fontsize=18)
plt.ylabel('Index of scan', fontsize=18)
plt.yticks(ticks=range(len(m245ScanstoInclude)), labels=np.sort(np.array(m245ScanstoInclude)))
counter=0
needPatches=[False,False,False,False]
xMin = 20000; xMax=0
for s in scanInds:
  print("s=",s)
  wmToUse = allScansBigDic[m][str(s)]#I hate that everything has to be stored with strings in json...
  wmToUse = "pdl" if wmToUse == "pdl" else int(wmToUse)
  needWrite = (not os.path.exists('./FrequencyConvertedDatasets/%d/scan_%d'%(m,s))) or rewrite
  resolution = .1
  prdf = lmd.rawDatPrep(m, s, wmToUse, cleanWM=True, verbose=False)#preppedRawDataFrame. Everything but together, but not yet binned by wavenumber
  newDataFrame = lmd.makeUseable(prdf, resolution=resolution, noNaNsense=True, cropSparseEnds=True)
  #newDataFrame = lmd.doEverything(m, s, wmToUse, resolution=resolution, writeToFile=needWrite, makePlot=False, cleanWM=True, verbose=False)
  xdat = newDataFrame.loc[:,'wavenumber_mean']; ydat = newDataFrame.loc[:,'signal_value']; sigydat = newDataFrame.loc[:,'signal_uncertainty']
  '''if (len(xdat)<100 and np.mean(sigydat)<1):
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
  plt.figure(scanOverviewFigm.number)'''
  xMin = min(xMin, np.min(xdat)); xMax = max(xMax, np.max(xdat))
  
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
xRange=xMax-xMin; plt.xlim([xMin-.05*xRange, xMax+.05*xRange])
red_patch = mpatches.Patch(color='red', label="Dye Laser Scans")
green_patch = mpatches.Patch(color='green', label="Injected Ti:Sa (HighRes)?")
blue_patch = mpatches.Patch(color='blue', label="Grating Ti:Sa Scans")
yellow_patch = mpatches.Patch(color='yellow', label="laser source unclear")
patchList=[red_patch,green_patch,blue_patch,yellow_patch]; handleList=[];
for i in range(len(needPatches)):
  if needPatches[i]:handleList.append(patchList[i])

vLinePlotter(vlineArrayPI12, r'$X^2\Sigma^{+}_{1/2} \rightarrow A^2\Pi_{1/2}$', vlabelArray, reflections=True, numericLabels=False)
vLinePlotter(vlineArrayPI32, r'$\rightarrow ^2\Pi_{3/2}$', vlabelArray, reflections=False, numericLabels=False)
vLinePlotter(vlineArrayDELTA32, r'$\rightarrow ^2\Delta_{3/2}$', vlabelArray, reflections=False, numericLabels=False)
vLinePlotter(vlineArrayDELTA52, r'$\rightarrow ^2\Delta_{5/2}$', vlabelArray, reflections=False, numericLabels=False)
lgd = plt.legend(loc="upper right", handles=handleList, fontsize=16, bbox_to_anchor=(1,1.0+.07*len(handleList)))
plt.savefig('./OverviewFigures/%dScanOverviewFigure_LowResSubsetAnalyzedForNaturePaper.png'%m, bbox_extra_artists=(lgd,))#, bbox_inches='tight')
#plt.close()
"""
m=245
scanInds = np.sort(np.array(list(allScansBigDic[m].keys())).astype(int))
print("Doing same task but in full. m = ",m)
if not os.path.exists('./FrequencyConvertedDatasets/Plots/Mass%dPlots/'%m):
  os.mkdir('./FrequencyConvertedDatasets/Plots/Mass%dPlots/'%m)
scanOverviewFigm = plt.figure('Mass: %d ; Scan Overview Figure'%m)
plt.gcf().set_size_inches(20, 12)
plt.title(r'Wavenumber Ranges of $^{%d}$Ra$^{19}$F Scans'%(m-19), fontsize=18)
plt.xlabel(r'Wavenumber (cm$^{-1}$)', fontsize=18)
plt.ylabel('Index of scan', fontsize=18)
plt.yticks(ticks=range(len(scanInds)), labels=scanInds)
vLinePlotter(vlineArrayPI12, r'$X^2\Sigma^{+}_{1/2} \rightarrow A^2\Pi_{1/2}$', vlabelArray, reflections=True, numericLabels=False)
vLinePlotter(vlineArrayPI32, r'$\rightarrow ^2\Pi_{3/2}$', vlabelArray, reflections=False, numericLabels=False)
vLinePlotter(vlineArrayDELTA32, r'$\rightarrow ^2\Delta_{3/2}$', vlabelArray, reflections=False, numericLabels=False)
vLinePlotter(vlineArrayDELTA52, r'$\rightarrow ^2\Delta_{5/2}$', vlabelArray, reflections=False, numericLabels=False)
counter=0
needPatches=[False,False,False,False]
xMin = 20000; xMax=0
for s in scanInds:
  print("s=",s)
  wmToUse = allScansBigDic[m][str(s)]#I hate that everything has to be stored with strings in json...
  wmToUse = "pdl" if wmToUse == "pdl" else int(wmToUse)
  if wmToUse==1: print("purportedly a high res scan, will pass."); pass
  else:
    needWrite = (not os.path.exists('./FrequencyConvertedDatasets/%d/scan_%d'%(m,s))) or rewrite
    resolution = .1
    prdf = lmd.rawDatPrep(m, s, wmToUse, cleanWM=True, verbose=False)#preppedRawDataFrame. Everything but together, but not yet binned by wavenumber
    newDataFrame = lmd.makeUseable(prdf, resolution=resolution, noNaNsense=True, cropSparseEnds=True)
    #newDataFrame = lmd.doEverything(m, s, wmToUse, resolution=resolution, writeToFile=needWrite, makePlot=False, cleanWM=True, verbose=False)
    xdat = newDataFrame.loc[:,'wavenumber_mean']; ydat = newDataFrame.loc[:,'signal_value']; sigydat = newDataFrame.loc[:,'signal_uncertainty']
    '''if (len(xdat)<100 and np.mean(sigydat)<1):
      resolution=float(max(xdat)-min(xdat))/100.
      print("sigh. Redoing scan at %.3f resolution"%resolution)
      newDataFrame = lmd.makeUseable(prdf, resolution=resolution, noNaNsense=True, cropSparseEnds=True)
      xdat = newDataFrame.loc[:,'wavenumber_mean']; ydat = newDataFrame.loc[:,'signal_value']; sigydat = newDataFrame.loc[:,'signal_uncertainty']'''
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
    xMin = min(xMin, np.min(xdat)); xMax = max(xMax, np.max(xdat))
    
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
xRange=xMax-xMin; plt.xlim([xMin-.05*xRange, xMax+.05*xRange])
red_patch = mpatches.Patch(color='red', label="COBRA Scans")
green_patch = mpatches.Patch(color='green', label="Injected Ti:Sa (HighRes)?")
blue_patch = mpatches.Patch(color='blue', label="Grating Ti:Sa (LowRes)")
yellow_patch = mpatches.Patch(color='yellow', label="laser source unclear")
patchList=[red_patch,green_patch,blue_patch,yellow_patch]; handleList=[];
for i in range(len(needPatches)):
  if needPatches[i]:handleList.append(patchList[i])
lgd = plt.legend(loc=1, handles=handleList, fontsize=16)
vLinePlotter(vlineArrayPI12, r'$X^2\Sigma^{+}_{1/2} \rightarrow A^2\Pi_{1/2}$', vlabelArray, reflections=True, numericLabels=False)
vLinePlotter(vlineArrayPI32, r'$\rightarrow ^2\Pi_{3/2}$', vlabelArray, reflections=False, numericLabels=False)
vLinePlotter(vlineArrayDELTA32, r'$\rightarrow ^2\Delta_{3/2}$', vlabelArray, reflections=False, numericLabels=False)
vLinePlotter(vlineArrayDELTA52, r'$\rightarrow ^2\Delta_{5/2}$', vlabelArray, reflections=False, numericLabels=False)
plt.savefig('./OverviewFigures/%dScanOverviewFigure_AllScansExceptHighRes.png'%m, bbox_extra_artists=(lgd,), bbox_inches='tight')"""
plt.show()
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