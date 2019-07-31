import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import matplotlib.patches as mpatches
import csv
#from matplotlib.widgets import TextBox
#from matplotlib.widgets import Button
import os
#import os.path

"""TODO: A lotta stuff...
1. Figure out what to do with wavemeter_pdl?
2. How to know which wavemeter to use???
"""

"""
wavemeter_1 = injection seeded (at least for scans 2170-2176...)
wavemeter_2 = 
wavemeter_3 = 
wavemeter_4 = 
"""

def computeBeta(m, voltage):
  #computes bunch velocity from isotope mass and iscool voltage T = m/2 v^2 ==> v = sqrt(2*T/m)
  amu2eV = 931494102 #1 amu(*c^2) ~= 931494273 eV
  beta = math.sqrt(2*voltage/(m*amu2eV))
  return(beta)

def dopplerCorrectionFactor(m, voltage):
  #uses isotope mass and iscool voltage to compute doppler correction factor for wavenumber measurements
  amu2eV = 931494102 #1 amu(*c^2) ~= 931494273 eV
  beta = math.sqrt(2*voltage/(m*amu2eV))
  gamma = 1/math.sqrt(1-beta**2)
  dcf = gamma*(1+beta)
  return(dcf)

def wavemeterDataCleaner(df, wavenumberToUse):
  #Removes garbage wavemeter data
  wmCopy = df.sort_values(by='timestamp')
  wmVals = np.array(wmCopy.loc[:,wavenumberToUse])
  wmtypicalDiff = np.mean(np.abs(wmVals[1:]-wmVals[:-1]))
  for i in range(len(wmVals)-1):
    if np.abs(wmVals[i+1]-wmVals[i])>max(20*wmtypicalDiff,20):
      print("woah dere. wmVal[%d] = %f is way out there, yo"%(i+1,wmVals[i+1]))
      assert(wmCopy.loc[i+1,wavenumberToUse]==wmVals[i+1])
      for j in range(i+1,len(wmVals)-1):
        if np.abs(wmVals[j+1]-wmVals[j])>max(20*wmtypicalDiff,20) and np.abs(wmVals[j+1]-wmVals[i]<max(20*wmtypicalDiff,20)):
          print("bad data segment ends at wmVal[%d] = %f " %(j+1,wmVals[j+1]))
          wmCopy.loc[i+1:j,wavenumberToUse]= float('nan')*np.ones(j-i)
          i = j+1
          break
    elif wmVals[i]<0:
      print("woah dere. wmVal[%d] = %f is negative, yo"%(i, wmVals[i]))
      wmCopy.loc[i, wavenumberToUse] = float('nan')
  wmCleaned = wmCopy[pd.notna(wmCopy[wavenumberToUse])]
  #print("TEST. wmCleaned:\n", wmCleaned)
  return(wmCleaned)

def rawDatPrep(m, scanInd, wavenumber, verbose=False, cleanWM=False):
  #TODO: function description
  mass = m
  scanIndex = scanInd
  wmNum = wavenumber
  wavenumberToUse = "wavenumber_"+str(wmNum)
  scanDir = '../RaF_RawData/'+str(mass)+'/scan_'+str(scanIndex)+"/"
  dSetTypes = ['iscool', 'tagger', 'wavemeter', 'wavemeter_pdl']
  scanDataDic = {}
  scanDataDic['has_iscool'] = False; scanDataDic['has_tagger'] = False; scanDataDic['has_wavemeter'] = False; scanDataDic['has_wavemeter_pdl'] = False;
  dirlist=os.listdir(scanDir)

  for i in range(len(dirlist)):
    if dirlist[i]=='metadata_iscool_ds.txt':
      iscool_colNames = ['timestamp', 'offset', 'voltage']
      ic = pd.read_csv(scanDir + "iscool_ds.csv", sep=';', names=iscool_colNames)
      #betaFunc = np.vectorize(computeBeta, excluded=['m'])
      ic['betaVals'] = ic['voltage'].map(lambda V: computeBeta(m, V))
      ic['dopplerShiftFactor'] = ic['voltage'].map(lambda V: dopplerCorrectionFactor(m, V))
      scanDataDic['iscool'] = ic
      scanDataDic['has_iscool'] = True

    elif dirlist[i] == 'metadata_tagger_ds.txt':
      #tagger_mdFile = open()
      tag_colNames = ['timestamp', 'offset', 'bunch_no', 'events_per_bunch', 'channel', 'delta_t']
      tag = pd.read_csv(scanDir + "tagger_ds.csv", sep=';', names=tag_colNames)
      scanDataDic['tagger'] = tag
      scanDataDic['has_tagger'] = True #I want to call this bool ['is_cool'], but I guess I'll be informative instead :(

    elif dirlist[i] == 'metadata_wavemeter_ds.txt':
      wm_colNames = ['timestamp', 'offset', 'wavenumber_1', 'wavenumber_2', 'wavenumber_3', 'wavenumber_4']
      wm = pd.read_csv(scanDir + "wavemeter_ds.csv", sep=';', names=wm_colNames)
      if cleanWM == True : wm = wavemeterDataCleaner(wm, wavenumberToUse)
      scanDataDic['wavemeter'] = wm
      scanDataDic['has_wavemeter'] = True  

    elif dirlist[i] == 'metadata_wavemeter_ds.txt':
      pdl_colNames = ['timestamp', 'offset', 'wavenumber_pdl']
      pdl = pd.read_csv(scanDir + "wavemeter_pdl_ds.csv", sep=';', names=pdl_colNames)
      scanDataDic['wavemeter_pdl'] = pdl
      scanDataDic['has_wavemeter_pdl'] = True

  mfouter = pd.merge_ordered(tag, wm, on='timestamp', how='outer')# every scan _should_ have tagger and wavemeter data (or else what's the point?)
  if scanDataDic['has_iscool'] == True:
    mfouter = pd.merge_ordered(mfouter, ic, on='timestamp', how='outer')
  if scanDataDic['has_wavemeter_pdl'] == True:
    mfouter = pd.merge_ordered(mfouter, pdl, on='timestamp', how='outer')
  #mfouter.loc[:,"wavenumber_1":"wavenumber_4"].fillna(method='backfill',inplace=True) #".loc indexed to a list of columns won't support inplace operations"...
  mfouter.loc[:,"wavenumber_1":"wavenumber_4"] = mfouter.loc[:,"wavenumber_1":"wavenumber_4"].fillna(method='backfill')
  if scanDataDic['has_iscool'] == True:
       mfouter.loc[:,['voltage','betaVals','dopplerShiftFactor']] = mfouter.loc[:,['voltage','betaVals','dopplerShiftFactor']].fillna(method='backfill')
  if scanDataDic['has_wavemeter_pdl'] == True:
       mfouter.loc[:,'wavenumber_pdl'] = mfouter.loc[:,'wavenumber_pdl'].fillna(method='backfill')

  if verbose: print("TEST2:\n", mfouter.loc[:49,["timestamp","events_per_bunch", wavenumberToUse, 'voltage' if scanDataDic['has_iscool'] == True else 'bunch_no']])

  mfouter["events_per_bunch"]=mfouter["events_per_bunch"].map(lambda a: 1 if a > 0 else a)

  if verbose: print("TEST5:\n", mfouter.loc[:49,"timestamp":wavenumberToUse])
  mfouter = mfouter[pd.notna(mfouter['bunch_no'])]
  mfouter = mfouter[pd.notna(mfouter[wavenumberToUse])]
  if verbose: print("TEST6:\n", mfouter.loc[:49,["timestamp","bunch_no","events_per_bunch",wavenumberToUse,'betaVals','dopplerShiftFactor']])
  tStamps = np.array(mfouter.loc[:,'timestamp'])
  mfouter.loc[0:,'timeDiffs'] = pd.Series(np.append(0,tStamps[1:]-tStamps[:-1]), index=mfouter.index[0:])
  if scanDataDic['has_iscool'] == True:
    #mfouter.loc[:,'wavenumber'] = mfouter.loc[:,wavenumberToUse]*mfouter.loc[:, 'dopplerShiftFactor'] #FOUND ERROR IN PAPER
    mfouter.loc[:,'wavenumber'] = mfouter.loc[:,wavenumberToUse]/mfouter.loc[:, 'dopplerShiftFactor']
  #print(mfouterBf.loc[:49,["timestamp","timeDiffs","bunch_no","events_per_bunch","wavenumber_2"]])
  else:
    print("YO... No iscool data. How am I supposed to correct these wavenumber measurements?!?")
    nextScan=scanInd
    conditionMet = False
    while conditionMet == False:
      nextScan+=1
      try:
        nextDirList = os.listdir('../RaF_RawData/'+str(mass)+'/scan_'+str(nextScan)+"/")
        if 'metadata_iscool_ds.txt' in nextDirList:
          #nextIsCool = open('../RaF_RawData/'+str(mass)+'/scan_'+str(nextScan)+'/iscool_ds.csv','r')
          #with open('../RaF_RawData/'+str(mass)+'/scan_'+str(nextScan)+'/iscool_ds.csv', newline='') as f:
          #  reader = csv.reader(f)
          #  row1 = next(reader)
          isCoolVoltage = np.loadtxt('../RaF_RawData/'+str(mass)+'/scan_'+str(nextScan)+'/iscool_ds.csv',delimiter=';',max_rows=1)[-1]
          print("Using Scan %d initial isCool reading; Voltage=%d"%(nextScan, isCoolVoltage))
          #nextIsCool.close()
          dcf = dopplerCorrectionFactor(m, isCoolVoltage)
          print("dcf=%.4f"%dcf)
          #mfouter.loc[:,'wavenumber'] = mfouter.loc[:,wavenumberToUse]*dcf#FOUND ERROR IN PAPER
          mfouter.loc[:,'wavenumber'] = mfouter.loc[:,wavenumberToUse]/dcf
          break
      except OSError:
        conditionMet = False
  preppedDataFrame = mfouter.loc[:,["timestamp", 'timeDiffs', 'wavenumber', 'events_per_bunch']].copy()
  del(mfouter)

  return(preppedDataFrame)#TODO add in other wavenumber correction thing

def makeUseable(df, nBins=100, resolution=-1):
  
  kVals = np.array(df.loc[:,"wavenumber"]); kRange=max(kVals)-min(kVals)
  print("testing wavenumber range: min=%.3f; max=%.3f"%(min(kVals),max(kVals)))
  if resolution<0: binQuant = nBins
  else: binQuant = math.ceil(kRange/resolution)
  print("TESTSTSSTSTS: numBins=%d"%binQuant)
  kBins = pd.cut(df.loc[:,"wavenumber"], bins=binQuant)#, retbins=True)

  print("test8.\n", df.groupby(kBins).head() )
  #print("test8.\n", kBins )

  aggDat = df.groupby(kBins).agg({'wavenumber':['mean', 'min', 'max'], 'events_per_bunch':['sum'], 'timeDiffs':['sum']}).reset_index() #Wtf apparently reset_index() is p important... 

  """outputDF = pd.DataFrame({"wavenumber_mean"   : aggDat.loc[:,(wavenumberToUse,'mean')],
                        "signal_value"         : aggDat.loc[:,('events_per_bunch','sum')]/aggDat.loc[:,('timeDiffs','sum')],
                        "signal_uncertainty"   : np.sqrt(aggDat.loc[:,('events_per_bunch','sum')])/aggDat.loc[:,('timeDiffs','sum')],
                        "measurement_duration" : aggDat.loc[:,('timeDiffs','sum')], 
                        "wavenumber_lowerUncert"     : aggDat.loc[:,(wavenumberToUse,'mean')]-aggDat.loc[:,(wavenumberToUse,'min')],
                        "wavenumber_upperUncert"     : aggDat.loc[:,(wavenumberToUse,'max')]-aggDat.loc[:,(wavenumberToUse,'mean')]},index=range(len(kBins) ) )"""
  outputDF = pd.DataFrame({"wavenumber_mean"   : aggDat.loc[:,('wavenumber','mean')],
                        "signal_value"         : aggDat.loc[:,('events_per_bunch','sum')]/aggDat.loc[:,('timeDiffs','sum')],
                        "signal_uncertainty"   : np.sqrt(aggDat.loc[:,('events_per_bunch','sum')])/aggDat.loc[:,('timeDiffs','sum')],
                        "measurement_duration" : aggDat.loc[:,('timeDiffs','sum')]},index=range(binQuant) )
  return(outputDF)

def plotData(output, m, scanInd, wavenumber, nBins=-1, resolution=-1):
  if resolution ==-1:
    plt.figure("output Plot, mass: %d scan: "%m +str(scanInd)+ " wavenumber: %d numBins: %d"%(wavenumber, len(output.loc[:,'wavenumber_mean'])) )
    plt.title('Mass: %d ; scan: '%m +str(scanInd)+ ' wavemeter_%d\ncount rate vs wavenumber for %d wavenumber bins'%(wavenumber, len(output.loc[:,'wavenumber_mean'])))
  else:
    plt.figure('output Plot, mass: %d scan: '%m +str(scanInd)+ ' wavenumber: %d resolution: %.3f '%(wavenumber, resolution) )
    plt.title(r'Mass: %d ; scan: '%m +str(scanInd)+ ' wavemeter_%d\ncount rate vs wavenumber at %.3f $cm^{-1}$ resolution'%(wavenumber, resolution))
  plt.errorbar(x=output.loc[:,'wavenumber_mean'], y=output.loc[:,'signal_value'], yerr=output.loc[:,'signal_uncertainty'], fmt="bo",ecolor='k')#, xerr = kBins)
  plt.xlabel(r'wavenumber ($cm^{-1}$)')
  plt.ylabel('rate (counts/s?) TODO: determine unit on timestamp')

def doEverything(m, scanInd, wavenumber, nBins=100):
  mfba =  rawDatPrep(m, scanInd, wavenumber)
  output = makeUseable(mfba, nBins=nBins)
  plotData(output, m, scanInd, wavenumber, nBins=nBins)

if __name__ == '__main__':

  
  mass = 245
  scanIndex = 2310
  wmNum = 2
  numBins = 500

  mfba =  rawDatPrep(mass, scanIndex, wmNum)
  print("test 9:\n", mfba.head)
  print("test 10:\n", mfba.tail())
  output = makeUseable(mfba, nBins=numBins)
  print("test11:\n", output)
  plotData(output, mass, scanIndex, wmNum, nBins=numBins)
    
  #New merging thing?
  mass = 245
  indices=[2170,2171,2172,2173,2175,2176]
  wmNum = 1
  res=.01

  dfs=[]
  for ind in indices:
    dfs.append(rawDatPrep(mass,ind,wmNum))
  df=pd.concat(dfs)
  print("test whatever:\n", df)
  audi=makeUseable(df, resolution=res)
  print("test whatever+1:\n", audi)
  plotData(audi, mass, indices, wmNum, resolution=res)
  """m2=243
  indices=[2300,2302,2303,2283]
  wmNum = 2
  res=.1

  dfs=[]
  for ind in indices:
    dfs.append(rawDatPrep(m2,ind,wmNum))
  df=pd.concat(dfs)
  print("test whatever:\n", df)
  audi=makeUseable(df, resolution=res)
  print("test whatever+1:\n", audi)
  plotData(audi, m2, indices, wmNum, resolution=res)"""

  plt.show()
