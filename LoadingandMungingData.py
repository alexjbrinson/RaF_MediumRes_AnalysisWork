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
1. Incorporate iscool voltages to correct for doppler shift variation
2. Figure out what to do with wavemeter_pdl?
3. How to know which wavemeter to use???
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

def getScanDir(m, scanInd):
  return('../RaF_RawData/'+str(m)+'/scan_'+str(scanInd)) #Okay so maybe this didn't need to be a function...

def rawDatPrep(m, scanInd, wavenumber):
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

  print("TEST2:\n", mfouter.loc[:49,["timestamp","events_per_bunch", wavenumberToUse, 'voltage' if scanDataDic['has_iscool'] == True else 'bunch_no']])

  mfouter["events_per_bunch"]=mfouter["events_per_bunch"].map(lambda a: 1 if a > 0 else a)
  print("TEST5:\n", mfouter.loc[:49,"timestamp":wavenumberToUse])
  mfouter = mfouter[pd.notna(mfouter['bunch_no'])]
  mfouter = mfouter[pd.notna(mfouter[wavenumberToUse])]
  print("TEST6:\n", mfouter.loc[:49,["timestamp","bunch_no","events_per_bunch",wavenumberToUse,'betaVals','dopplerShiftFactor']])
  tStamps = np.array(mfouter.loc[:,'timestamp'])
  mfouter.loc[0:,'timeDiffs'] = pd.Series(np.append(0,tStamps[1:]-tStamps[:-1]), index=mfouter.index[0:])
  if scanDataDic['has_iscool'] == True:
    mfouter.loc[:,'wavenumber'] = mfouter.loc[:,wavenumberToUse]*mfouter.loc[:, 'dopplerShiftFactor']
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
          print("dcf=%d"%dcf)
          mfouter.loc[:,'wavenumber'] = mfouter.loc[:,wavenumberToUse]*dcf
          break
      except OSError:
        conditionMet = False
  preppedDataFrame = mfouter.loc[:,["timestamp", 'timeDiffs', 'wavenumber', 'events_per_bunch']].copy()
  del(mfouter)

  return(preppedDataFrame)#TODO add in other wavenumber correction thing

def makeUseable(df, nBins=100):
  wavenumberToUse = "wavenumber"
  kBins = pd.cut(df.loc[:,wavenumberToUse], bins=nBins)#, retbins=True)

  #print("test8.\n", df.groupby(kBins) )
  print("test8.\n", kBins )

  aggDat = df.groupby(kBins).agg({wavenumberToUse:['mean', 'min', 'max'], 'events_per_bunch':['sum'], 'timeDiffs':['sum']}).reset_index() #Wtf apparently reset_index() is p important... 

  """outputDF = pd.DataFrame({"wavenumber_mean"   : aggDat.loc[:,(wavenumberToUse,'mean')],
                        "signal_value"         : aggDat.loc[:,('events_per_bunch','sum')]/aggDat.loc[:,('timeDiffs','sum')],
                        "signal_uncertainty"   : np.sqrt(aggDat.loc[:,('events_per_bunch','sum')])/aggDat.loc[:,('timeDiffs','sum')],
                        "measurement_duration" : aggDat.loc[:,('timeDiffs','sum')], 
                        "wavenumber_lowerUncert"     : aggDat.loc[:,(wavenumberToUse,'mean')]-aggDat.loc[:,(wavenumberToUse,'min')],
                        "wavenumber_upperUncert"     : aggDat.loc[:,(wavenumberToUse,'max')]-aggDat.loc[:,(wavenumberToUse,'mean')]},index=range(len(kBins) ) )"""
  outputDF = pd.DataFrame({"wavenumber_mean"   : aggDat.loc[:,(wavenumberToUse,'mean')],
                        "signal_value"         : aggDat.loc[:,('events_per_bunch','sum')]/aggDat.loc[:,('timeDiffs','sum')],
                        "signal_uncertainty"   : np.sqrt(aggDat.loc[:,('events_per_bunch','sum')])/aggDat.loc[:,('timeDiffs','sum')],
                        "measurement_duration" : aggDat.loc[:,('timeDiffs','sum')]},index=range(len(kBins) ) )
  return(outputDF)

def plotData(output, m, scanInd, wavenumber, nBins=-1):
  plt.figure("output Plot, mass: %d scan: %d wavenumber: %d numBins: %d"%(m, scanInd, wavenumber, nBins) )
  plt.errorbar(x=output.loc[:,'wavenumber_mean'], y=output.loc[:,'signal_value'], yerr=output.loc[:,'signal_uncertainty'], fmt="b-",ecolor='k')#, xerr = kBins)
  plt.title("Mass: %d ; scan: %d wavemeter_%d\ncount rate vs wavenumber for %d wavenumber bins"%(m, scanInd, wavenumber, nBins))
  plt.xlabel(r'wavenumber ($cm^{-1}$)')
  plt.ylabel('rate (counts/s?) TODO: determine unit on timestamp')

def doEverything(m, scanInd, wavenumber, nBins=100):
  mfba =  rawDatPrep(m, scanInd, wavenumber)
  output = makeUseable(mfba, nBins=nBins)
  plotData(output, m, scanInd, wavenumber, nBins=nBins)

if __name__ == '__main__':

  mass = 245
  scanIndex = 2133
  wmNum = 2
  numBins = 500

  mfba =  rawDatPrep(mass, scanIndex, wmNum)
  print("test 9:\n", mfba.head)
  print("test 10:\n", mfba.tail())
  output = makeUseable(mfba, nBins=numBins)

  #print("test11:\n", output.loc[25:50,['wavenumber_mean','signal_value', 'signal_uncertainty']])
  print("test11:\n", output.iloc[-50:-1,:])
  #print("test11:\n", output.loc[25:50,['wavenumber_lowerUncert','wavenumber_upperUncert']])

  """plt.figure("outputDF Plot")
  plt.errorbar(x=output.loc[:,'wavenumber_mean'], y=output.loc[:,'signal_value'], yerr=output.loc[:,'signal_uncertainty'], fmt="b-",ecolor='k')#, xerr = kBins)
  plt.title("Mass: %d ; scan: %d wavemeter_%d\ncount rate vs wavenumber for %d wavenumber bins"%(mass, scanIndex, wmNum, numBins))
  plt.xlabel(r'wavenumber ($cm^{-1}$)')
  plt.ylabel('rate (counts/s?) TODO: determine unit on timestamp')"""
  plotData(output, mass, scanIndex, wmNum, nBins=numBins)
  plt.show()

""" wavenumberToUse = "wavenumber_"+str(wmNum)
  scanDir = '../RaF_RawData/'+str(mass)+'/scan_'+str(scanIndex)+"/"
  dSetTypes = ['iscool', 'tagger', 'wavemeter', 'wavemeter_pdl']
  scanDataDic = {}
  scanDataDic['has_iscool'] = False; scanDataDic['has_tagger'] = False; scanDataDic['has_wavemeter'] = False; scanDataDic['has_wavemeter_pdl'] = False;

  dirlist=os.listdir(scanDir)
  print("test0. os.listdir('scans245'):\n",dirlist)
  for i in range(len(dirlist)):
    if dirlist[i]=='metadata_iscool_ds.txt':
      iscool_colNames = ['timestamp', 'offset', 'voltage']
      ic = pd.read_csv(scanDir + "iscool_ds.csv", sep=';', names=iscool_colNames)
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
      scanDataDic['wavemeter'] = wm
      scanDataDic['has_wavemeter'] = True

    elif dirlist[i] == 'metadata_wavemeter_ds.txt':
      pdl_colNames = ['timestamp', 'offset', 'wavenumber_1']
      pdl = pd.read_csv(scanDir + "wavemeter_pdl_ds.csv", sep=';', names=pdl_colNames)
      scanDataDic['wavemeter_pdl'] = pdl
      scanDataDic['has_wavemeter_pdl'] = True

  for d in dSetTypes:
    if scanDataDic['has_'+d] == True:
      df = scanDataDic[d]
      print("test"+d+":\n", df.loc[df.index[0]:df.index[16], "timestamp":df.columns[-1]])

  mfouter = pd.merge_ordered(tag, wm, on='timestamp', how='outer')
  print(mfouter.loc[25:75,["timestamp","events_per_bunch","wavenumber_1","wavenumber_4"]])

  plt.figure()
  mask1 = np.array(mfouter.isna().loc[:,"events_per_bunch"])# == False
  mask2 = np.array(mfouter.isna().loc[:,"wavenumber_1"])# == False
  xlen=1000
  x1dat = np.ma.masked_where(mask1[:xlen], mfouter.index[:xlen]).compressed()
  x2dat = np.ma.masked_where(mask2[:xlen], mfouter.index[:xlen]).compressed()
  y1dat = np.ma.masked_where(mask1[:xlen], mfouter.loc[:xlen-1,"timestamp"]).compressed()
  y2dat = np.ma.masked_where(mask2[:xlen], mfouter.loc[:xlen-1,"timestamp"]).compressed()
  print("x1length = %d; x2length = %d"%(len(x1dat),len(x2dat)))
  yTotDat = np.array(mfouter.loc[:,"timestamp"])
  print("timestamp monotonic?", np.all(yTotDat[1:]-yTotDat[:-1] >=0 ))
  plt.plot(x1dat, y1dat, "ro", markersize=3, label='tagger data')
  plt.plot(x2dat, y2dat, "bo", markersize=3, label='wavemeter data')
  plt.title("Comparing timestamps for raw tagger and wavemeter datasets")
  plt.xlabel("Index of entry in merged dataframe")
  plt.ylabel("timestamp")
  plt.legend(loc=5)
  plt.text(.6,.2,"Timestamps monotonic?\n yTotDat = np.array(mfouter.loc[:,'timestamp'])\nnp.all(yTotDat[1:]-yTotDat[:-1] >=0 ) = "+str( np.all(yTotDat[1:]-yTotDat[:-1] >=0 ) ), transform=plt.gca().transAxes)
  plt.close()

  mfouterBf = mfouter.copy(); mfouterBf.loc[:,"wavenumber_1":"wavenumber_4"].fillna(method='backfill',inplace=True)
  mfouterFf = mfouter.copy(); mfouterFf.loc[:,"wavenumber_1":"wavenumber_4"].fillna(method='ffill',inplace=True)
  print("TEST2:\n", mfouterBf.loc[:49,["timestamp","bunch_no","events_per_bunch", wavenumberToUse]])


  mfouterBf["events_per_bunch"]=mfouterBf["events_per_bunch"].map(lambda a: 1 if a > 0 else a)
  print("TEST5:\n", mfouterBf.loc[:49,"timestamp":wavenumberToUse])
  mfouterBf = mfouterBf[pd.notna(mfouterBf['bunch_no'])]
  tStamps = np.array(mfouterBf.loc[:,'timestamp'])
  mfouterBf.loc[0:,'timeDiffs'] = pd.Series(np.append(0,tStamps[1:]-tStamps[:-1]), index=mfouterBf.index[0:])
  #print(mfouterBf.loc[:49,["timestamp","timeDiffs","bunch_no","events_per_bunch","wavenumber_2"]])

  kBins = pd.cut(mfouterBf.loc[:,wavenumberToUse], bins=numBins)

  print("test8.\n", mfouterBf.groupby(kBins) )

  aggDat = mfouterBf.groupby(kBins).agg({wavenumberToUse:['mean', 'min', 'max'], 'events_per_bunch':['sum'], 'timeDiffs':['sum']}).reset_index() #Wtf apparently reset_index() is p important... 

  outputDF = pd.DataFrame({"wavenumber_mean"   : aggDat.loc[:,(wavenumberToUse,'mean')],
                        "signal_value"         : aggDat.loc[:,('events_per_bunch','sum')]/aggDat.loc[:,('timeDiffs','sum')],
                        "signal_uncertainty"   : np.sqrt(aggDat.loc[:,('events_per_bunch','sum')])/aggDat.loc[:,('timeDiffs','sum')],
                        "measurement_duration" : aggDat.loc[:,('timeDiffs','sum')], 
                        "wavenumber_lowerUncert"     : aggDat.loc[:,(wavenumberToUse,'mean')]-aggDat.loc[:,(wavenumberToUse,'min')],
                        "wavenumber_upperUncert"     : aggDat.loc[:,(wavenumberToUse,'max')]-aggDat.loc[:,(wavenumberToUse,'mean')]},index=range(len(kBins) ) )
  print("test11:\n", outputDF.loc[25:50,['wavenumber_mean','signal_value', 'signal_uncertainty']])
  print("test11:\n", outputDF.loc[25:50,['wavenumber_lowerUncert','wavenumber_upperUncert']])

  plt.figure("outputDF Plot")
  plt.errorbar(x=outputDF.loc[:,'wavenumber_mean'], y=outputDF.loc[:,'signal_value'], yerr=outputDF.loc[:,'signal_uncertainty'], fmt="b-",ecolor='k')#, xerr = kBins)
  plt.title("Mass: %d ; scan: %d wavemeter_%d\ncount rate vs wavenumber for %d wavenumber bins"%(mass, scanIndex, wmNum, numBins))
  plt.xlabel(r'wavenumber ($cm^{-1}$)')
  plt.ylabel('rate (counts/s?) TODO: determine unit on timestamp')

  plt.show()
"""