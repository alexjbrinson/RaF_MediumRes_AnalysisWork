import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import matplotlib.patches as mpatches
#from matplotlib.widgets import TextBox
#from matplotlib.widgets import Button
import os
#import os.path

"""TODO: A lotta stuff...
1. Incorporate iscool voltages to correct for doppler shift variation
2. Figure out what to do with wavemeter_pdl?
3. How to know which wavemeter to use???
4. turn all of this messy script into nice clean functions that I can import as a module
"""
def cleanDataSet(dRay):
  """crops data in v-space to remove sparse, pure-noise portions of scans"""
  meanSpacing = np.mean(dRay[1:,2]-dRay[:-1,2])
  i = 0
  while (dRay[i+1,2]-dRay[i,2]>3*meanSpacing) or (dRay[i+2,2]-dRay[i+1,2]>3*meanSpacing) or (dRay[i+3,2]-dRay[i+2,2]>3*meanSpacing):
  # If the any of the next 3 v-spacings are greater than 3 times the average spacing, increment the index at which to start cropping.
    i+=1
  len1 = len(dRay[:,2])
  len2 = len(dRay[i:,2])
  #print("test5: len1 = %d, len2 = %d"%(len1,len2))
  return dRay[i:,:]

#def loadRawDataset(directory, scanID):
  #TODO
#dataDir = os.listdir('RaF_RawData')

#cwd = os.getcwd()

def getScanDir(m, scanInd):
  return('../RaF_RawData/'+str(m)+'/scan_'+str(scanInd)) #Okay so maybe this didn't need to be a function...

def readRawData(m, scanInd, wavenumber, nums):
  return()#TODO

mass = 245
scanIndex = 2462
wmNum = 1
numBins = 500
wavenumberToUse = "wavenumber_"+str(wmNum)
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
"""
if scanDataDic['has_iscool'] == True:
  mfouter = pd.merge_ordered(mfouter, ic, on='timestamp', how='outer')
if scanDataDic['has_wavemeter_pdl'] == True:
  mfouter = pd.merge_ordered(mfouter, pdl, on='timestamp', how='outer')
"""
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

#print("TEST:\n", pd.concat([mfouter.loc[:,:"offset_y"], mfouter.loc[:,"wavenumber_1":"wavenumber_4"].fillna(method="backfill")], axis=1))
mfouterBf = mfouter.copy(); mfouterBf.loc[:,"wavenumber_1":"wavenumber_4"].fillna(method='backfill',inplace=True)
mfouterFf = mfouter.copy(); mfouterFf.loc[:,"wavenumber_1":"wavenumber_4"].fillna(method='ffill',inplace=True)
#mfouter.loc[:,"wavenumber_1":"wavenumber_4"].fillna(method="backfill",inplace=True) 
print("TEST2:\n", mfouterBf.loc[:49,["timestamp","bunch_no","events_per_bunch", wavenumberToUse]])
#print("TEST3:\n", mfouterFf)
#print("TEST4:\n", mfouter)

"""mfouterBf.loc[::50,"wavenumber_1":"wavenumber_4"].plot()
plt.title("wavenumber_1-4 vs. dataframe index")"""

mfouterBf["events_per_bunch"]=mfouterBf["events_per_bunch"].map(lambda a: 1 if a > 0 else a)
print("TEST5:\n", mfouterBf.loc[:49,"timestamp":wavenumberToUse])
mfouterBf = mfouterBf[pd.notna(mfouterBf['bunch_no'])]
#print("TEST6:\n", mfouterBf.loc[:49,["timestamp","bunch_no","events_per_bunch","wavenumber_2"]])

"""Primitive Histograms for time series and frequency binned (not rate normalized!) counts"""
"""
mfouterBf[mfouterBf['events_per_bunch']>0].hist(column="wavenumber_2", bins=1000)
plt.title("wavenumber_2\n Scan: %d Mass: %d"%(scanIndex, mass))
mfouterBf[mfouterBf['events_per_bunch']>0].hist(column="timestamp", bins=900)
plt.title("timestamp\n Scan: %d Mass: %d"%(scanIndex, mass))
"""
tStamps = np.array(mfouterBf.loc[:,'timestamp'])
mfouterBf.loc[0:,'timeDiffs'] = pd.Series(np.append(0,tStamps[1:]-tStamps[:-1]), index=mfouterBf.index[0:])
#print(mfouterBf.loc[:49,["timestamp","timeDiffs","bunch_no","events_per_bunch","wavenumber_2"]])

kBins = pd.cut(mfouterBf.loc[:,wavenumberToUse], bins=numBins)

print("test8.\n", mfouterBf.groupby(kBins) )
"""
aggDat = mfouterBf.groupby(kBins).agg({'wavenumber_2':['mean'], 'events_per_bunch':['sum'], 'timeDiffs':['sum'], 'timestamp':['min','max']})
print("test9:\n", aggDat)
print("test10. dataframe:\n", aggDat[[("wavenumber_2",'mean'),('events_per_bunch','sum'), ('timeDiffs','sum')]] )
#assert(np.all(np.array(aggDat.loc[:,('timeDiffs','sum')]) == np.array(aggDat.loc[:,('timestamp','max')]) - np.array(aggDat.loc[:,('timestamp','min')]) ))
#This assertion will fail because the actual time spent in the wavenumber bin is max(tstamp) - min(tstamp), _plus_ time spent between max(tstamp) and start of next bin.
#print(np.c_[np.array(aggDat.loc[:,('timeDiffs','sum')]), np.array(aggDat.loc[:,('timestamp','max')]) - np.array(aggDat.loc[:,('timestamp','min')])] )
"""

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