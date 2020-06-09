import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import matplotlib.patches as mpatches
import csv
#from matplotlib.widgets import TextBox
#from matplotlib.widgets import Button
import os
import json
#import os.path

"""TODO: A lotta stuff...
1. How to know which wavemeter to use??? (In Progress!)
2. Fix "dead time" error related to negative channel number (work in progress...) (or actually possible progress, but need to ask if it's reasonable to always implement)
3. change wavenumber statistic in makeUsable() to take weighted averages. #DONE! unless I should be weighting by events counted...
4. New Tasks...
5. Wavemeter correction for high res scans
"""

"""
wavemeter_1 = injection seeded (at least for scans 2170-2176...)
wavemeter_2 = Grating Ti:Sa
wavemeter_3 = 
wavemeter_4 = 
wavemeter_pdl = COBRA
"""

def computeBeta(m, voltage):
  #computes bunch velocity from isotope mass and iscool voltage T = m/2 v^2 ==> v = sqrt(2*T/m)
  amu2eV = np.int64(931494102) #1 amu(*c^2) ~= 931494273 eV
  beta = np.sqrt(2*voltage/(m*np.int64(amu2eV)))#math.sqrt(2*voltage/(m*amu2eV))
  return(beta)

def dopplerCorrectionFactor(m, voltage):
  #uses isotope mass and iscool voltage to compute doppler correction factor for wavenumber measurements
  amu2eV = np.int64(931494102) #1 amu(*c^2) ~= 931494273 eV
  beta = np.sqrt(2*voltage/(m*amu2eV))
  dcf = np.sqrt(1+beta)/np.sqrt(1-beta)
  return(dcf)

def whichWavemeter(mass, scanInd, verbose=False):
  #Determines which wavemeter should be used based on which has the largest range (post-cleaning)
  wmRanges = [-1,-1,-1,-1,-1]
  scanDir = '../RaF_RawData/'+str(mass)+'/scan_'+str(scanInd)+'/'
  dirlist=os.listdir(scanDir)
  for i in range(len(dirlist)):
    if dirlist[i] == 'metadata_wavemeter_ds.txt':
      wm_colNames = ['timestamp', 'offset', 'wavenumber_1', 'wavenumber_2', 'wavenumber_3', 'wavenumber_4']
      wm = pd.read_csv(scanDir + "wavemeter_ds.csv", sep=';', names=wm_colNames)
      for i in [1,2,3,4]:
        wmArray = np.array(wm['wavenumber_'+str(i)])
        wmFiltered = np.ma.compressed(np.ma.masked_less(wmArray, 1))
        if verbose: print("whichWavemeterTest1: wmFiltered:\n", wmFiltered)
        if len(wmFiltered)==0: wmRanges[i]=-1
        else: wmRanges[i] = np.max(wmFiltered)- np.min(wmFiltered)

    elif dirlist[i] == 'metadata_wavemeter_pdl_ds.txt':
        pdl_colNames = ['timestamp', 'offset', 'wavenumber_pdl']
        pdl = pd.read_csv(scanDir + "wavemeter_pdl_ds.csv", sep=';', names=pdl_colNames)
        pdlArray = np.array(pdl['wavenumber_pdl'])
        pdlFiltered = np.ma.compressed(np.ma.masked_less(pdlArray, 1))
        if verbose: print("whichWavemeterTest1: pdlFiltered:\n", pdlFiltered)
        if len(pdlFiltered)==0: wmRanges[0]=-1
        else: wmRanges[0] = np.max(pdlFiltered)- np.min(pdlFiltered)
  maxArg = np.argmax(wmRanges)
  thisWavemeter = 'pdl' if maxArg == 0 else maxArg
  return(thisWavemeter)

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
      wavemeterDic[str(s)]=str(whichWavemeter(mass,s)) # this strat usually works, but can fail; individual entries are ammended via the .txt file directly. Sue me...
    if not os.path.exists('./wavemeterToUseDictionaries/'):
      os.mkdir('./wavemeterToUseDictionaries/')
    with open('./wavemeterToUseDictionaries/RaF%dWavemeterDictionary.txt'%mass,'wb') as dicFile:
      dicFile.write(json.dumps(wavemeterDic,sort_keys=True).encode("utf-8"))
  if verbose: print("mass: %d  wavemeter Dictionary:\n"%mass, wavemeterDic)
  return(wavemeterDic)

#def rawDatPrep(m, scanInd, wavenumber, verbose=False, cleanWM=True, glitchMitigation=False):
def rawDatPrep(*args, **kwds):  
  keyLabels = ["verbose", "cleanWM", "glitchMitigation"]
  keyParms = [False,True,False] # assigning default values for keyword args
  for i in range(len(keyParms)):
    if keyLabels[i] in kwds.keys():
      keyParms[i] = kwds[str(keyLabels[i])] #...why does everything have to be complicated?
      #print("this works!"+str(keyLabels[i])+" = ", keyParms[i])
  [verbose, cleanWM, glitchMitigation] = keyParms
  #print("keyParms:", keyParms)
  if len(args)==3:
    (mass, scanInd, wavenumber) = args
  elif len(args)==2:
    (mass, scanInd) = args
    allScansBigDic = {}
    for mVal in [241,242,243,244,245,247]:
      allScansBigDic[mVal] = makeScanToWavemeterDic(mVal, redo=False, verbose=False)
    wavenumber = 'pdl' if allScansBigDic[mass][str(scanInd)] == 'pdl' else int(allScansBigDic[mass][str(scanInd)])
  else: print("yo wtf are you doing?"); quit()
  #TODO: function description
  #print("TEST. args:", args, "\nkwds?\n",kwds)
  if verbose: print("rawDatPrep called with m=%d, s=%d, wavenumber=%s"%(mass,scanInd,str(wavenumber)))
  scanIndex = scanInd
  wmNum = wavenumber
  wavenumberToUse = "wavenumber_"+str(wmNum)
  scanDir = '../RaF_RawData/'+str(mass)+'/scan_'+str(scanIndex)+'/'
  dSetTypes = ['iscool', 'tagger', 'wavemeter', 'wavemeter_pdl']
  has_iscool = False; has_tagger = False; has_wavemeter = False; has_wavemeter_pdl = False;
  dirlist=os.listdir(scanDir)

  for i in range(len(dirlist)):
    if dirlist[i]=='metadata_iscool_ds.txt':
      iscool_colNames = ['timestamp', 'offset', 'voltage']
      ic = pd.read_csv(scanDir + "iscool_ds.csv", sep=';', names=iscool_colNames)
      #if verbose: print("isCool Voltages:\n", ic.loc[:,'voltage'])
      #betaFunc = np.vectorize(computeBeta, excluded=['m'])
      #ic.loc[:,'betaVals'] = computeBeta(mass,np.array(ic.loc[:,'voltage']))#ic['voltage'].map(lambda V: computeBeta(mass, V))
      ic.loc[:,'dopplerShiftFactor'] = dopplerCorrectionFactor(mass,np.array(ic.loc[:,'voltage']))#.astype('float64'))#ic['voltage'].map(lambda V: dopplerCorrectionFactor(mass, V))
      #ic.loc[:,['voltage','betaVals','dopplerShiftFactor']] = ic[['voltage','betaVals','dopplerShiftFactor']].apply(pd.to_numeric,downcast='float')
      ic.loc[:,['voltage','dopplerShiftFactor']] = ic[['voltage','dopplerShiftFactor']].apply(pd.to_numeric,downcast='float')
      ic=ic[pd.notna(ic['timestamp'])]
      has_iscool = True #I want to call this bool ['is_cool'], but I guess I'll be informative instead :(

    elif dirlist[i] == 'metadata_tagger_ds.txt':
      #tagger_mdFile = open()
      tag_colNames = ['timestamp', 'offset', 'bunch_no', 'events_per_bunch', 'channel', 'delta_t']
      tag = pd.read_csv(scanDir + "tagger_ds.csv", sep=';', names=tag_colNames, dtype={'bunch_no':'Int32', 'events_per_bunch':'Int32', 'channel':'Int8'})
      #tag['bunch_no'].apply()
      #tag['events_per_bunch'].apply()
      tag.loc[:,['bunch_no','events_per_bunch']]=tag[['bunch_no','events_per_bunch']].apply(pd.to_numeric, downcast='unsigned')
      tag.loc[:,['channel']]=tag[['channel']].apply(pd.to_numeric, downcast='integer')
      tag=tag[pd.notna(tag['timestamp'])]#4/Aug/2019. For Mass242, scan 2512, this is for some reason necessary. Looks like garbage timestamp in tagger file
      tag.sort_values(by='timestamp',inplace=True)
      tStamps = np.array(tag.loc[:,'timestamp']); tDiffs = tStamps[1:]-tStamps[:-1]; # 10Aug2019 - computing timediffs from tagger time stamps only, since that's what I care about for rates, right?
      if verbose: print('np.mean(tDiffs) = ',np.mean(tDiffs));
      assert(np.mean(tDiffs)<1) #If assertion fails, average tDiff is larger than I'd been expecting... Maybe take a look at this.
      tag.loc[0:,'timeDiffs'] = pd.Series(np.append(np.mean(tDiffs),tDiffs), index=tag.index[0:]) #appending mean timeDiff at front, bc I don't want the first bunch/s rate to be infinite
      if verbose: print("testing tag dataframe timeDiffs:\n", tag.loc[:,['timestamp','timeDiffs','events_per_bunch']])
      has_tagger = True 

    elif dirlist[i] == 'metadata_wavemeter_ds.txt':
      wm_colNames = ['timestamp', 'offset', 'wavenumber_1', 'wavenumber_2', 'wavenumber_3', 'wavenumber_4']
      wm = pd.read_csv(scanDir + "wavemeter_ds.csv", sep=';', names=wm_colNames)
      wm=wm[pd.notna(wm['timestamp'])]
      wm[['wavenumber_1', 'wavenumber_2', 'wavenumber_3', 'wavenumber_4']] = wm[['wavenumber_1', 'wavenumber_2', 'wavenumber_3', 'wavenumber_4']].apply(pd.to_numeric,downcast='float')
      has_wavemeter = True  

    elif dirlist[i] == 'metadata_wavemeter_pdl_ds.txt':
      if wavenumber == "pdl":
        pdl_colNames = ['timestamp', 'offset', 'wavenumber_pdl']
        pdl = pd.read_csv(scanDir + "wavemeter_pdl_ds.csv", sep=';', names=pdl_colNames)
        pdl=pdl[pd.notna(pdl['timestamp'])]
        pdl.loc[:,'wavenumber_pdl'] = pdl['wavenumber_pdl'].apply(pd.to_numeric,downcast='float')
        has_wavemeter_pdl = True

  #Note: Whenever channel = -1 ~(which is almost certainly intended to indicate a glitch, right?)~, events_per_bunch is invariably 0;

  if glitchMitigation: 
    mfouter = tag.loc[:,['timestamp','timeDiffs','bunch_no','events_per_bunch','channel']]
    chans = np.array(mfouter.loc[:,'channel'])
    mfouter.loc[:,"chanSums"]=pd.Series(np.append([15,15],np.append(chans[:-4]+chans[1:-3]+chans[2:-2]+chans[3:-1]+chans[4:],[15,15])), index=mfouter.index[0:])
  else: mfouter = tag.loc[:,['timestamp','timeDiffs','bunch_no','events_per_bunch']]
  if (has_wavemeter_pdl==False and wavenumber=="pdl"): print("what the frick? this scan doesn't even have a pdl file, dummy!")
  if has_wavemeter and type(wavenumber)==int:
    mfouter = pd.merge_ordered(mfouter, wm.loc[:,['timestamp',wavenumberToUse]], on='timestamp', how='outer')#'wavenumber_1','wavenumber_2','wavenumber_3','wavenumber_4']], on='timestamp', how='outer')
  if has_iscool == True:
    #mfouter = pd.merge_ordered(mfouter, ic.loc[:,['timestamp','voltage','betaVals','dopplerShiftFactor']], on='timestamp', how='outer') #2/Aug/2019. only keeping dopplerShiftFactor to further reduce data usage
    mfouter = pd.merge_ordered(mfouter, ic.loc[:,['timestamp','dopplerShiftFactor']], on='timestamp', how='outer') #2/Aug/2019. only keeping dopplerShiftFactor to further reduce data usage
  if has_wavemeter_pdl and (wavenumber=="pdl"):
    mfouter = pd.merge_ordered(mfouter, pdl.loc[:,['timestamp','wavenumber_pdl']], on='timestamp', how='outer')

  mfouter.loc[:,wavenumberToUse].fillna(method='backfill', inplace=True)
  if glitchMitigation: mfouter.loc[:,'chanSums'].fillna(method='backfill', inplace=True)

  #mfouter.loc[:,"wavenumber_1":"wavenumber_4"].fillna(method='backfill',inplace=True) #".loc indexed to a list of columns won't support inplace operations"...
  #mfouter.loc[:,"wavenumber_1":"wavenumber_4"] = mfouter.loc[:,"wavenumber_1":"wavenumber_4"].fillna(method='backfill') #2/Aug/2019. It looks like this is causing a MemoryError sometimes?
  #Don't forget to backfill reference laser data as well, once I figure out how/when to do that... 
  if has_iscool == True:
       mfouter.loc[:,'dopplerShiftFactor'].fillna(method='backfill', inplace=True)

  mfouter.loc[:,"events_per_bunch"]=mfouter["events_per_bunch"].map(lambda a: 1 if a > 0 else a)
  mfouter.loc[:,"events_per_bunch"]=mfouter["events_per_bunch"].astype('Int8',downcast='unsigned')
  #remove "NaN" entries from events_per_bunch now? so that timeDiffs aren't computed including these non-counting event counts.
  if verbose: print("TEST5:\n", mfouter.loc[:,["timestamp",'timeDiffs','events_per_bunch',wavenumberToUse]])
  if verbose: print(mfouter.info())
  mfouter = mfouter[pd.notna(mfouter['bunch_no'])]#2/Aug/2019. It looks like this is causing a MemoryError sometimes? #1/Feb/2020 probably not anymore, now that I upgraded to 64-bit Python smh...

  if verbose: print("TEST6:\n", mfouter.loc[:49,["timestamp",'timeDiffs',"events_per_bunch",wavenumberToUse]])
  '''tStamps = np.array(mfouter.loc[:,'timestamp']); tDiffs = tStamps[1:]-tStamps[:-1]; #10Aug2019-computing timediffs from tagger time stamps only, since that's what I care about for rates, right?
  if verbose: print('np.mean(tDiffs) = ',np.mean(tDiffs));
  assert(np.mean(tDiffs)<1) #If assertion fails, average tDiff is larger than I'd been expecting... Maybe take a look at this.
  mfouter.loc[0:,'timeDiffs'] = pd.Series(np.append(np.mean(tDiffs),tDiffs), index=mfouter.index[0:]) #appending mean timeDiff at front, bc I don't want the first bunch/s rate to be infinite'''
  
  if cleanWM==True: #August4/2019, this is my new wavemeter cleaning implementation
    mfouter[wavenumberToUse]=mfouter[wavenumberToUse].map(lambda v: v if v > 0 else float('NaN'))
    mfouter = mfouter[pd.notna(mfouter[wavenumberToUse])]

  if has_iscool == True:
    mfouter.loc[:,'wavenumber'] = mfouter.loc[:,wavenumberToUse]/mfouter.loc[:, 'dopplerShiftFactor']
  else:
    print("YO... No iscool data. How am I supposed to correct these wavenumber measurements?!?")
    nextScan=scanInd
    conditionMet = False
    while conditionMet == False:
      nextScan+=1
      try:
        nextDirList = os.listdir('../RaF_RawData/'+str(mass)+'/scan_'+str(nextScan)+"/")
        if 'metadata_iscool_ds.txt' in nextDirList:
          isCoolVoltage = np.loadtxt('../RaF_RawData/'+str(mass)+'/scan_'+str(nextScan)+'/iscool_ds.csv',delimiter=';',max_rows=1)[-1]
          print("Using Scan %d initial isCool reading; Voltage=%d"%(nextScan, isCoolVoltage))
          dcf = dopplerCorrectionFactor(mass, isCoolVoltage)
          print("dcf=%.4f"%dcf)
          mfouter.loc[:,'wavenumber'] = mfouter.loc[:,wavenumberToUse]/dcf
          break
      except OSError:
        conditionMet = False
  if verbose:print("test something:\n",mfouter.loc[:,['timestamp','timeDiffs','events_per_bunch', 'wavenumber']])
  if glitchMitigation:
    if verbose and scanInd==2178: print("TEST7:\n", mfouter.loc[mfouter.index[88800:88900],["timestamp",'timeDiffs',"events_per_bunch",'channel' if glitchMitigation else wavenumberToUse]])
    mfouter.loc[:,'channel'].fillna(method='backfill', inplace=True) #3/Aug/2019. 3:20 PM want to back-fill values _before_ I throw out all the NaNs!
    if verbose and scanInd==2178: print("TEST8:\n", mfouter.loc[mfouter.index[88800:88900],["timestamp",'timeDiffs',"events_per_bunch",'channel' if glitchMitigation else wavenumberToUse]])
    mfouter["chanSums"]=mfouter["chanSums"].map(lambda a: a if a > -5 else float('NaN')) #3/Aug/2019. 1:50AM pls work!
    mfouter["chanSums"]=mfouter["chanSums"].astype('Int8',downcast='integer')
    if verbose and scanInd==2178: print("TEST9:\n", mfouter.loc[mfouter.index[88800:88900],['timeDiffs',"events_per_bunch",'chanSums', wavenumberToUse]])
    mfouter=mfouter[pd.notna(mfouter['chanSums'])]
    if verbose: print("TEST10:\n", mfouter.loc[:,['timeDiffs',"events_per_bunch",'chanSums', wavenumberToUse]])

  preppedDataFrame = mfouter.loc[:,["timestamp", 'timeDiffs', 'wavenumber', 'events_per_bunch']].copy()
  del(mfouter)
  return(preppedDataFrame)#TODO add in other wavenumber correction thing

def mergeDatRaw(mass, scanList,verbose=False):
  #creates dataframe of same format as rawDatPrep(), but combining multiple scans
  dfList=[]
  for scan in scanList:
    dfList.append(rawDatPrep(mass,scan,verbose=verbose))
  return(pd.concat(dfList))

def mergeDatPrepped(dir, mass, scanList,verbose=False):
  #creates dataframe of same format as rawDatPrep(), but combining multiple scans
  dfList=[]
  for scan in scanList:
    dfList.append(pd.read_csv(dir+'mass%d_scan%dDataframe.csv'%(mass,scan),index_col=0))
  return(pd.concat(dfList))

def trimRange(outputDF, ltrim=-1, rtrim=-1):
  if ltrim>0:
    outputDF.loc[:,'wavenumber_mean']=outputDF['wavenumber_mean'].map(lambda v: v if v > ltrim else float('NaN'))
    outputDF = outputDF[pd.notna(outputDF['wavenumber_mean'])]
  if rtrim>0:
    outputDF.loc[:,'wavenumber_mean']=outputDF['wavenumber_mean'].map(lambda v: v if v < rtrim else float('NaN'))
    outputDF = outputDF[pd.notna(outputDF['wavenumber_mean'])]
  return(outputDF)

def normalizer(outputDF, normalizedOn=False):
  if normalizedOn == "Integral":
    sigTot = np.sum(outputDF.loc[:,'signal_value'])
    print("test: sigTot=", sigTot)
    outputDF['signal_value']=outputDF['signal_value']/sigTot
    outputDF['signal_uncertainty']=outputDF['signal_uncertainty']/sigTot
  elif normalizedOn=="MaxValue":
    minVal=np.min(outputDF.loc[:,'signal_value'])
    outputDF.loc[:,'signal_value']=outputDF['signal_value']-minVal
    maxVal = np.max(outputDF.loc[:,'signal_value']); 
    print("test: maxVal=", maxVal)
    outputDF.loc[:,'signal_value']=outputDF['signal_value']/maxVal
    outputDF.loc[:,'signal_uncertainty']=outputDF['signal_uncertainty']/maxVal
  return(outputDF)

def randSubsets(df, frac=0.1, nSamps=100):
  # generates nSamps random subsamples (with replacement between samples, but not within individual sample selection) from rows of dataframe df, each of size = ceil(size(df)*frac).
  #subsetFrame = pd.DataFrame(index=range(nSamps),columns=["sampleDFrame"])
  subsetArray=[]
  #print("uhh:\n", subsetFrame,subsetFrame.iloc[0])
  for i in range(nSamps):
    #subsetFrame.iloc[i] = df.sample(frac=frac)
    subsetArray.append(df.sample(frac=frac))
  return(subsetArray)

def makeUseable(df, nBins=100, resolution=-1, noNaNsense=True, cropSparseEnds=True, normalizedOn=False,ltrim=-1,rtrim=-1, verbose=False):
  #converts (usually huge) time-centric dataframes from rawDatPrep() into spectrum-friendly wavenumber-based dataframes
  kVals = np.array(df.loc[:,"wavenumber"]); kRange=max(kVals)-min(kVals)
  if kRange>0:
    if verbose: print("testing wavenumber range: min=%.3f; max=%.3f"%(min(kVals),max(kVals)))
    if resolution<0: binQuant = nBins
    else: binQuant = math.ceil(kRange/resolution)
  else:
    print("what tf tf... kmin = ",min(kVals),"  kMax = ",max(kVals)," kRange = ",kRange)
  if verbose: print("TESTSTSSTSTS: numBins=%d"%binQuant)
  df.loc[:,'waveProds'] = df.loc[:,'wavenumber']*df.loc[:,'timeDiffs']
  kBins = pd.cut(df.loc[:,"wavenumber"], bins=binQuant)#, retbins=True)

  #print("makeUsable test1.\n", df.groupby(kBins).head() )
  #print("test8.\n", kBins )

  aggDat = df.groupby(kBins).agg({'wavenumber':['mean', 'min', 'max'], 'events_per_bunch':['sum'], 'timeDiffs':['sum'], 'waveProds':['sum']}).reset_index() #Wtf apparently reset_index() is p important... 
  #print("aggDat:\n", aggDat[30:100])
  #print("Specifically...\n", np.array(aggDat.loc[:,('events_per_bunch','sum')]))
  #print("Furthermore...\n", np.sqrt(np.array(aggDat.loc[:,('events_per_bunch','sum')]))) #lmao, wt actual f is going on with numpy here? You were my rock, np! :( 
  #print("Furthermore...\n", np.array(list(map(math.sqrt, aggDat.loc[:,('events_per_bunch','sum')]))))

  outputDF = pd.DataFrame({"wavenumber_mean"   : aggDat.loc[:,('waveProds','sum')]/aggDat.loc[:,('timeDiffs','sum')], #aggDat.loc[:,('wavenumber','mean')], #small change, but reported wavenumber is now weighted by measurement time.
                        "signal_value"         : aggDat.loc[:,('events_per_bunch','sum')]/aggDat.loc[:,('timeDiffs','sum')],
                        "signal_uncertainty"   : np.maximum(2, np.array(list(map(math.sqrt, aggDat.loc[:,('events_per_bunch','sum')]))))/aggDat.loc[:,('timeDiffs','sum')], #resorting to this mess bc numpy is throwing the weirdest fkn error...
                        "measurement_duration" : aggDat.loc[:,('timeDiffs','sum')]},index=range(binQuant) )

  outputDF.sort_values('wavenumber_mean',inplace=True)
  if noNaNsense:
    outputDF = outputDF.dropna(how='any',axis=0)
  if (ltrim > -1 or rtrim > -1):
    outputDF=trimRange(outputDF, ltrim=ltrim, rtrim=rtrim)

  if cropSparseEnds and len(outputDF['wavenumber_mean'])>3:
    dRay = np.array(outputDF.dropna(how='any',axis=0, inplace=False).loc[:,'wavenumber_mean'])
    meanSpacing = np.mean(dRay[1:]-dRay[:-1])
    if verbose: print("crop check: meanSpacing=%f"%meanSpacing)
    i = 0 #cropping left
    while (dRay[i+1]-dRay[i]>3*meanSpacing) or (dRay[i+2]-dRay[i+1]>3*meanSpacing) or (dRay[i+3]-dRay[i+2]>3*meanSpacing):
    # If the any of the next 3 v-spacings are greater than 3 times the average spacing, increment the index at which to start cropping.
      i+=1
    e = len(dRay) #cropping right
    while (dRay[e-1]-dRay[e-2]>3*meanSpacing) or (dRay[e-2]-dRay[e-3]>3*meanSpacing) or (dRay[e-3]-dRay[e-4]>3*meanSpacing):
    # If the any of the preceding 3 v-spacings are greater than 3 times the average spacing, increment the index at which to start cropping(?)
      e-=1
    outputDF = outputDF.iloc[i:e,:]

  if (normalizedOn == "Integral" or normalizedOn=="MaxValue"):
    outputDF=normalizer(outputDF, normalizedOn=normalizedOn)
  outputDF.reset_index(drop=True, inplace=True)
  return(outputDF)

def plotData(output, title='',fig=-1, resolution=-1):
  if fig==-1: plt.figure("output Plot")
  else: plt.figure(fig)
  if title=='': plt.title("Output Plot. numBins = %d"%len(output.loc[:,'wavenumber_mean']))
  else: plt.title(title)
  plt.errorbar(x=output.loc[:,'wavenumber_mean'], y=output.loc[:,'signal_value'], yerr=output.loc[:,'signal_uncertainty'], fmt="o",ecolor='k', alpha=.75)#, xerr = kBins)
  plt.xlabel(r'wavenumber ($cm^{-1}$)')
  plt.ylabel('Rate (counts/s)') #TODO: determine unit on timestamp

def fileWriter(output, m=-1, scanInd='not given', target='NaN'):
  #Write dataframe to file. This function is pretty useless though idk why I made it... Just use pd.to_csv()
  if target=="NaN":
    if not os.path.exists('./FrequencyConvertedDatasets/%d'%m): os.mkdir('./FrequencyConvertedDatasets/%d'%m)
    output.to_csv(path_or_buf='./FrequencyConvertedDatasets/%d/scan_%s.csv'%(m, str(scanInd)), sep=',', float_format='%.11f', columns=['signal_uncertainty','wavenumber_mean','signal_value'], index=True, header=['error','freq','rate'])
  else: output.to_csv(path_or_buf=target, sep=',', float_format='%.11f', columns=['signal_uncertainty','wavenumber_mean','signal_value'], index=True, header=['error','freq','rate'])

def doEverything(m, scanInd, wavenumber, nBins=100, resolution=-1, writeToFile=False, makePlot=False, cleanWM=True, verbose=False, cropSparseEnds=True, noNaNsense=True):
  mfba =  rawDatPrep(m, scanInd, wavenumber, verbose=verbose, cleanWM=cleanWM)
  if resolution==-1: output = makeUseable(mfba, nBins=nBins)
  else: output = makeUseable(mfba, resolution=resolution,cropSparseEnds=cropSparseEnds, noNaNsense=noNaNsense)
  if writeToFile: fileWriter(output, m, scanInd)
  if makePlot: plotData(output, m, scanInd, wavenumber, nBins=nBins, resolution=resolution)
  return(output)

if __name__ == '__main__':
  mass = 244
  scanIndex = [2304,2305,2306]
  #wmNum = whichWavemeter(mass,scanIndex)
  #wmNum = 'pdl' if wmNum=='pdl' else int(wmNum)
  res=.1
  
  mfba =  mergeDatRaw(mass, scanIndex)#, wmNum, cleanWM=True, verbose=False)
  #print("test 9:\n", mfba.head)
  #print("test 10:\n", mfba.tail)
  output = makeUseable(mfba, resolution=res)
  #print("test11:\n", output)
  plotData(output,title=r'$Ra^{%d}F^{19}$,   $A^2\Pi_{1/2} \leftarrow X^2\Sigma^{+}$, $\Delta v=0$'%(mass-19)+'\nScan: %s, Resolution=%.2f $cm^{-1}$'%(str(scanIndex), res), fig=1)
  #output2=makeUseable(mergeDatRaw(mass,[2367,2368]),resolution=res)
  #plotData(output2,title=r'$Ra^{%d}F^{19}$,   $B^2\Delta_{1/2} \leftarrow X^2\Sigma^{+}$, $\Delta v=0$'%(mass-19)+'\nScan: %s, Resolution=%.2f $cm^{-1}$'%(str([2367, 2368]), res), fig=2)
  
  subSamps=randSubsets(mfba, 0.25, 20)
  print("Test:\n", subSamps[0])
  subOut = makeUseable(subSamps[0], resolution=res)
  plotData(subOut,title=r'$Ra^{%d}F^{19}$,   $A^2\Pi_{1/2} \leftarrow X^2\Sigma^{+}$, $\Delta v=0$'%(mass-19)+'\nScan: %s, Resolution=%.2f $cm^{-1}$\nrandom subsample'%(str(scanIndex), res), fig=2)
  #doEverything(234, 2127,)

  '''m=242;scans=[2312, 2313]
  mfba=mergeDatRaw(m,scans, verbose=True)
  print("meep")
  output = makeUseable(mfba, resolution=.07)
  plotData(output, m, 2312, 2, resolution=.07)
'''
  plt.show()
