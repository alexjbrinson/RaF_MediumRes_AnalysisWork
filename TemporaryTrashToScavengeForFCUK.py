'''
#I can worry about this Stuff later...
def Scanalyzer(x, minR=1, maxR=20, maxN=10, method="emcee", fitPlots=True, binSpreadPlot=True, sameSkew=False, useWeights=True):
  rbin = maxR #initial bin settings
  print("Running Scanalyzer. x =",x)
  (cfrx,ccex) = fitScanX(x, rbin, method=method, minR=minR, maxN=maxN, useWeights=useWeights, sameSkew= (sameSkew))# and ((scanCount>9)) ) #After scan 10-1, mirror removed, so we can add assumption about peak shapes similarities(?)
  #print("fitScanX: compiled fits reduced chi^2 vals:\n", cfrx, "\nccex:", ccex)
  """print("test1:\n",np.mean(ccex[:,:,0], axis=0))
  print("test2:\n", np.std(ccex[:,:,0], axis=0))
  print("test3:\n", np.max(ccex[:,:,0], axis=0)-np.min(ccex[minR-1:,:,0], axis=0))"""
  finalScanEstimates = np.c_[np.mean(ccex[:,:,0], axis=0), np.std(ccex[:,:,0], axis=0), np.max(ccex[:,:,0], axis=0)-np.min(ccex[:,:,0], axis=0)]
  #print("finalScanEstimates.shape=",finalScanEstimates.shape,"\nfinalScanEstimates:\n",finalScanEstimates)
  #finalScanEsts = np.copy(finalScanEstimates)
  for p in range(len(ccex[0,:,0])):
    weightStats = weightedStatistics(ccex[:,p,0], ccex[:,p,1])
    #print("test: weightStats = ", weightStats)
    finalScanEstimates[p,0] = weightStats[0]; finalScanEstimates[p,1] = weightStats[1]
  print("Also test: finalScanEstimates.shape=",finalScanEstimates.shape," finalScanEstimates:\n",finalScanEstimates)

  if not os.path.exists('FitResults/Scan%dFits/OutputFiles'%x):
    os.mkdir('FitResults/Scan%dFits/OutputFiles'%x)
  np.savetxt("FitResults/Scan"+str(x)+"Fits/OutputFiles/CompiledFitRedChis.csv", cfrx, delimiter=",")
  np.savetxt("FitResults/Scan"+str(x)+"Fits/OutputFiles/ScanalyzerOutput.csv", cfrx, delimiter=",")
  xFile=open("FitResults/Scan"+str(x)+"Fits/OutputFiles/CompiledFitEstimates.txt",'w+')
  xFile.write("#Compiled Fit Center Estimates:\n"+str(ccex))
  xFile.write("\n#Scanalyzer Final Estimates:\n"+str(finalScanEstimates[:,0])+"\n#1Sigma:\n"+str(finalScanEstimates[:,1])+"\n#Range:\n"+str(finalScanEstimates[:,2]))
  xFile.close()
  if binSpreadPlot==True:
    MultiBinSpreadPlotter(x, ccex, minR=minR, maxR=maxR)
  return(finalScanEstimates)

def NearTheory(targetDict, peakEntry, predicArray, vibrationalLabelList, electricLabel="", plotResults=True, predicUncerts=0):
  meanK=peakEntry[0]; sigmaK=peakEntry[1]; rangeK=peakEntry[2]
  if np.any(abs(meanK-predicArray)<=math.sqrt(rangeK**2+predicUncerts**2)):
    idLevelIndex = np.argmin(abs(meanK-predicArray)); vibeLabel = vibrationalLabelList[idLevelIndex]
    targetDict[vibeLabel].append(peakEntry)
    return(True)
  elif np.any(abs(meanK-predicArray)<=math.sqrt((5*sigmaK)**2+predicUncerts**2)):
    idLevelIndex = np.argmin(abs(meanK-predicArray)); vibeLabel = vibrationalLabelList[idLevelIndex]
    targetDict[vibeLabel].append(peakEntry)
    print("rangeK did not suffice, but the fitted peak was within 5 sigma of a theory peak.\nPeak Entry:", peakEntry, "\nPredic Entry:",electricLabel,vibeLabel)
    return(True)
  return(False)'''

  '''startTime = time.clock()
  """Loading in data from scans"""
  dirlist=os.listdir('scans245')
  scanInds = []
  print("test1. os.listdir('scans245'):\n",dirlist)
  for i in range(len(dirlist)):
    if (dirlist[i].endswith('.csv') and dirlist[i].startswith('245RaF_LR_')):
      scanInds.append( int(dirlist[i].replace('.csv',"").replace('245RaF_LR_',"")) )
  print("test2. scanInds:\n",scanInds)

  datDic = {}

  for i in range(len(scanInds)):
    datArray = np.loadtxt('scans245/245RaF_LR_'+str(scanInds[i])+'.csv', dtype=float, skiprows=1, delimiter=',')
    if datArray.ndim != 2:
      print("Junk dataset from scan "+str(scanInds[i])+". Will throw out.")
    elif len(datArray[:,2])<20:
      print("Few datapoints in scan "+str(scanInds[i])+". Will throw out.")
    else:
      if np.any(datArray[:,2]<0):
        mask = datArray[:,2]>0
        datArray = np.array(datArray[[mask==True]])
        print("Scan "+str(scanInds[i])+" contained negative wavenumbers...", str(len(mask)-len(datArray[:,2])) + " data point(s) have been removed. Updated array shape =", datArray.shape)
      datDic[scanInds[i]] = cleanDataSet(datArray)
      if scanInds[i]==2137:
        print("All of the signal in Scan 2137 occurs in the first 6th of the dataset. Will crop the rest so it doesn't dominate the fits.")
        datArray=datDic[2137]
        rCutoff = np.argmin(np.abs(datArray[:,2]-13325))
        datDic[2137] = datArray[:rCutoff]
      elif scanInds[i]==2138:
        print("All of the signal in Scan 2138 occurs in the last 6th of the dataset. Will crop the rest so it doesn't dominate the fits.")
        datArray=datDic[2138]
        lCutoff = np.argmin(np.abs(datArray[:,2]-13225))
        datDic[2138] = datArray[lCutoff:]
      elif scanInds[i]==2178:
        print("There's some fishy business going on int the first quarter of Scan 2178. Will crop so it doesn't screw up my fits.")
        datArray=datDic[2178]
        lCutoff = np.argmin(np.abs(datArray[:,2]-13256))
        datDic[2178] = datArray[lCutoff:]
      elif scanInds[i]==2368:
        print("Scan 2368 has garbage at the very end. Will crop so it doesn't screw up my fits.")
        datArray=datDic[2368]
        datDic[2368] = datArray[:-2]

  print("datDic.keys()", list(datDic.keys()))

  beta = 0.0005920684 #v_bunch/c
  gamma = 1.00000017527255 #1/sqrt(1-beta^2)
  dyeInds = [2130, 2131,2132,2135,2136,2137,2138,2139,2323,2324,2325,2346,2360,2364,2365,2368,2375,2376]
  tiSapInds = [2164,2165,2178,2309,2310,2317,2319,2320,2340,2341,2349,2350]

  vlabelArray = np.array(["6->5","5->4","4->3","3->2","2->1","1->0",
            "5->5","4->4","3->3","2->2","1->1","0->0",
            "5->6","4->5","3->4","2->3","1->2","0->1"])

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
                    -1,-1,-1,-1,-1,-1,]

  vlineArrayPI12Reflex = vlineArrayPI12 - 2*beta*gamma*np.array(vlineArrayPI12)

  predictedKs = np.union1d(np.union1d(vlineArrayPI12, vlineArrayPI12), np.union1d(vlineArrayDELTA32, vlineArrayDELTA52))

  """(Temporarily?) Global Variables:"""
  peakEsts = {}; peakUncerts = {};
  ImportPeakEstimateDics(["peakEsts", "peakUncerts"], "IdentifyingPeaksInScansByEye_Take2.txt")
  print("test: len(list(peakEsts.keys())) = ", len(list(peakEsts.keys())))
  rebinMinDic = {};rebinMaxDic = {}
  ImportPeakEstimateDics(["rebinMinDic","rebinMaxDic"], "rebinRangeSettingsDictionaries.txt")
  #print("test: rebinMaxDic = ", rebinMaxDic)

  PI12Dic={}; DELTA32Dic={}; DELTA52Dic={}; PI32Dic={}; PI12ReflexDic = {}
  initializeIDedPeakDictionary(PI12Dic, vlabelArray); initializeIDedPeakDictionary(DELTA32Dic, vlabelArray); initializeIDedPeakDictionary(DELTA52Dic, vlabelArray)
  initializeIDedPeakDictionary(PI32Dic, vlabelArray); initializeIDedPeakDictionary(PI12ReflexDic, vlabelArray)
  idDicList = [PI12Dic,DELTA32Dic,DELTA52Dic,PI32Dic, PI12ReflexDic] #List of dictionaries where I'll store Scanalyzer results consistent with theory
  predictionsList = [vlineArrayPI12,vlineArrayDELTA32,vlineArrayDELTA52,vlineArrayPI32, vlineArrayPI12Reflex]
  predictionsListUncertainties = [.5, 1, .5, 5, .5]
  electronicLabelList = ["PI_1/2","DELTA_3/2","DELTA_5/2","PI_32", "PI_1/2(Ref)"]
  theoryList=[]; mysteryList=[]

  showFinalPlot = True
  dickeys = np.sort(np.array(list(datDic.keys())))
  scanCountRange = range(len(dickeys))
  rebin=20 #maximum rebin setting, unless specified to be lower by rebinMaxDic
  minnerBinner = 1 #minimum rebin setting, unless specified to be higher by rebinMinDic
  for scanCount in scanCountRange:
    x=dickeys[scanCount]
    print("running the main event! x =%d"%x)
    print("test: rebinMaxDic[%d]=%d"%(x,rebinMaxDic[x]))
    minRebin = max(minnerBinner,rebinMinDic[x])
    maxRebin = min(rebin, rebinMaxDic[x])
    finScanEstsX = Scanalyzer(x, minR=minRebin, maxR=maxRebin, maxN=6, method="emcee", sameSkew=True, useWeights=True)#"leastsq")
    if maxRebin==minnerBinner: #if only one rebin setting exists, we have to add uncertainty by hand, since there's no data spread to draw from.
      finScanEstsX = np.c_[finScanEstsX[:,0], .5*np.ones_like(finScanEstsX[:,0]), np.ones_like(finScanEstsX[:,0])]
    print("Scanalyzer finished scanning for x=%d.\nFinal Scan Estimates:"%x,finScanEstsX[:,0],"\nNow comparing to theory predictions:")
    for j in range(len(finScanEstsX[:,0])):
      pkEntry = finScanEstsX[j]
      homeFound=False
      print("x=%d; j=%d"%(x,j))
      if scanCount<=10: #if we're trying to ID peaks in an early scan, we should first check if it's a "reflected" peak.
        if NearTheory(PI12ReflexDic, pkEntry, vlineArrayPI12Reflex, vlabelArray, electricLabel="PI12_REFLEX", predicUncerts=.5):
          homeFound=True
      if homeFound==False: #Only bother comparing to other predictions if we've already established that it wasn't a "reflected" peak.
        for i in range(len(idDicList)):
          #print("x=%d; j=%d; i=%d"%(x,j,i))
          if NearTheory(idDicList[i], pkEntry, predictionsList[i], vlabelArray, electricLabel=electronicLabelList[i], predicUncerts=predictionsListUncertainties[i]):
            print("Ayyy, peak identified! peak value = %.2f; electricLabel = "%pkEntry[0], electronicLabelList[i])
            homeFound=True
            break
      if homeFound==True:
        theoryList.append([scanCount, pkEntry[0], pkEntry[2]])
      else:
        print("This little peaky had no home :(", pkEntry)
        mysteryList.append([scanCount, pkEntry[0], pkEntry[2]])
  averagesFile1 = open("FitResults/AveragesFile_FromSigmas.txt","w+")
  averagesFile2 = open("FitResults/AveragesFile_FromSpreads.txt","w+")
  finalOutputFile1 = open("FitResults/FinalOutputFile_FromSigmas.txt","w+")
  finalOutputFile2 = open("FitResults/FinalOutputFile_FromSpreads.txt","w+")
  anyTransitionsIdentified = False
  observedTransitionsDic = {}
  finalOutputFile1.write("Transition\n"+10*"\t"+"v''->v'\t\tExp.(cm^-1)\n"+"-"*50+"\n")
  finalOutputFile2.write("Transition\n"+10*"\t"+"v''->v'\t\tExp.(cm^-1)\n"+"-"*50+"\n")

  for i in range(len(idDicList)):
        print(electronicLabelList[i]+" results:\n", idDicList[i])
        finalOutputFile1.write("-"*50+"\nSIGMA->"+electronicLabelList[i]+"\n")
        finalOutputFile2.write("-"*50+"\nSIGMA->"+electronicLabelList[i]+"\n")
        for transition in list(idDicList[i].keys()):
          if not(idDicList[i][transition]==[]):
            anyTransitionsIdentified = True
            transitionData = np.array(idDicList[i][transition])
            print("JulyTest1: i=%d, transition = ",transition)
            print("test: transitionData.shape = ", transitionData.shape)
            print("transitionData:\n", transitionData)
            #wStats = weightedStatistics(transitionData[:,0], transitionData[:,1], transitionLabel=electronicLabelList[i]+"_"+transition)
            #^ Previous line modified 19/July/2019 to replace traditional scat/stat uncerts with ranges of peak ests over diff rebin settings
            wStats1 = weightedStatistics(transitionData[:,0], transitionData[:,1], transitionLabel=electronicLabelList[i]+"_"+transition)
            observedTransitionsDic[electronicLabelList[i]+"_"+transition]  = wStats1
            averagesFile1.write("Transition: "+electronicLabelList[i]+" "+transition+"  Average Value: %.2f +/- %.5f"%wStats1)
            averagesFile1.write("\nConsistent Observations:\n")
            averagesFile1.write(np.array2string(np.array(idDicList[i][transition]), formatter={'float_kind':lambda k: "%.4f" % k}))
            averagesFile1.write("\n\n")
            finalOutputFile1.write("\t"*10+transition+"\t\t\t%.2f +/- %.5f\n"%wStats1)

            wStats2 = weightedStatistics(transitionData[:,0], transitionData[:,2], transitionLabel=electronicLabelList[i]+"_"+transition)
            observedTransitionsDic[electronicLabelList[i]+"_"+transition]  = wStats2
            averagesFile2.write("Transition: "+electronicLabelList[i]+" "+transition+"  Average Value: %.2f +/- %.5f"%wStats2)
            averagesFile2.write("\nConsistent Observations:\n")
            averagesFile2.write(np.array2string(np.array(idDicList[i][transition]), formatter={'float_kind':lambda k: "%.4f" % k}))
            averagesFile2.write("\n\n")
            finalOutputFile2.write("\t"*10+transition+"\t\t\t%.2f +/- %.5f\n"%wStats2)

  theoryList=np.array(theoryList); mysteryList=np.array(mysteryList)
  print("tests. theoryList:\n",theoryList,"\nmysteryList:\n",mysteryList)
  print("testy: observedTransitionsDic:\n", observedTransitionsDic)
  averagesFile1.write("\nObservations that weren't assigned to known transitions:\n")
  averagesFile1.write(np.array2string(mysteryList, formatter={'float_kind':lambda k: "%.2f" % k}))
  averagesFile1.close()
  finalOutputFile1.close()
  averagesFile2.write("\nObservations that weren't assigned to known transitions:\n")
  averagesFile2.write(np.array2string(mysteryList, formatter={'float_kind':lambda k: "%.2f" % k}))
  averagesFile2.close()
  finalOutputFile2.close()

  fig = plt.figure(3)
  fig.set_size_inches(20, 12)
  if theoryList.shape[0]>0:
    plt.errorbar(theoryList[:,1], theoryList[:,0], xerr=theoryList[:,2], fmt='o', color="green", label="Peaks that agree with prediction", markersize=6)
  if mysteryList.shape[0]>0:
    plt.errorbar(mysteryList[:,1], mysteryList[:,0], xerr=mysteryList[:,2], fmt='o', color="red", label="Peaks with no home :(", markersize=6)
  vLinePlotter(vlineArrayPI12, r'$^2\Sigma^{+}_{1/2} \rightarrow ^2\Pi_{1/2}$', reflections=True)
  vLinePlotter(vlineArrayDELTA32, r'$^2\Delta_{3/2}$')
  vLinePlotter(vlineArrayDELTA52, r'$^2\Delta_{5/2}$')
  if anyTransitionsIdentified:
    for transition in observedTransitionsDic:
      kPos=observedTransitionsDic[transition][0]; kSig=observedTransitionsDic[transition][1]
      print("kPos =",kPos,"; kSig =",kSig)
      plt.gca().add_patch(Rectangle((kPos-1*kSig, scanCountRange[0]-1), 2*kSig, scanCountRange[-1]-scanCountRange[0]+2, color="blue", alpha=.25))
      plt.axvline(x=kPos, ymin=0, ymax=1, color="blue", linestyle='-', linewidth=.5)
      plt.annotate("observedTransition\n"+transition, xy=(kPos, -.1), fontsize=14, ha='center', xycoords=('data','figure fraction'),color="blue")

  plt.xlim(plt.gca().get_xlim())
  box = plt.gca().get_position()
  plt.gca().set_position([box.x0, box.y0, box.width * 0.9, box.height])
  plt.legend(loc="center right", bbox_to_anchor=(1.25,.5), fontsize=10)
  plt.title("Compiling all Scanalyzer Results", fontsize=18)
  plt.xlabel(r'Wavenumber (cm$^{-1}$)', fontsize=18)
  plt.ylabel("Scan Index", fontsize=18)
  plt.yticks(ticks=scanCountRange, labels=dickeys[scanCountRange] )
  if not os.path.exists('FitResults'):
    os.mkdir('FitResults')
  plt.savefig("FitResults/CompiledScanalyzerResultsPlot.png")

  try:
    endTime = time.clock()
    print("Holy fudge... This actually took %f s to run :("%(endTime-startTime))
  except (ValueError, TypeError):
    print("Holy fudge... This actually took so long to run that %(endTime-startTime) doesn't even register as a number anymore...")

  if showFinalPlot:
    plt.show()'''