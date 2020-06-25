import math
import numpy as np
import pandas as pd
import os.path
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from scipy import special
import lmfit
from lmfit import Model, Parameter
from lmfit.models import SkewedVoigtModel, LinearModel, ConstantModel, GaussianModel, LorentzianModel
import emcee
#import csv
import time
import numdifftools
import LoadingAndMungingData as lmd

xDat = np.linspace(0,100,num=101)
yDat = np.sin((math.pi/10)*xDat)+0.2*np.random.randn(len(xDat))



smoothWidth=3
test=1/(np.abs(np.linspace(-smoothWidth, smoothWidth,num=2*smoothWidth+1))+1)
norm=1/np.sum(test)
print("test:",test,"norm = ",norm)
smoothedX = np.zeros(len(xDat)-2*smoothWidth)
smoothedY = np.zeros(len(xDat)-2*smoothWidth)
for i in range(-smoothWidth, smoothWidth+1):
  #print("i=%d, um:\n"%i,norm*(1/(np.abs(i)+1))*xDat[smoothWidth+i:-(smoothWidth-i)])
  if i==smoothWidth:
    smoothedX+=norm*(1/(np.abs(i)+1))*xDat[smoothWidth+i:]
    smoothedY+=norm*(1/(np.abs(i)+1))*yDat[smoothWidth+i:]
  else: 
    smoothedX+=norm*(1/(np.abs(i)+1))*xDat[smoothWidth+i:-(smoothWidth-i)]
    smoothedY+=norm*(1/(np.abs(i)+1))*yDat[smoothWidth+i:-(smoothWidth-i)]

plt.figure(1)
plt.plot(xDat,yDat,'bo')
plt.plot(smoothedX,smoothedY,"r-")

dframe=pd.DataFrame(np.transpose(np.array([xDat, yDat])),columns=["xDat","yDat"])
smoothFrame = dframe.rolling(smoothWidth,win_type='gaussian').sum(std=1)

print('dframe:\n',dframe.head(),'\nsmoothFrame:\n', smoothFrame)
plt.plot(smoothFrame.loc[:,'xDat'],smoothFrame.loc[:,'yDat'], 'g-')

plt.show()

def smoother(dframe, smoothWidth):#, meth='def', σ = 1):
  xDat=np.array(dframe.loc[:,'wavenumber_mean']); yDat=np.array(dframe.loc[:,'signal_value'])
  σyDat = np.array(dframe.loc[:,'signal_uncertainty']); yVarDat=np.square(σyDat)
  durDat=np.array(dframe.loc[:,'measurement_duration'])
  test=1/(np.abs(np.linspace(-smoothWidth, smoothWidth,num=2*smoothWidth+1))+1)
  norm=1/np.sum(test)
  print("test:",test,"norm = ",norm)
  smoothedX = np.zeros(len(xDat)-2*smoothWidth); smoothedY = np.zeros(len(xDat)-2*smoothWidth)
  smoothedVar = np.zeros(len(xDat)-2*smoothWidth); smoothedDur = np.zeros(len(xDat)-2*smoothWidth)
  for i in range(-smoothWidth, smoothWidth+1):
    #print("i=%d, um:\n"%i,norm*(1/(np.abs(i)+1))*xDat[smoothWidth+i:-(smoothWidth-i)])
    if i==smoothWidth:
      smoothedX+=norm*(1/(np.abs(i)+1))*xDat[smoothWidth+i:]
      smoothedY+=norm*(1/(np.abs(i)+1))*yDat[smoothWidth+i:]
      smoothedVar+=norm*(1/(np.abs(i)+1))*yVarDat[smoothWidth+i:]
      smoothedDur+=norm*(1/(np.abs(i)+1))*durDat[smoothWidth+i:]
    else: 
      smoothedX+=norm*(1/(np.abs(i)+1))*xDat[smoothWidth+i:-(smoothWidth-i)]
      smoothedY+=norm*(1/(np.abs(i)+1))*yDat[smoothWidth+i:-(smoothWidth-i)]
      smoothedVar+=norm*(1/(np.abs(i)+1))*yVarDat[smoothWidth+i:-(smoothWidth-i)]
      smoothedDur+=norm*(1/(np.abs(i)+1))*durDat[smoothWidth+i:-(smoothWidth-i)]
  smoothDF = pd.DataFrame({"wavenumber_mean"   : smoothedX, #aggDat.loc[:,('wavenumber','mean')], #small change, but reported wavenumber is now weighted by measurement time.
                        "signal_value"         : smoothedY,
                        "signal_uncertainty"   : np.sqrt(smoothedVar), #resorting to this mess bc numpy is throwing the weirdest fkn error...
                        "measurement_duration" : smoothedDur},index=range(len(smoothedX)) )
  return(smoothDF)

