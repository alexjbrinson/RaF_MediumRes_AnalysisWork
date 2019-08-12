import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os.path
import LoadingAndMungingData as lmd
import FrequencyConvertedUtilityKit as fcuk
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import json

if __name__ == '__main__':
  m=245
  oldScans = np.array([2130,2131,2132,2324,2325,2346,2360,2364,2365,2368,2375,2376,2135,2136,2137,2138,2139,2164,2165,2178,2309,2310,2317,2319,2320,2340,2341,2349,2350])
  newScans = np.array([2184,2185,2274,2275,2276,2277,2278,2278,2280,2282,2316,2347,2353,2354,2360,2367,2372,2373,2374,2382])
  scansToSend = np.concatenate((oldScans,newScans))
  print("scansToSend = %s"%repr(scansToSend))
  if not os.path.exists('./ForSilviu/Mass%dtimestampDataframes/'%m): os.makedirs('./ForSilviu/Mass%dtimestampDataframes/'%m)
  '''for scan in scansToSend:
    print("now preparing scan %d"%scan)
    mfba = lmd.rawDatPrep(m,scan, cleanWM=True,verbose=False)
    mfba.to_csv(path_or_buf='./ForSilviu/Mass%dtimestampDataframes/mass%d_scan%dDataframe.csv'%(m,m,scan))'''
  #Test
  #mfba = pd.read_csv('./ForSilviu/Mass%dtimestampDataframes/mass%d_scan%dDataframe.csv'%(m,m,2309),index_col=0)
  mfba = lmd.mergeDatPrepped('./ForSilviu/Mass%dtimestampDataframes/'%m,m,[2309,2310])
  print(mfba)
  output = lmd.makeUseable(mfba, resolution=.1, ltrim=13000,rtrim=15000, noNaNsense=True,cropSparseEnds=True, verbose=True)
  print(output)
  lmd.plotData(output,m,2,2309,resolution=.1)
  plt.show()
