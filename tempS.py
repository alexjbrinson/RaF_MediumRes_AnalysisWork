import math
import numpy as np
import scipy as sp
import scipy.odr as spodr
import pandas as pd
import os.path
import json
import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches
#from matplotlib.patches import Rectangle
#import numdifftools
import time
from datetime import date
import LoadingAndMungingData as lmd
import FrequencyConvertedUtilityKitAlternate as FCUK
massList=np.array([242,243,244,245, 247])
transitionLabels = np.array(["%d->%d"%(i,i) for i in range(4)])
estimationMethodLabels = np.array(['SkewedMu','LocMaxDat','LocMaxFit','LocMaxPoly'])
estimationStatisticsLabels = np.array(['mean','stderr','range'])
isotopeDataFrame = pd.DataFrame(data=np.loadtxt('./FitResults/OutputFiles/isotopeDataFrame.csv'), index=massList, columns = pd.MultiIndex.from_product([transitionLabels,estimationMethodLabels, estimationStatisticsLabels], names=['Transitions','Methods','Stats']) )
print("isotopeDataFrame:\n",isotopeDataFrame)