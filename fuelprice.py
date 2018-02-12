from datetime import datetime
import pandas as pd
import numpy as np
import scipy
from constants import CONSTANTS
import matplotlib.pyplot as plt
import math

START = CONSTANTS['STARTDATE']
END = CONSTANTS['ENDDATE']

def getfuelseries():
	export = pd.read_csv('data/original/fuel_india_monthly.csv')
	export[3] = export.apply(lambda row: datetime.strptime(row[3], '%d-%m-%Y'), axis=1)
	export.index = export[3]
	idx = pd.date_range(START, END)
	export = export.reindex(idx, fill_value=0)
	exportseriesDelhi = export['Delhi']
	exportseriesDelhi = exportseriesDelhi.replace(0.0, np.NaN, regex=True)
	exportseriesDelhi = exportseriesDelhi.interpolate(method='linear',limit_direction='both')
	exportseriesMumbai = export['Mumbai']
	exportseriesMumbai = exportseriesMumbai.replace(0.0, np.NaN, regex=True)
	exportseriesMumbai = exportseriesMumbai.interpolate(method='linear',limit_direction='both')
	return [exportseriesDelhi,exportseriesMumbai] 


fuelpricedelhi,fuelpricemumbai = getfuelseries()


def RemoveNaNFront(series):
  index = 0
  while True:
    if(not np.isfinite(series[index])):
      index += 1
    else:
      break
  if(index < len(series)):
    for i in range(0, index):
      series[i] = series[index]
  return series


fuelpricemumbai = RemoveNaNFront(fuelpricemumbai)
fuelpricedelhi = RemoveNaNFront(fuelpricedelhi)