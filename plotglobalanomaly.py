from datetime import datetime
import pandas as pd
import numpy as np
import scipy
from constants import CONSTANTS
import matplotlib.pyplot as plt
import math


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


from averagemandi import mandiarrivalseries
#from averagemandi import mandipriceseries
#from averageretail import retailpriceseries
#from rainfallmonthly import rainfallmonthly
#from rainfallmonthly import avgrainfallmonthly
#from average_export import exportseries
#from averagerainfall import meanrainfallseries

#meanseries = mandiarrivalseries

meanseries = mandiarrivalseries
meanseries = meanseries.rolling(window=30).mean()
meanseries = RemoveNaNFront(meanseries)




def plot_series(series):
  yaxis = list(series)
  xaxis = list(range(0,len(series.index)))
  plt.plot(xaxis,yaxis)
  plt.show()

'''
plt.title('Average Yearly Retail Prices')
plt.xlabel('Time')
plt.ylabel('Prices per Quintal')
meanseriesyear = meanseries.groupby([meanseries.index.month, meanseries.index.day]).mean()
plot_series(meanseriesyear)
'''


start_date = '2006-01-01'
end_date = '2015-06-23'
colors = [ '#e6194b', '#3cb44b' ,'#ffe119', '#0082c8', '#f58231', '#911eb4',  '#000080' , '#800000'  ,'#000000' ]

def plot_year_series(inpseries,givenyear):
  s = str(givenyear)+'-01-01'
  e = str(givenyear)+'-12-31'
  s = datetime.strptime(s,'%Y-%m-%d')
  e = datetime.strptime(e,'%Y-%m-%d')
  series = inpseries[s:e]
  #series = (series - series.mean())/series.std()
  yaxis = list(series)
  xaxis = list(pd.date_range('2006/01/01',freq='D',periods=len(series)))
  plt.plot(xaxis,yaxis, color =colors[(givenyear-2006)%9] , label=str(givenyear))
  
def plot_season1_series(inpseries,givenyear):
  s = str(givenyear)+'-11-01'
  e = str(givenyear+1)+'-01-31'
  s = datetime.strptime(s,'%Y-%m-%d')
  e = datetime.strptime(e,'%Y-%m-%d')
  series = inpseries[s:e]
  yaxis = list(series)
  xaxis = list(pd.date_range('2006/01/01',freq='D',periods=len(series)))
  plt.plot(xaxis,yaxis, color =colors[(givenyear-2006)%9] , label=str(givenyear))


def plot_season2_series(inpseries,givenyear):
  s = str(givenyear)+'-01-01'
  e = str(givenyear)+'-03-31'
  s = datetime.strptime(s,'%Y-%m-%d')
  e = datetime.strptime(e,'%Y-%m-%d')
  series = inpseries[s:e]
  yaxis = list(series)
  xaxis = list(pd.date_range('2006/01/01',freq='D',periods=len(series)))
  plt.plot(xaxis,yaxis, color =colors[(givenyear-2006)%9] , label=str(givenyear))


def plot_nonseason_series(inpseries,givenyear):
  s = str(givenyear)+'-04-01'
  e = str(givenyear)+'-10-31'
  s = datetime.strptime(s,'%Y-%m-%d')
  e = datetime.strptime(e,'%Y-%m-%d')
  series = inpseries[s:e]
  yaxis = list(series)
  xaxis = list(pd.date_range('2006/01/01',freq='D',periods=len(series)))
  plt.plot(xaxis,yaxis, color =colors[(givenyear-2006)%9] , label=str(givenyear))


'''
plt.xlabel('Time')
plt.title('Complete Series')
plot_series(meanseries)
plt.show()
'''


plt.xlabel('Time')
plt.title('Global Anomaly: Average yearly Arrival.')
plt.ylabel('Arrival in Tonnes')
for i in range(0,9):
  plot_year_series(meanseries,2006+i)
plt.legend(loc='best')
plt.show()


plt.xlabel('Time')
plt.title('Global Anomaly: Average Season1.')
for i in range(0,9):
  plot_season1_series(meanseries,2006+i)

plt.legend(loc='best')
plt.show()


plt.xlabel('Time')
plt.title('Global Anomaly: Average Season2.')
for i in range(0,9):
  plot_season2_series(meanseries,2006+i)

plt.legend(loc='best')
plt.show()

plt.xlabel('Time')
plt.title('Global Anomaly: Average Non Season')
for i in range(0,9):
  plot_nonseason_series(meanseries,2006+i)

plt.legend(loc='best')
plt.show()