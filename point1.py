from datetime import datetime
import pandas as pd
import numpy as np
from constants import CONSTANTS
import matplotlib.pyplot as plt
import matplotlib
import sklearn.preprocessing as preprocessing



def whiten(series):
  '''
  Whitening Function
  Formula is
    W[x x.T] = E(D^(-1/2))E.T
  Here x: is the observed series
  Read here more:
  https://www.cs.helsinki.fi/u/ahyvarin/papers/NN00new.pdf
  '''
  import scipy
  EigenValues, EigenVectors = np.linalg.eig(series.cov())
  D = [[0.0 for i in range(0, len(EigenValues))] for j in range(0, len(EigenValues))]
  for i in range(0, len(EigenValues)):
    D[i][i] = EigenValues[i]
  DInverse = np.linalg.matrix_power(D, -1)
  DInverseSqRoot = scipy.linalg.sqrtm(D)
  V = np.dot(np.dot(EigenVectors, DInverseSqRoot), EigenVectors.T)
  series = series.apply(lambda row: np.dot(V, row.T).T, axis=1)
  return series

def whiten_series_list(list):
	for i in range(0,len(list)):
		mean = list[i].mean()
		list[i] -= mean
	temp = pd.DataFrame()
	for i in range(0,len(list)):
		temp[i] = list[i]
	temp = whiten(temp)
	newlist = [temp[i] for i in range(0,len(list))]
	return newlist

def give_average_series(start,end,mandiarrivalseries):
  mandiarrivalexpected = mandiarrivalseries.rolling(window=30,center=True).mean()
  #mandiarrivalexpected = mandiarrivalseries
  type(mandiarrivalexpected)
  mandiarrivalexpected = mandiarrivalexpected.groupby([mandiarrivalseries.index.month, mandiarrivalseries.index.day]).mean()
  idx = pd.date_range(start, end)
  data = [ (mandiarrivalexpected[index.month][index.day]) for index in idx]
  expectedarrivalseries = pd.Series(data, index=idx)
  return expectedarrivalseries

'''
Get Retail Price Series
'''
from averageretail import getcenter
retailpriceseriesmumbai = getcenter('MUMBAI')
retailpriceseriesdelhi = getcenter('DELHI')
retailpriceserieslucknow = getcenter('LUCKNOW')
retailpriceseriesbhub = getcenter('BHUBANESHWAR')
retailpriceseriespatna = getcenter('PATNA')
# [retailpriceseriesbhub,retailpriceseriesdelhi,retailpriceserieslucknow,retailpriceseriesmumbai,retailpriceseriespatna] = whiten_series_list([retailpriceseriesbhub,retailpriceseriesdelhi,retailpriceserieslucknow,retailpriceseriesmumbai,retailpriceseriespatna])
#[retailpriceseriesdelhi,retailpriceserieslucknow,retailpriceseriesmumbai] = whiten_series_list([retailpriceseriesdelhi,retailpriceserieslucknow,retailpriceseriesmumbai])

'''
Get Mandi Price Series
'''
from averagemandi import getmandi
mandipriceseriesdelhi = getmandi('Azadpur',True)
mandiarrivalseriesdelhi = getmandi('Azadpur',False)
mandipriceserieslucknow = getmandi('Devariya',True)
mandiarrivalserieslucknow = getmandi('Devariya',False)
from averagemandi import mandipriceseries
from averagemandi import mandiarrivalseries 
# [mandipriceseriesdelhi,mandipriceserieslucknow,mandipriceseriesmumbai] = whiten_series_list([mandipriceseriesdelhi,mandipriceserieslucknow,mandipriceseries])
# [mandiarrivalseriesdelhi,mandiarrivalserieslucknow,mandiarrivalseriesmumbai] = whiten_series_list([mandiarrivalseriesdelhi,mandiarrivalserieslucknow,mandiarrivalseries])

'''
Get Rainfall Data
'''
from rainfallmonthly import rainfallmonthly

avgrainfall = give_average_series(CONSTANTS['STARTDATE'],CONSTANTS['ENDDATEOLD'],rainfallmonthly)

'''
To measure disturbances during monsoon period July to September
'''
def monsoon():
  for year in range(2006,2015):
    a = rainfallmonthly[str(year)+'-07-01':str(year)+'-09-30']
    b = avgrainfall[str(year)+'-07-01':str(year)+'-09-30']
    corr = np.corrcoef(a,b)[0,1]
    if corr<0.9:
      print year
'''
To measure unseasonal rainfall during 
'''
def unseasonal():
  diff = []
  for year in range(2006,2015):
      a = rainfallmonthly[str(year)+'-10-01':str(year)+'-11-30']
      b = avgrainfall[str(year)+'-10-01':str(year)+'-11-30']
      diff.append((a-b).mean())

  diff = diff / max(diff)

  for year in range(2006,2015):
    d = diff[year-2006]
    if d>0.5:
      print year
'''
To measure excess monsoon 
'''
def excessmonsoon():
  diff = []
  for year in range(2006,2015):
      a = rainfallmonthly[str(year)+'-07-01':str(year)+'-09-30']
      b = avgrainfall[str(year)+'-07-01':str(year)+'-09-30']
      diff.append((a-b).mean())

  diff = diff / max(diff)

  for year in range(2006,2015):
    if d>0.3:
      print year

def arrival(hoarding):
  sdate = []
  h = []
  l = []
  date = CONSTANTS['STARTDATE']
  enddate = CONSTANTS['ENDDATEOLD']
  from datetime import timedelta
  date = datetime.strptime(date,'%Y-%m-%d')
  enddate = datetime.strptime(enddate,'%Y-%m-%d')
  window = 120
  duration = timedelta(days=window)
  while(duration <= enddate-date):
    s = datetime.strftime(date,'%Y-%m-%d')
    e = datetime.strftime(date+timedelta(days=window),'%Y-%m-%d') 
    x1 = (mandiarrivalseries.rolling(window=14,center=True).mean())[s:e]
    x2 = give_average_series(s,e,mandiarrivalseries)
    date = date+timedelta(days=15)
    series = x1-x2
    a = series[60:].max()
    b = series[0:16].min()
    sdate.append(s)
    h.append(a)
    l.append(b)
    # if s == '2011-03-06':
    #   for i in range(0,16):
    #     print x1[i],x2[i]

  hmax = max(h)
  lmin = min(l)

  if hoarding:
    t = 0.5
  else:
    t =0.0

  for i in range(0,len(sdate)):
    if l[i]/lmin > 0.5:
      if h[i]/hmax > t:
        print sdate[i],h[i]/hmax,l[i]/lmin,h[i]-(l[i])


def retail():
  sdate = []
  h = []
  date = CONSTANTS['STARTDATE']
  enddate = CONSTANTS['ENDDATEOLD']
  from datetime import timedelta
  date = datetime.strptime(date,'%Y-%m-%d')
  enddate = datetime.strptime(enddate,'%Y-%m-%d')
  window = 30
  duration = timedelta(days=window)
  while(duration <= enddate-date):
    s = datetime.strftime(date,'%Y-%m-%d')
    e = datetime.strftime(date+timedelta(days=window),'%Y-%m-%d') 
    x1 = (retailpriceseriesmumbai.rolling(window=14,center=True).mean())[s:e]
    x2 = give_average_series(s,e,retailpriceseriesmumbai)
    date = date+timedelta(days=30)
    series = x1-x2
    a = series.max()
    sdate.append(s)
    h.append(a)
    
  hmax = max(h)
  for i in range(0,len(sdate)):
    if h[i]/hmax > 0.5:
      print sdate[i],h[i]/hmax