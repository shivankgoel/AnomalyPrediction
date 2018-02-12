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


def get_anomalies(path):
  anomalies = pd.read_csv(path, header=None, index_col=None)
  anomalies[0] = [ datetime.strftime(datetime.strptime(date, '%d/%m/%Y'),'%Y-%m-%d') for date in anomalies[0]]
  anomalies[1] = [ datetime.strftime(datetime.strptime(date, ' %d/%m/%Y'),'%Y-%m-%d') for date in anomalies[1]]
  return anomalies

anomaliesmumbai = get_anomalies('data/anomaly/mumbai.csv')
anomaliesdelhi = get_anomalies('data/anomaly/delhi.csv')
anomalieslucknow = get_anomalies('data/anomaly/lucknow.csv')

# Labelling 
# Transport:  1
# Weather:  2
# Inflation:  3
# Fuel:   4
# Hoarding: 5


delhilabels = [2,4,1,3,1,2,2,2,3,4,1,2,2,1,4,2,5,5,2,2,3,1,5,4,2,5,5,5,3,5,3,5,2,2,5,2,2,5,5,5,2,5,5]
lucknowlabels = [2,1,1,2,2,2,5,4,3,1,5,5,5,3,2,2,5,5,4,3,4,5,4,2,5,5,5,5,2,2]
mumbailabels = [2,2,2,3,5,1,2,5,2,5,2,2,2,4,2,3,2,3,3,1,1,2,5,5,3,3,2,5,3,5,5,5,2,5,5,5,2,5,2,5]

def overlapping(anomalies,s,e):
  for i in range(0,len(anomalies)):
    if((anomalies[0][i]<=s and s<=anomalies[1][i]) or  (anomalies[0][i]<=e and e<=anomalies[1][i])):
      return True
  return False

def findnormal(anomalies,series):
  sdate = []
  edate = []
  date = CONSTANTS['STARTDATE']
  enddate = CONSTANTS['ENDDATEOLD']
  from datetime import timedelta
  date = datetime.strptime(date,'%Y-%m-%d')
  enddate = datetime.strptime(enddate,'%Y-%m-%d')
  window = 43
  duration = timedelta(days=window)
  while(duration <= enddate-date):
    s = datetime.strftime(date,'%Y-%m-%d')
    e = datetime.strftime(date+timedelta(days=window),'%Y-%m-%d') 
    x1 = (series.rolling(window=14,center=True).mean())[s:e]
    date = date+timedelta(days=10)
    if not overlapping(anomalies,s,e):
      a = x1.min()
      b = x1.max()
      if(b-a <=120 and b-a>=50):
        sdate.append(s)
        edate.append(e)
        print b-a
        date = date+timedelta(days=30)
  return sdate,edate


def createnormalfile(path,anomaliesmumbai,retailpriceseriesmumbai):
  a,b = findnormal(anomaliesmumbai,retailpriceseriesmumbai)
  newdf = anomaliesmumbai
  newdf[1] = ' '+newdf[1]
  for i in range(len(a)):
    newdf.loc[i+len(anomaliesmumbai)] = [a[i],' '+b[i],' Normal']
  result = newdf.sort_values([0])
  result.to_csv(path, header=None,index=None)

createnormalfile('data/anomaly/normalmumbai.csv',anomaliesmumbai,retailpriceseriesmumbai)
createnormalfile('data/anomaly/normaldelhi.csv',anomaliesdelhi,retailpriceseriesdelhi)
createnormalfile('data/anomaly/normallucknow.csv',anomalieslucknow,retailpriceserieslucknow)


def get_anomalies_new(path):
  anomalies = pd.read_csv(path, header=None, index_col=None)
  anomalies[0] = [ datetime.strftime(datetime.strptime(date, '%Y-%m-%d'),'%Y-%m-%d') for date in anomalies[0]]
  anomalies[1] = [ datetime.strftime(datetime.strptime(date, ' %Y-%m-%d'),'%Y-%m-%d') for date in anomalies[1]]
  return anomalies

anomaliesmumbai = get_anomalies_new('data/anomaly/normalmumbai.csv')
anomaliesdelhi = get_anomalies_new('data/anomaly/normaldelhi.csv')
anomalieslucknow = get_anomalies_new('data/anomaly/normallucknow.csv')


def newlabels(anomalies,oldlabels):
  print len(anomalies[anomalies[2] != ' Normal']), len(oldlabels)
  labels = []
  k=0
  for i in range(0,len(anomalies)):
    if(anomalies[2][i] != ' Normal'):
      labels.append(oldlabels[k])
      #print k,oldlabels[k]
      k = k+1
    else:
      labels.append(6)
  return labels


delhilabelsnew = newlabels(anomaliesdelhi,delhilabels)
lucknowlabelsnew = newlabels(anomalieslucknow,lucknowlabels)
mumbailabelsnew = newlabels(anomaliesmumbai,mumbailabels)