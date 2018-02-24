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

def display_anomalies(anomalieslist, anomaly, labels):
  count = {'01':0,'02':0,'03':0,'04':0,'05':0,'06':0,'07':0,'08':0,'09':0,'10':0,'11':0,'12':0}
  for i in range(0,len(anomalieslist)):
    if( labels[i] == anomaly):
      count[anomalieslist[0][i][5:7]] = count[anomalieslist[0][i][5:7]] + 1
  return count


def overlapping(anomalies,s,e,labels):
  startdate = datetime.strptime(s,'%Y-%m-%d')
  if(startdate.month >=2 and startdate.month <= 7):
    return True
  for i in range(0,len(anomalies)):
    if(labels[i] == 2 or labels[i] == 3 or labels[i] == 5):  
      if((anomalies[0][i]<=s and s<=anomalies[1][i]) or  (anomalies[0][i]<=e and e<=anomalies[1][i])):
        return True
  return False

def findnormal_restricted(anomalies,series,labels):
  sdate = []
  edate = []
  date = CONSTANTS['STARTDATE']
  enddate = CONSTANTS['ENDDATEOLD']
  from datetime import timedelta
  date = datetime.strptime(date,'%Y-%m-%d')+timedelta(days=21)
  enddate = datetime.strptime(enddate,'%Y-%m-%d')
  window = 42
  duration = timedelta(days=window)
  while(duration <= enddate-date):
    s = datetime.strftime(date,'%Y-%m-%d')
    e = datetime.strftime(date+timedelta(days=window),'%Y-%m-%d') 
    x1 = (series.rolling(window=14,center=True).mean())[s:e]
    date = date+timedelta(days=10)
    if not overlapping(anomalies,s,e,labels):
      a = x1.min()
      b = x1.max()
      if(b-a <=400 and b-a>=200):
        sdate.append(s)
        edate.append(e)
        print b-a
        date = date+timedelta(days=30)
  return sdate,edate


def createnormalfile(path,anomaliesmumbai,retailpriceseriesmumbai,labels):
  a,b = findnormal_restricted(anomaliesmumbai,retailpriceseriesmumbai,labels)
  newdf = anomaliesmumbai
  newdf[1] = ' '+newdf[1]
  for i in range(len(a)):
    newdf.loc[i+len(anomaliesmumbai)] = [a[i],' '+b[i],' NormalR']
  result = newdf.sort_values([0])
  result.to_csv(path, header=None,index=None)

createnormalfile('data/anomaly/newnormalmumbai.csv',anomaliesmumbai,retailpriceseriesmumbai,mumbailabels)
createnormalfile('data/anomaly/newnormaldelhi.csv',anomaliesdelhi,retailpriceseriesdelhi,delhilabels)
createnormalfile('data/anomaly/newnormallucknow.csv',anomalieslucknow,retailpriceserieslucknow,lucknowlabels)


def get_anomalies_new(path):
  anomalies = pd.read_csv(path, header=None, index_col=None)
  anomalies[0] = [ datetime.strftime(datetime.strptime(date, '%Y-%m-%d'),'%Y-%m-%d') for date in anomalies[0]]
  anomalies[1] = [ datetime.strftime(datetime.strptime(date, ' %Y-%m-%d'),'%Y-%m-%d') for date in anomalies[1]]
  return anomalies

anomaliesmumbai = get_anomalies_new('data/anomaly/newnormalmumbai.csv')
anomaliesdelhi = get_anomalies_new('data/anomaly/newnormaldelhi.csv')
anomalieslucknow = get_anomalies_new('data/anomaly/newnormallucknow.csv')


def newlabels(anomalies,oldlabels):
  print len(anomalies[anomalies[2] != ' NormalR']), len(oldlabels)
  labels = []
  k=0
  for i in range(0,len(anomalies)):
    if(anomalies[2][i] != ' NormalR'):
      labels.append(oldlabels[k])
      #print k,oldlabels[k]
      k = k+1
    else:
      labels.append(6)
  return labels


delhilabelsnew = newlabels(anomaliesdelhi,delhilabels)
lucknowlabelsnew = newlabels(anomalieslucknow,lucknowlabels)
mumbailabelsnew = newlabels(anomaliesmumbai,mumbailabels)
print len(delhilabelsnew)
print len(lucknowlabelsnew)
print len(mumbailabelsnew)
# print len(delhilabelsnew == 2)
count1 = display_anomalies(anomaliesmumbai,6,mumbailabelsnew)
count2 = display_anomalies(anomaliesdelhi,6,delhilabelsnew)
count3 = display_anomalies(anomalieslucknow,6,lucknowlabelsnew)
c = 0
for keys in count1:
  # print keys, count1[keys]+count2[keys]+count3[keys]
  c = c + count1[keys] + count2[keys] + count3[keys]

print c
