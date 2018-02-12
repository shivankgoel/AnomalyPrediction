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
Plot Series from dates specified in the function
'''
colors = [ '#e6194b', '#3cb44b' ,'#ffe119', '#0082c8', '#f58231', '#911eb4',  '#000080' , '#800000'  ,'#000000', '#900c3f' ]

def plot_series(inpseries,lbl,clr):
  yaxis = list(inpseries)
  xaxis = list(inpseries.index)
  plt.plot(xaxis,yaxis, color =colors[clr] , label=lbl, linewidth=1.5)


def plotretailprice(series,start,end,averagetoo,roll=False):
  a = series[start:end]
  b = give_average_series(start,end,series)
  if roll:
    a = a.rolling(window=14,center=True).mean()
    b = b.rolling(window=14,center=True).mean()
  plt.title('Retail Price')
  plt.xlabel('Time')
  plt.ylabel('Retail Price per Quintal')
  plot_series(a,'Retail Price',6)
  if(averagetoo):
    plot_series(b,'Average Retail Price',4)
    ma = b
    mstd = give_std_series(start,end,series)
    plt.fill_between(mstd.index, ma-mstd, ma+mstd, color=colors[4], alpha=0.2)
  plt.legend(loc='best')
  plt.show()


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

def printanomalies(series,anomalies,labels,lbl):
  for i in range(0,len(anomalies)):
    if labels[i]==lbl:
      plotretailprice(series,anomalies[0][i],anomalies[1][i],False,True)


def myplot(lbl):
  printanomalies(retailpriceseriesmumbai,anomaliesmumbai,mumbailabels,lbl)
  printanomalies(retailpriceseriesdelhi,anomaliesdelhi,delhilabels,lbl)
  printanomalies(retailpriceserieslucknow,anomalieslucknow,lucknowlabels,lbl)

myplot(4)
