from datetime import datetime
import pandas as pd
import numpy as np
import scipy
from constants import CONSTANTS
import matplotlib
import matplotlib.pyplot as plt
import math
import matplotlib.dates as mdates


font = {'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 15}

matplotlib.rc('font', **font)

colors = [ '#e6194b', '#3cb44b' ,'#ffe119', '#0082c8', '#f58231', '#911eb4',  '#000080' , '#800000'  ,'#000000', '#900c3f' ]

'''
Plot Series from dates specified in the function
'''
def plot_series(inpseries,lbl,clr):
  s = '2006-01-01'
  e = '2015-01-01'
  s = CONSTANTS['STARTDATE']
  e = CONSTANTS['ENDDATE']
  s = datetime.strptime(s,'%Y-%m-%d')
  e = datetime.strptime(e,'%Y-%m-%d')
  inpseries = inpseries[s:e]
  yaxis = list(inpseries)
  xaxis = list(inpseries.index)
  plt.plot(xaxis,yaxis, color =colors[clr] , label=lbl, linewidth=1.5)

'''
Plot Series from dates specified in the function
'''
def plot_series_axis(inpseries,lbl,clr,ax):
  s = '2006-01-01'
  e = '2015-01-01'
  s = CONSTANTS['STARTDATE']
  e = CONSTANTS['ENDDATE']
  s = datetime.strptime(s,'%Y-%m-%d')
  e = datetime.strptime(e,'%Y-%m-%d')
  inpseries = inpseries[s:e]
  yaxis = list(inpseries)
  xaxis = list(inpseries.index)
  ax.plot(xaxis,yaxis, color =colors[clr] , label=lbl, linewidth=1.5)
  
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

def Normalize(series):
  series = (series - series.mean())/series.std()
  return series

from averagemandi import mandiarrivalseries
from averagemandi import mandipriceseries
from averagemandi import specificarrivalseries
from averagemandi import specificpriceseries

from averageretail import retailpriceseries
from averageretail import specificretailprice

from average_export import exportseries
from rainfallmonthly import rainfallmonthly
#Avg Rainfall of 3 regions
from rainfallmonthly import avgrainfallmonthly

from fuelprice import fuelpricedelhi
from fuelprice import fuelpricemumbai

from oilmonthlyseries import oilmonthlyseries
from cpimonthlyseries import cpimonthlyseries

'''
0 Red
1 Green
2 Yellow
3 Light Blue
4 Orange
5 Purple
6 Dark Blue
7 Maroon
'''

'''
Returns average series from start date to end date after rolling.
If end-start > 1year the pattern repeats
'''
def give_average_series(start,end,mandiarrivalseries):
  mandiarrivalexpected = mandiarrivalseries.rolling(window=30,center=True).mean()
  #mandiarrivalexpected = mandiarrivalseries
  mandiarrivalexpected = mandiarrivalexpected.groupby([mandiarrivalseries.index.month, mandiarrivalseries.index.day]).mean()
  idx = pd.date_range(start, end)
  data = [ (mandiarrivalexpected[index.month][index.day]) for index in idx]
  expectedarrivalseries = pd.Series(data, index=idx)
  return expectedarrivalseries

def give_std_series(start,end,mandiarrivalseries):
  mandiarrivalexpected = mandiarrivalseries.rolling(window=30,center=True).mean()
  #mandiarrivalexpected = mandiarrivalseries
  mandiarrivalexpected = mandiarrivalexpected.groupby([mandiarrivalseries.index.month, mandiarrivalseries.index.day]).std()
  idx = pd.date_range(start, end)
  data = [ (mandiarrivalexpected[index.month][index.day]) for index in idx]
  expectedarrivalseries = pd.Series(data, index=idx)
  return expectedarrivalseries

def plotweather(start,end,averagetoo,roll=False):
  a = rainfallmonthly[start:end]
  b = give_average_series(start,end,rainfallmonthly)
  if roll:
    a = a.rolling(window=14,center=True).mean()
    b = b.rolling(window=14,center=True).mean()
  plt.title('Rainfall')
  plt.xlabel('Time')
  plt.ylabel('Rainfall in mm')
  plot_series(a,'Rainfall',6)
  if(averagetoo):
    plot_series(b,'Average Rainfall',4)
    ma=b
    mstd = give_std_series(start,end,rainfallmonthly)
    plt.fill_between(mstd.index, ma-0.5*mstd, ma+0.5*mstd, color=colors[4], alpha=0.2)
  plt.legend(loc='best')
  plt.show()

def plotcpi(start,end,averagetoo,roll=False):
  a = cpimonthlyseries[start:end]
  b = give_average_series(start,end,cpimonthlyseries)
  if roll:
    a = a.rolling(window=14,center=True).mean()
    b = b.rolling(window=14,center=True).mean()
  plt.title('Cumulative Price Index')
  plt.xlabel('Time')
  plt.ylabel('CPI')
  plot_series(a,'CPI',6)
  if(averagetoo):
    plot_series(b,'Average CPI',4)
    ma = b
    mstd = give_std_series(start,end,cpimonthlyseries)
    plt.fill_between(mstd.index, ma-mstd, ma+mstd, color=colors[4], alpha=0.2)
  plt.legend(loc='best')
  plt.show()

def plotarrival(start,end,averagetoo,roll=False):
  a = mandiarrivalseries[start:end]
  b = give_average_series(start,end,mandiarrivalseries)
  if roll:
    a = a.rolling(window=14,center=True).mean()
    b = b.rolling(window=14,center=True).mean()
  plt.title('Arrival')
  plt.xlabel('Time')
  plt.ylabel('Arrival in Metric Tons(Tonnes)')
  plot_series(a,'Arrival',1)
  if(averagetoo):
    plot_series(b,'Average Arrival',2)
    ma = b
    mstd = give_std_series(start,end,mandiarrivalseries)
    plt.fill_between(mstd.index, ma-mstd, ma+mstd, color=colors[2], alpha=0.2)
  plt.legend(loc='best')
  plt.show()

def plotmandiprice(start,end,averagetoo,roll=False):
  a = mandipriceseries[start:end]
  b = give_average_series(start,end,mandipriceseries)
  if roll:
    a = a.rolling(window=14,center=True).mean()
    b = b.rolling(window=14,center=True).mean()
  plt.title('Mandi Price')
  plt.xlabel('Time')
  plt.ylabel('Mandi Price per Quintal')
  plot_series(a,'Mandi Price',6)
  if(averagetoo):
    plot_series(b,'Average Mandi Price',4)
    ma = b
    mstd = give_std_series(start,end,mandipriceseries)
    plt.fill_between(mstd.index, ma-mstd, ma+mstd, color=colors[4], alpha=0.2)
  plt.legend(loc='best')
  plt.show()

def plotretailprice(start,end,averagetoo,roll=False):
  a = retailpriceseries[start:end]
  b = give_average_series(start,end,retailpriceseries)
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
    mstd = give_std_series(start,end,retailpriceseries)
    plt.fill_between(mstd.index, ma-mstd, ma+mstd, color=colors[4], alpha=0.2)
  plt.legend(loc='best')
  plt.show()

def plotretailvsmandi(start,end,averagetoo,roll=False):
  a = retailpriceseries[start:end]
  b = mandipriceseries[start:end]
  if roll:
    a = a.rolling(window=14,center=True).mean()
    b = b.rolling(window=14,center=True).mean()
  plt.title('Retail vs Mandi price')
  plt.xlabel('Time')
  plt.ylabel('Price per Quintal')
  plot_series(a,'Retail Price',6)
  if(averagetoo):
    plot_series(b,'Mandi Price',3)
  plt.legend(loc='best')
  plt.show()

def plotsingleseries(series,title,xlabel,ylabel,start,end,averagetoo,roll=False,sigma=1):
  a = series[start:end]
  if(averagetoo):
    b = give_average_series(start,end,series)
  if roll:
    a = a.rolling(window=14,center=True).mean()
    if(averagetoo):
      b = b.rolling(window=14,center=True).mean()
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plot_series(a,title,6)
  if(averagetoo):
    plot_series(b,'Average '+title,4)
    ma = b
    mstd = give_std_series(start,end,series)
    plt.fill_between(mstd.index, ma-sigma*mstd, ma+sigma*mstd, color=colors[4], alpha=0.2)
  plt.legend(loc='best')
  plt.show()

def plottwosingleseries(series1,series2,title,xlabel,ylabel,start,end,averagetoo,roll=False,sigma=1):
  a = series1[start:end]
  b = series2[start:end]
  if(averagetoo):
    b1 = give_average_series(start,end,series2)
  if roll:
    a = a.rolling(window=14,center=True).mean()
    if(averagetoo):
      b = b.rolling(window=14,center=True).mean()
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plot_series(a,title,6)
  plot_series(b,title,3)
  if(averagetoo):
    plot_series(b,'Average '+title,4)
    ma = b1
    mstd = give_std_series(start,end,series2)
    plt.fill_between(mstd.index, ma-sigma*mstd, ma+sigma*mstd, color=colors[4], alpha=0.2)
  plt.legend(loc='best')
  plt.show()

def plotsingleseriesylimit(series,title,xlabel,ylabel,ymin,ymax,start,end,averagetoo,roll=False,sigma=1):
  a = series[start:end]
  if(averagetoo):
    b = give_average_series(start,end,series)
  if roll:
    a = a.rolling(window=14,center=True).mean()
    if(averagetoo):
      b = b.rolling(window=14,center=True).mean()
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  axes = plt.gca()
  axes.set_ylim([ymin,ymax])
  plot_series(a,title,6)
  if(averagetoo):
    plot_series(b,'Average '+title,4)
    ma = b
    mstd = give_std_series(start,end,series)
    plt.fill_between(mstd.index, ma-sigma*mstd, ma+sigma*mstd, color=colors[4], alpha=0.2)
  plt.legend(loc='best')
  plt.show()

def plotdoubleseries(s1,s2,x1,x2,y1,y2,start,end,avg1=False,avg2=False,roll1=False,roll2=False):
  fig, ax1 = plt.subplots()
  ax1.set_xlabel(x1)
  ax1.set_ylabel(y1)
  ax2 = ax1.twinx()
  ax2.set_xlabel(x2)
  ax2.set_ylabel(y2)
  if roll1:
    s1 = s1.rolling(window=14,center=True).mean()
  if roll2: 
    s2 = s2.rolling(window=14,center=True).mean()
  plot_series_axis(s1,y1,6,ax1)
  plot_series_axis(s2,y2,3,ax2)
  if avg1:
    a = give_average_series(s1)
    plot_series_axis(a,'Average '+y1,4,1)
    ma1 = a
    mstd1 = give_std_series(start,end,s1)
    ax1.fill_between(mstd1.index, ma1-mstd1, ma1+mstd1, color=colors[6], alpha=0.1)
  if avg2:
    b = give_average_series(s2)
    plot_series_axis(b,'Average '+y2,4,2)
    ma2 = b
    mstd2 = give_std_series(start,end,s2)
    ax2.fill_between(mstd2.index, ma2-mstd2, ma2+mstd2, color=colors[3], alpha=0.1)
  fig.tight_layout()
  ax1.legend(loc = (0.05,0.9), frameon = False)
  ax2.legend(loc = (0.05,0.80), frameon = False)
  plt.show()

def linear_reg(x,y):
  from sklearn import linear_model
  from sklearn.metrics import mean_squared_error, r2_score
  from sklearn.utils import shuffle
  x,y = shuffle(x,y)
  train_size = (int)(0.80 * len(x))
  train = x[:train_size]
  train_labels = y[:train_size]
  test = x[train_size:]
  test_labels = y[train_size:]
  regr = linear_model.LinearRegression()
  regr.fit(train.values.reshape(-1,1), train_labels)
  predicted_labels = regr.predict(test.values.reshape(-1,1))
  print('Variance score: %.2f' % r2_score(test_labels, predicted_labels ))
  print("Mean squared error: %.2f" % mean_squared_error(test_labels, predicted_labels))
  plt.plot(test, test_labels, color='blue', linewidth=2)
  plt.plot(test, predicted_labels, color='red', linewidth=2)
  plt.show()


'''
weather only: '01-06-2007','31-12-2007'

'''


fstart = CONSTANTS['STARTDATE']
fend = CONSTANTS['ENDDATEOLD']
#pstart = ['2016-04-01']
#pend = ['2017-02-01']
pstart = ['2013-03-01','2010-03-01','2007-01-01','2006-04-01','2008-05-01']
pend = ['2014-04-01','2011-05-01','2008-05-01','2007-03-01','2009-01-01']
# pstart = [fstart]
# pend = [fend]

for i in range(0,len(pstart)):
    #Make All True Except First
    plotweather(pstart[i],pend[i],True,False)
    plotarrival(pstart[i],pend[i],True,True)
    plotmandiprice(pstart[i],pend[i],True,True)
    plotretailprice(pstart[i],pend[i],True,True)
    plotretailvsmandi(pstart[i],pend[i],True,False)
    plotsingleseries(retailpriceseries-mandipriceseries,'Difference','Time','Price per Quintal',pstart[i],pend[i],False,False )

# plotsingleseries(exportseries,'Export','Time','Export in Metric Tons',pstart,pend,False,True)
# plotcpi(fstart,fend,False,False)
# plotdoubleseries(mandipriceseries,cpimonthlyseries,'Time','Time','Mandi Price','CPI',fstart,fend)
# linear_reg(cpimonthlyseries,mandipriceseries)
# from loadmonthlyseries import mpimonthlyseries
# from loadmonthlyseries import mpionionmonthlyseries
# from loadmonthlyseries import cpimonthlyseries 
# plotdoubleseries(cpimonthlyseries,mpimonthlyseries,'Time','Time','CPI','WPI',fstart,fend)
# plottwosingleseries(cpimonthlyseries,mpionionmonthlyseries,'','Time','',fstart,fend,False,True )
#plotweather(CONSTANTS['STARTDATE'],CONSTANTS['ENDDATEOLD'],True)