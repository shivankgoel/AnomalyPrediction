from datetime import datetime
import pandas as pd
import numpy as np
from constants import CONSTANTS
import matplotlib.pyplot as plt
import matplotlib
import sklearn.preprocessing as preprocessing
import math



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
mandipriceserieslucknow = getmandi('Bahraich',True)
mandiarrivalserieslucknow = getmandi('Bahraich',False)
from averagemandi import mandipriceseries
from averagemandi import mandiarrivalseries 
# [mandipriceseriesdelhi,mandipriceserieslucknow,mandipriceseriesmumbai] = whiten_series_list([mandipriceseriesdelhi,mandipriceserieslucknow,mandipriceseries])
# [mandiarrivalseriesdelhi,mandiarrivalserieslucknow,mandiarrivalseriesmumbai] = whiten_series_list([mandiarrivalseriesdelhi,mandiarrivalserieslucknow,mandiarrivalseries])


def get_anomalies(path):
  anomalies = pd.read_csv(path, header=None, index_col=None)
  anomalies[0] = [ datetime.strftime(datetime.strptime(date, '%d/%m/%Y'),'%Y-%m-%d') for date in anomalies[0]]
  anomalies[1] = [ datetime.strftime(datetime.strptime(date, ' %d/%m/%Y'),'%Y-%m-%d') for date in anomalies[1]]
  return anomalies

def get_anomalies_new(path):
  anomalies = pd.read_csv(path, header=None, index_col=None)
  anomalies[0] = [ datetime.strftime(datetime.strptime(date, '%Y-%m-%d'),'%Y-%m-%d') for date in anomalies[0]]
  anomalies[1] = [ datetime.strftime(datetime.strptime(date, ' %Y-%m-%d'),'%Y-%m-%d') for date in anomalies[1]]
  return anomalies


anomaliesmumbai = get_anomalies_new('data/anomaly/newnormalmumbai.csv')
anomaliesdelhi = get_anomalies_new('data/anomaly/newnormaldelhi.csv')
anomalieslucknow = get_anomalies_new('data/anomaly/newnormallucknow.csv')

# Labelling 
# Transport:  1
# Weather:  2
# Inflation:  3
# Fuel:   4
# Hoarding: 5

delhilabels = [2,4,1,3,1,2,2,2,3,4,1,2,2,1,4,2,5,5,2,2,3,1,5,4,2,5,5,5,3,5,3,5,2,2,5,2,2,5,5,5,2,5,5,5,2,2,2,3,1,5,1,2]
lucknowlabels = [2,1,1,2,2,2,5,4,3,1,5,5,5,3,2,2,5,5,4,3,4,5,4,2,5,5,5,5,2,2,3,2,2,5,3,2,5,2]
mumbailabels = [2,2,2,3,5,1,2,5,2,5,2,2,2,4,2,3,2,3,3,1,1,2,5,5,3,3,2,5,3,5,5,5,2,5,5,5,2,5,2,5,3,2,5,2,5,3,2,1,5,5,2,1,2,2,2,1,5,5,2]


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

def overlapping(anomalies,s,e,labels):
  for i in range(0,len(anomalies)):
    # if(labels[i] == 2 or labels[i] == 3 or labels[i] == 5 or labels[i] == 6):  
    if((anomalies[0][i]<=s and s<=anomalies[1][i]) or  (anomalies[0][i]<=e and e<=anomalies[1][i])):
      return True
  return False

def findnormal(anomalies,series,labels):
  sdate = []
  edate = []
  date = CONSTANTS['STARTDATE']
  enddate = CONSTANTS['ENDDATE']
  from datetime import timedelta
  date = datetime.strptime(date,'%Y-%m-%d')+timedelta(days=21)
  enddate = datetime.strptime(enddate,'%Y-%m-%d')
  window = 42
  duration = timedelta(days=window)
  while(duration <= enddate-date):
    s = datetime.strftime(date,'%Y-%m-%d')
    e = datetime.strftime(date+timedelta(days=window),'%Y-%m-%d') 
    x1 = (series.rolling(window=14,center=True).mean())[s:e]
    date = date+timedelta(days=15)
    if not overlapping(anomalies,s,e,labels):
      a = x1.min()
      b = x1.max()
      if(math.isnan(a) == False and math.isnan(b) == False and b-a > 0 ):
        sdate.append(s)
        edate.append(e)
        print a,b
        date = date+timedelta(days=30)
  return sdate,edate


def createnormalfile(path,anomaliesmumbai,retailpriceseriesmumbai,labels):
  a,b = findnormal(anomaliesmumbai,retailpriceseriesmumbai,labels)
  newdf = anomaliesmumbai
  newdf[1] = ' '+newdf[1]
  for i in range(len(a)):
    newdf.loc[i+len(anomaliesmumbai)] = [a[i],' '+b[i],' Normal']
  result = newdf.sort_values([0])
  result.to_csv(path, header=None,index=None)

createnormalfile('data/anomaly/normalmumbai.csv',anomaliesmumbai,retailpriceseriesmumbai,mumbailabelsnew)
createnormalfile('data/anomaly/normaldelhi.csv',anomaliesdelhi,retailpriceseriesdelhi,delhilabelsnew)
createnormalfile('data/anomaly/normallucknow.csv',anomalieslucknow,retailpriceserieslucknow,lucknowlabelsnew)



# def get_anomalies_new(path):
#   anomalies = pd.read_csv(path, header=None, index_col=None)
#   anomalies[0] = [ datetime.strftime(datetime.strptime(date, '%Y-%m-%d'),'%Y-%m-%d') for date in anomalies[0]]
#   anomalies[1] = [ datetime.strftime(datetime.strptime(date, ' %Y-%m-%d'),'%Y-%m-%d') for date in anomalies[1]]
#   return anomalies

anomaliesmumbai = get_anomalies_new('data/anomaly/normalmumbai.csv')
anomaliesdelhi = get_anomalies_new('data/anomaly/normaldelhi.csv')
anomalieslucknow = get_anomalies_new('data/anomaly/normallucknow.csv')


def newlabels2(anomalies,oldlabels):
  print len(anomalies[anomalies[2] != ' Normal']), len(oldlabels)
  labels = []
  k=0
  for i in range(0,len(anomalies)):
    if(anomalies[2][i] != ' Normal'):
      labels.append(oldlabels[k])
      #print k,oldlabels[k]
      k = k+1
    else:
      labels.append(7)
  return labels


delhilabelsnew = newlabels2(anomaliesdelhi,delhilabelsnew)
lucknowlabelsnew = newlabels2(anomalieslucknow,lucknowlabelsnew)
mumbailabelsnew = newlabels2(anomaliesmumbai,mumbailabelsnew)
print len(delhilabelsnew)
print len(lucknowlabelsnew)
print len(mumbailabelsnew)

def display_anomalies(anomalieslist, anomaly, labels):
  count = {'01':0,'02':0,'03':0,'04':0,'05':0,'06':0,'07':0,'08':0,'09':0,'10':0,'11':0,'12':0}
  for i in range(0,len(anomalieslist)):
    if( labels[i] == anomaly):
      count[anomalieslist[0][i][5:7]] = count[anomalieslist[0][i][5:7]] + 1
  return count

# count1 = display_anomalies(anomaliesmumbai,7,mumbailabelsnew)
# count2 = display_anomalies(anomaliesdelhi,7,delhilabelsnew)
# count3 = display_anomalies(anomalieslucknow,7,lucknowlabelsnew)
# c = 0
# for keys in count1:
#   # print keys, count1[keys]+count2[keys]+count3[keys]
#   c = c + count1[keys] + count2[keys] + count3[keys]

# print c