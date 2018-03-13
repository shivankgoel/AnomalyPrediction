from sklearn.metrics import f1_score
from datetime import timedelta
from datetime import datetime
import pandas as pd
import numpy as np
from constants import CONSTANTS
import matplotlib.pyplot as plt
import matplotlib
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import os
cwd = os.getcwd()


delhilabels = [2,4,1,3,1,2,2,2,3,4,1,2,2,1,4,2,5,5,2,2,3,1,5,4,2,5,5,5,3,5,3,5,2,2,5,2,2,5,5,5,2,5,5,5,2,2,2,3,1,5,1,2]
lucknowlabels = [2,1,1,2,2,2,5,4,3,1,5,5,5,3,2,2,5,5,4,3,4,5,4,2,5,5,5,5,2,2,3,2,2,5,3,2,5,2]
mumbailabels = [2,2,2,3,5,1,2,5,2,5,2,2,2,4,2,3,2,3,3,1,1,2,5,5,3,3,2,5,3,5,5,5,2,5,5,5,2,5,2,5,3,2,5,2,5,3,2,1,5,5,2,1,2,2,2,1,5,5,2]


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


from averageretail import getcenter
retailpriceseriesmumbai = getcenter('MUMBAI')
retailpriceseriesdelhi = getcenter('DELHI')
retailpriceserieslucknow = getcenter('LUCKNOW')
retailpriceseriesbhub = getcenter('BHUBANESHWAR')
retailpriceseriespatna = getcenter('PATNA')

[retailpriceseriesdelhi,retailpriceserieslucknow,retailpriceseriesmumbai] = whiten_series_list([retailpriceseriesdelhi,retailpriceserieslucknow,retailpriceseriesmumbai])

from averagemandi import getmandi
mandipriceseriesdelhi = getmandi('Azadpur',True)
mandiarrivalseriesdelhi = getmandi('Azadpur',False)
mandipriceserieslucknow = getmandi('Bahraich',True)
mandiarrivalserieslucknow = getmandi('Bahraich',False)
from averagemandi import mandipriceseries
from averagemandi import mandiarrivalseries 
mandipriceseriesmumbai = mandipriceseries
mandiarrivalseriesmumbai = mandiarrivalseries
[mandipriceseriesdelhi,mandipriceserieslucknow,mandipriceseriesmumbai] = whiten_series_list([mandipriceseriesdelhi,mandipriceserieslucknow,mandipriceseries])
[mandiarrivalseriesdelhi,mandiarrivalserieslucknow,mandiarrivalseriesmumbai] = whiten_series_list([mandiarrivalseriesdelhi,mandiarrivalserieslucknow,mandiarrivalseries])


def adjust_anomaly_window(anomalies,series):
  for i in range(0,len(anomalies)):
    anomaly_period = series[anomalies[0][i]:anomalies[1][i]]
    mid_date_index = anomaly_period[10:31].idxmin()
    # print type(mid_date_index),mid_date_index
    # mid_date_index - timedelta(days=21)
    anomalies[0][i] = mid_date_index - timedelta(days=21)
    anomalies[1][i] = mid_date_index + timedelta(days=21)
    anomalies[0][i] = datetime.strftime(anomalies[0][i],'%Y-%m-%d')
    anomalies[1][i] = datetime.strftime(anomalies[1][i],'%Y-%m-%d')
  return anomalies




def get_anomalies_new(path,series):
  anomalies = pd.read_csv(path, header=None, index_col=None)
  anomalies[0] = [ datetime.strftime(datetime.strptime(date, '%Y-%m-%d'),'%Y-%m-%d') for date in anomalies[0]]
  anomalies[1] = [ datetime.strftime(datetime.strptime(date, ' %Y-%m-%d'),'%Y-%m-%d') for date in anomalies[1]]
  anomalies = adjust_anomaly_window(anomalies,series)

  return anomalies

def get_qualify_anomalies(path,series):
  anomalies = pd.read_csv(path,header=None, index_col = None)
  anomalies[0] = [ datetime.strftime(datetime.strptime(date, '%Y-%m-%d'),'%Y-%m-%d') for date in anomalies[0]]
  anomalies[1] = [ datetime.strftime(datetime.strptime(date, ' %Y-%m-%d'),'%Y-%m-%d') for date in anomalies[1]]
  labels = []
  for i in range(0,len(anomalies)):
    # print anomalies[2][i]
    labels.append(int(anomalies[2][i]))

  return anomalies, labels

def get_anomalies_year(anomalies):
  mid_date_labels=[]
  for i in range(0,len(anomalies[0])):
    mid_date_labels.append(datetime.strftime(datetime.strptime(anomalies[0][i],'%Y-%m-%d')+timedelta(days=21),'%Y-%m-%d'))
  return mid_date_labels

def newlabels(anomalies,oldlabels):
  # print len(anomalies[anomalies[2] != ' Normal_train']), len(oldlabels)
  labels = []
  k=0
  for i in range(0,len(anomalies)):
    if(anomalies[2][i] != ' Normal_train'):
      labels.append(oldlabels[k])
      #print k,oldlabels[k]
      k = k+1
    else:
      labels.append(8)
  return labels




def prepare(anomalies,labels,priceserieslist):
  x = []
  for i in range(0,len(anomalies)):
    p=[]
    for j in range(0,len(priceserieslist)):
      start = anomalies[0][i]
      end = datetime.strptime(anomalies[0][i],'%Y-%m-%d') + timedelta(days = 29)
      end = datetime.strftime(end,'%Y-%m-%d')
      # p += (Normalise(np.array(priceserieslist[j][anomalies[0][i]:anomalies[1][i]].tolist()))).tolist()
      p += ((np.array(priceserieslist[j][start:anomalies[1][i]].tolist()))).tolist()

      # if(i==0):
      #   print anomalies[0][i], anomalies[1][i]
    x.append(np.array(p))
  return np.array(x),np.array(labels)   


def getKey(item):
  return item[0]

def partition(xseries,yseries,year,months):
  # min_month = datetime.strptime(min(year),'%Y-%m-%d')
  # max_month = datetime.strptime(max(year),'%Y-%m-%d')
  combined_series = zip(year,xseries,yseries)
  combined_series = sorted(combined_series,key=getKey)
  train = []
  train_labels = []
  fixed = datetime.strptime('2006-01-01','%Y-%m-%d')
  i=0
  while(fixed < datetime.strptime('2017-11-01','%Y-%m-%d')):

    currx=[]
    curry=[]
    for anomaly in combined_series:
      i += 1
      # print anomaly[0]
      if(datetime.strptime(anomaly[0],'%Y-%m-%d') > fixed and datetime.strptime(anomaly[0],'%Y-%m-%d')- fixed <= timedelta(days=months*30)):
        currx.append(anomaly[1])
        curry.append(anomaly[2])
        # print "I min if",i
      # else:
    train.append(currx)
    train_labels.append(curry)
    fixed = fixed +timedelta(days = months*30)
    # print fixed
    
  # train.append(currx)
  # train_labels.append(curry)
  return np.array(train),np.array(train_labels)

def get_score(xtrain,xtest,ytrain,ytest):
  scaler = preprocessing.StandardScaler().fit(xtrain)
  xtrain = scaler.transform(xtrain)
  xtest = scaler.transform(xtest)
  # model = RandomForestClassifier(max_depth=2, random_state=0)
  model = SVC(kernel='rbf', C=0.8)
  model.fit(xtrain,ytrain)
  test_pred = np.array(model.predict(xtest))
  # ytest = np.array(ytest)
  # if(test_pred[0] == ytest[0]):
  #   return 1
  # else:
  #   return 0
  return test_pred


def train_test_function(align_m,align_d,align_l,data_m,data_d,data_l):
  train_anomaliesmumbai = get_anomalies_new('data/anomaly/normal_h_w_mumbai.csv',align_m)
  train_anomaliesdelhi = get_anomalies_new('data/anomaly/normal_h_w_delhi.csv',align_d)
  train_anomalieslucknow = get_anomalies_new('data/anomaly/normal_h_w_lucknow.csv',align_l)
  train_delhilabelsnew = newlabels(train_anomaliesdelhi,delhilabels)
  train_lucknowlabelsnew = newlabels(train_anomalieslucknow,lucknowlabels)
  train_mumbailabelsnew = newlabels(train_anomaliesmumbai,mumbailabels)
  test_anomaliesmumbai, test_mumbai_labels = get_qualify_anomalies('data/anomaly/qualify_mumbai.csv',align_m)
  test_anomaliesdelhi, test_delhi_labels = get_qualify_anomalies('data/anomaly/qualify_delhi.csv',align_d)
  test_anomalieslucknow, test_lucknow_labels = get_qualify_anomalies('data/anomaly/qualify_lucknow.csv',align_l)
  
  delhi_anomalies_year = get_anomalies_year(train_anomaliesdelhi)
  mumbai_anomalies_year = get_anomalies_year(train_anomaliesmumbai)
  lucknow_anomalies_year = get_anomalies_year(train_anomalieslucknow)
  test_delhi_year = get_anomalies_year(test_anomaliesdelhi)
  test_mumbai_year = get_anomalies_year(test_anomaliesmumbai)
  test_lucknow_year = get_anomalies_year(test_anomalieslucknow)
  x1_train,y1_train = prepare(train_anomaliesdelhi,train_delhilabelsnew,data_d)
  x2_train,y2_train = prepare(train_anomaliesmumbai,train_mumbailabelsnew,data_m)
  x3_train,y3_train = prepare(train_anomalieslucknow,train_lucknowlabelsnew,data_l)
  xall_train = np.array(x1_train.tolist()+x2_train.tolist()+x3_train.tolist())
  yall_train = np.array(y1_train.tolist()+y2_train.tolist()+y3_train.tolist())
  xall_new =[]
  yall_new = []
  yearall_new = []
  yearall_train = np.array(delhi_anomalies_year+mumbai_anomalies_year+lucknow_anomalies_year)
  # for x in range(0,len(xall)):
  #   print len(xall[x]),yall[x]
  x1_test,y1_test = prepare(test_anomaliesdelhi,test_delhi_labels,data_d)
  x2_test,y2_test = prepare(test_anomaliesmumbai,test_mumbai_labels,data_m)
  x3_test,y3_test = prepare(test_anomalieslucknow,test_lucknow_labels,data_l)
  xall_test = np.array(x1_test.tolist()+x2_test.tolist()+x3_test.tolist())
  yall_test = np.array(y1_test.tolist()+y2_test.tolist()+y3_test.tolist())
  yearall_test = np.array(test_delhi_year+test_mumbai_year+test_lucknow_year)
  


  for y in range(0,len(yall_train)):
    if( yall_train[y] == 2 or yall_train[y]==8 or yall_train[y]==5):
      xall_new.append(xall_train[y])
      yall_new.append(yall_train[y])
      yearall_new.append(yearall_train[y])
    
  # print len(xall_test), len(yall_test)
  
  assert(len(xall_new) == len(yearall_new))
  train_data, train_labels = partition(xall_new,yall_new,yearall_new,6)
  test_data, test_labels = partition(xall_test,yall_test,yearall_test,6)
  # print len(train_data),len(test_data)
  assert(len(train_data) == len(test_data)) 

  temp = 0
  for j in range(0,len(test_data)):
    temp = temp + len(test_data[j])
  # print temp
  predicted = []
  actual_labels = []
  for i in range(0,len(train_data)):
    if(len(test_data[i]) != 0):
      # print i
      test_split = test_data[i]
      test_labels_split = test_labels[i]
      actual_labels = actual_labels + test_labels_split
      train_split = []
      train_labels_split = []
      for j in range(0,len(train_data)):
        if( j != i):
          train_split = train_split + train_data[j]
          train_labels_split = train_labels_split+train_labels[j]
      pred_test = get_score(train_split,test_split,train_labels_split,test_labels_split)  
      predicted = predicted + pred_test.tolist()
  predicted = np.array(predicted)
  print "----------------------opap-----"
  for ind in range(0,len(actual_labels)):
    if(actual_labels[ind] != 2 and actual_labels[ind] != 5):
      actual_labels[ind] = 8
    # print actual_labels[ind]
  # actual_labels= np.array(actual_labels)
  # print len(actual_labels)
  print sum(predicted == actual_labels)/156.0
  # print actual_labels
  # print predicted
  # print f1_score(actual_labels,predicted,labels=[0,1],average="macro")
  # from sklearn.metrics import confusion_matrix
  # print confusion_matrix(actual_labels,predicted)

# train_test_function(retailpriceseriesmumbai,retailpriceseriesdelhi,retailpriceserieslucknow,[retailpriceseriesmumbai],[retailpriceseriesdelhi],[retailpriceserieslucknow])
# train_test_function(retailpriceseriesmumbai,retailpriceseriesdelhi,retailpriceserieslucknow,[mandipriceseriesmumbai],[mandipriceseriesdelhi],[mandipriceserieslucknow])
# train_test_function(retailpriceseriesmumbai,retailpriceseriesdelhi,retailpriceserieslucknow,[retailpriceseriesmumbai,mandipriceseriesmumbai],[retailpriceseriesdelhi,mandipriceseriesdelhi],[retailpriceserieslucknow,mandipriceserieslucknow])
# train_test_function(retailpriceseriesmumbai,retailpriceseriesdelhi,retailpriceserieslucknow,[retailpriceseriesmumbai-mandipriceseriesmumbai,mandiarrivalseriesmumbai],[retailpriceseriesdelhi-mandipriceseriesdelhi,mandiarrivalseriesdelhi],[retailpriceserieslucknow-mandipriceserieslucknow,mandiarrivalserieslucknow])
# train_test_function(retailpriceseriesmumbai,retailpriceseriesdelhi,retailpriceserieslucknow,[retailpriceseriesmumbai-mandipriceseriesmumbai],[retailpriceseriesdelhi-mandipriceseriesdelhi],[retailpriceserieslucknow-mandipriceserieslucknow])
# train_test_function(retailpriceseriesmumbai,retailpriceseriesdelhi,retailpriceserieslucknow,[retailpriceseriesmumbai,mandiarrivalseriesmumbai],[retailpriceseriesdelhi,mandiarrivalseriesdelhi],[retailpriceserieslucknow,mandiarrivalserieslucknow])
# train_test_function(retailpriceseriesmumbai,retailpriceseriesdelhi,retailpriceserieslucknow,[retailpriceseriesmumbai,mandipriceseriesmumbai,mandiarrivalseriesmumbai],[retailpriceseriesdelhi,mandipriceseriesdelhi,mandiarrivalseriesdelhi],[retailpriceserieslucknow,mandipriceserieslucknow,mandiarrivalserieslucknow])
# train_test_function(retailpriceseriesmumbai,retailpriceseriesdelhi,retailpriceserieslucknow,[retailpriceseriesmumbai/mandipriceseriesmumbai],[retailpriceseriesdelhi/mandipriceseriesdelhi],[retailpriceserieslucknow/mandipriceserieslucknow])


# train_test_function(mandipriceseriesmumbai,mandipriceseriesdelhi,mandipriceserieslucknow,[retailpriceseriesmumbai],[retailpriceseriesdelhi],[retailpriceserieslucknow])
# train_test_function(mandipriceseriesmumbai,mandipriceseriesdelhi,mandipriceserieslucknow,[mandipriceseriesmumbai],[mandipriceseriesdelhi],[mandipriceserieslucknow])
# train_test_function(mandipriceseriesmumbai,mandipriceseriesdelhi,mandipriceserieslucknow,[retailpriceseriesmumbai,mandipriceseriesmumbai],[retailpriceseriesdelhi,mandipriceseriesdelhi],[retailpriceserieslucknow,mandipriceserieslucknow])
# train_test_function(mandipriceseriesmumbai,mandipriceseriesdelhi,mandipriceserieslucknow,[retailpriceseriesmumbai-mandipriceseriesmumbai,mandiarrivalseriesmumbai],[retailpriceseriesdelhi-mandipriceseriesdelhi,mandiarrivalseriesdelhi],[retailpriceserieslucknow-mandipriceserieslucknow,mandiarrivalserieslucknow])
# train_test_function(mandipriceseriesmumbai,mandipriceseriesdelhi,mandipriceserieslucknow,[retailpriceseriesmumbai-mandipriceseriesmumbai],[retailpriceseriesdelhi-mandipriceseriesdelhi],[retailpriceserieslucknow-mandipriceserieslucknow])
# train_test_function(mandipriceseriesmumbai,mandipriceseriesdelhi,mandipriceserieslucknow,[retailpriceseriesmumbai,mandiarrivalseriesmumbai],[retailpriceseriesdelhi,mandiarrivalseriesdelhi],[retailpriceserieslucknow,mandiarrivalserieslucknow])
# train_test_function(mandipriceseriesmumbai,mandipriceseriesdelhi,mandipriceserieslucknow,[retailpriceseriesmumbai,mandipriceseriesmumbai,mandiarrivalseriesmumbai],[retailpriceseriesdelhi,mandipriceseriesdelhi,mandiarrivalseriesdelhi],[retailpriceserieslucknow,mandipriceserieslucknow,mandiarrivalserieslucknow])
# train_test_function(mandipriceseriesmumbai,mandipriceseriesdelhi,mandipriceserieslucknow,[retailpriceseriesmumbai/mandipriceseriesmumbai],[retailpriceseriesdelhi/mandipriceseriesdelhi],[retailpriceserieslucknow/mandipriceserieslucknow])


# train_test_function(retailpriceseriesmumbai-mandipriceseriesmumbai,retailpriceseriesdelhi-mandipriceseriesdelhi,retailpriceserieslucknow-mandipriceserieslucknow,[retailpriceseriesmumbai],[retailpriceseriesdelhi],[retailpriceserieslucknow])
# train_test_function(retailpriceseriesmumbai-mandipriceseriesmumbai,retailpriceseriesdelhi-mandipriceseriesdelhi,retailpriceserieslucknow-mandipriceserieslucknow,[mandipriceseriesmumbai],[mandipriceseriesdelhi],[mandipriceserieslucknow])
# train_test_function(retailpriceseriesmumbai-mandipriceseriesmumbai,retailpriceseriesdelhi-mandipriceseriesdelhi,retailpriceserieslucknow-mandipriceserieslucknow,[retailpriceseriesmumbai,mandipriceseriesmumbai],[retailpriceseriesdelhi,mandipriceseriesdelhi],[retailpriceserieslucknow,mandipriceserieslucknow])
# train_test_function(retailpriceseriesmumbai-mandipriceseriesmumbai,retailpriceseriesdelhi-mandipriceseriesdelhi,retailpriceserieslucknow-mandipriceserieslucknow,[retailpriceseriesmumbai-mandipriceseriesmumbai,mandiarrivalseriesmumbai],[retailpriceseriesdelhi-mandipriceseriesdelhi,mandiarrivalseriesdelhi],[retailpriceserieslucknow-mandipriceserieslucknow,mandiarrivalserieslucknow])
# train_test_function(retailpriceseriesmumbai-mandipriceseriesmumbai,retailpriceseriesdelhi-mandipriceseriesdelhi,retailpriceserieslucknow-mandipriceserieslucknow,[retailpriceseriesmumbai-mandipriceseriesmumbai],[retailpriceseriesdelhi-mandipriceseriesdelhi],[retailpriceserieslucknow-mandipriceserieslucknow])
# train_test_function(retailpriceseriesmumbai-mandipriceseriesmumbai,retailpriceseriesdelhi-mandipriceseriesdelhi,retailpriceserieslucknow-mandipriceserieslucknow,[retailpriceseriesmumbai,mandiarrivalseriesmumbai],[retailpriceseriesdelhi,mandiarrivalseriesdelhi],[retailpriceserieslucknow,mandiarrivalserieslucknow])
# train_test_function(retailpriceseriesmumbai-mandipriceseriesmumbai,retailpriceseriesdelhi-mandipriceseriesdelhi,retailpriceserieslucknow-mandipriceserieslucknow,[retailpriceseriesmumbai,mandipriceseriesmumbai,mandiarrivalseriesmumbai],[retailpriceseriesdelhi,mandipriceseriesdelhi,mandiarrivalseriesdelhi],[retailpriceserieslucknow,mandipriceserieslucknow,mandiarrivalserieslucknow])
# train_test_function(retailpriceseriesmumbai-mandipriceseriesmumbai,retailpriceseriesdelhi-mandipriceseriesdelhi,retailpriceserieslucknow-mandipriceserieslucknow,[retailpriceseriesmumbai/mandipriceseriesmumbai],[retailpriceseriesdelhi/mandipriceseriesdelhi],[retailpriceserieslucknow/mandipriceserieslucknow])


# # Change the argmax to idxmin for running the part below

train_test_function(mandiarrivalseriesmumbai,mandiarrivalseriesdelhi,mandiarrivalserieslucknow,[retailpriceseriesmumbai],[retailpriceseriesdelhi],[retailpriceserieslucknow])
train_test_function(mandiarrivalseriesmumbai,mandiarrivalseriesdelhi,mandiarrivalserieslucknow,[mandipriceseriesmumbai],[mandipriceseriesdelhi],[mandipriceserieslucknow])
train_test_function(mandiarrivalseriesmumbai,mandiarrivalseriesdelhi,mandiarrivalserieslucknow,[retailpriceseriesmumbai,mandipriceseriesmumbai],[retailpriceseriesdelhi,mandipriceseriesdelhi],[retailpriceserieslucknow,mandipriceserieslucknow])
train_test_function(mandiarrivalseriesmumbai,mandiarrivalseriesdelhi,mandiarrivalserieslucknow,[retailpriceseriesmumbai-mandipriceseriesmumbai,mandiarrivalseriesmumbai],[retailpriceseriesdelhi-mandipriceseriesdelhi,mandiarrivalseriesdelhi],[retailpriceserieslucknow-mandipriceserieslucknow,mandiarrivalserieslucknow])
train_test_function(mandiarrivalseriesmumbai,mandiarrivalseriesdelhi,mandiarrivalserieslucknow,[retailpriceseriesmumbai-mandipriceseriesmumbai],[retailpriceseriesdelhi-mandipriceseriesdelhi],[retailpriceserieslucknow-mandipriceserieslucknow])
train_test_function(mandiarrivalseriesmumbai,mandiarrivalseriesdelhi,mandiarrivalserieslucknow,[retailpriceseriesmumbai,mandiarrivalseriesmumbai],[retailpriceseriesdelhi,mandiarrivalseriesdelhi],[retailpriceserieslucknow,mandiarrivalserieslucknow])
train_test_function(mandiarrivalseriesmumbai,mandiarrivalseriesdelhi,mandiarrivalserieslucknow,[retailpriceseriesmumbai,mandipriceseriesmumbai,mandiarrivalseriesmumbai],[retailpriceseriesdelhi,mandipriceseriesdelhi,mandiarrivalseriesdelhi],[retailpriceserieslucknow,mandipriceserieslucknow,mandiarrivalserieslucknow])
train_test_function(mandiarrivalseriesmumbai,mandiarrivalseriesdelhi,mandiarrivalserieslucknow,[retailpriceseriesmumbai/mandipriceseriesmumbai],[retailpriceseriesdelhi/mandipriceseriesdelhi],[retailpriceserieslucknow/mandipriceserieslucknow])

