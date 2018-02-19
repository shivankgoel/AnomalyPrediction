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






delhilabels = [2,4,1,3,1,2,2,2,3,4,1,2,2,1,4,2,5,5,2,2,3,1,5,4,2,5,5,5,3,5,3,5,2,2,5,2,2,5,5,5,2,5,5]
lucknowlabels = [2,1,1,2,2,2,5,4,3,1,5,5,5,3,2,2,5,5,4,3,4,5,4,2,5,5,5,5,2,2]
mumbailabels = [2,2,2,3,5,1,2,5,2,5,2,2,2,4,2,3,2,3,3,1,1,2,5,5,3,3,2,5,3,5,5,5,2,5,5,5,2,5,2,5]

'''
['BHUBANESHWAR']
['DELHI']
['LUCKNOW']
['MUMBAI']
['PATNA']
'''

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

# from reading_timeseries import retailP, mandiP, mandiA, retailPM, mandiPM, mandiAM
# retailpriceseriesmumbai = retailP[3]
# retailpriceseriesdelhi = retailP[1]
# retailpriceserieslucknow = retailP[2]
# print retailpriceseriesmumbai
from averageretail import getcenter
retailpriceseriesmumbai = getcenter('MUMBAI')
retailpriceseriesdelhi = getcenter('DELHI')
retailpriceserieslucknow = getcenter('LUCKNOW')
retailpriceseriesbhub = getcenter('BHUBANESHWAR')
retailpriceseriespatna = getcenter('PATNA')
# [retailpriceseriesbhub,retailpriceseriesdelhi,retailpriceserieslucknow,retailpriceseriesmumbai,retailpriceseriespatna] = whiten_series_list([retailpriceseriesbhub,retailpriceseriesdelhi,retailpriceserieslucknow,retailpriceseriesmumbai,retailpriceseriespatna])

[retailpriceseriesdelhi,retailpriceserieslucknow,retailpriceseriesmumbai] = whiten_series_list([retailpriceseriesdelhi,retailpriceserieslucknow,retailpriceseriesmumbai])

from averagemandi import getmandi
mandipriceseriesdelhi = getmandi('Azadpur',True)
mandiarrivalseriesdelhi = getmandi('Azadpur',False)
mandipriceserieslucknow = getmandi('Devariya',True)
mandiarrivalserieslucknow = getmandi('Devariya',False)
from averagemandi import mandipriceseries
from averagemandi import mandiarrivalseries 
mandipriceseriesmumbai = mandipriceseries
mandiarrivalseriesmumbai = mandiarrivalseries
[mandipriceseriesdelhi,mandipriceserieslucknow,mandipriceseriesmumbai] = whiten_series_list([mandipriceseriesdelhi,mandipriceserieslucknow,mandipriceseries])
[mandiarrivalseriesdelhi,mandiarrivalserieslucknow,mandiarrivalseriesmumbai] = whiten_series_list([mandiarrivalseriesdelhi,mandiarrivalserieslucknow,mandiarrivalseries])
# mandipriceseriesdelhi = mandiP[3]
# mandipriceserieslucknow = mandiP[4]
# mandipriceseriesmumbai = mandiP[5]
# mandiarrivalseriesdelhi = mandiA[3]
# mandiarrivalserieslucknow = mandiA[4]
# mandiarrivalseriesmumbai = mandiA[5]
# print mandipriceseriesdelhi

def Normalise(arr):
  '''
  Normalise each sample
  '''
  m = arr.mean()
  am = arr.min()
  aM = arr.max()
  arr -= m
  arr /= (aM - am)
  return arr

def adjust_anomaly_window(anomalies,series):
	for i in range(0,len(anomalies)):
		anomaly_period = series[anomalies[0][i]:anomalies[1][i]]
		mid_date_index = anomaly_period[10:31].argmax()
		# print type(mid_date_index),mid_date_index
		# mid_date_index - timedelta(days=21)
		anomalies[0][i] = mid_date_index - timedelta(days=21)
		anomalies[1][i] = mid_date_index + timedelta(days=21)
		anomalies[0][i] = datetime.strftime(anomalies[0][i],'%Y-%m-%d')
		anomalies[1][i] = datetime.strftime(anomalies[1][i],'%Y-%m-%d')
	return anomalies

def get_anomalies(path,series):
	anomalies = pd.read_csv(path, header=None, index_col=None)
	anomalies[0] = [ datetime.strftime(datetime.strptime(date, '%Y-%m-%d'),'%Y-%m-%d') for date in anomalies[0]]
	anomalies[1] = [ datetime.strftime(datetime.strptime(date, ' %Y-%m-%d'),'%Y-%m-%d') for date in anomalies[1]]
	anomalies = adjust_anomaly_window(anomalies,series)
	return anomalies

def get_anomalies_year(anomalies):
	mid_date_labels=[]
	for i in range(0,len(anomalies[0])):
		mid_date_labels.append(datetime.strftime(datetime.strptime(anomalies[0][i],'%Y-%m-%d')+timedelta(days=21),'%Y-%m-%d'))
	return mid_date_labels

anomaliesmumbai = get_anomalies('data/anomaly/normalmumbai.csv',retailpriceseriesmumbai)
anomaliesdelhi = get_anomalies('data/anomaly/normaldelhi.csv',retailpriceseriesdelhi)
anomalieslucknow = get_anomalies('data/anomaly/normallucknow.csv',retailpriceserieslucknow)

delhi_anomalies_year = get_anomalies_year(anomaliesdelhi)
mumbai_anomalies_year = get_anomalies_year(anomaliesmumbai)
lucknow_anomalies_year = get_anomalies_year(anomalieslucknow)

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


def prepare(anomalies,labels,priceserieslist):
	x = []
	for i in range(0,len(anomalies)):

		p=[]
		for j in range(0,len(priceserieslist)):
			p += (Normalise(np.array(priceserieslist[j][anomalies[0][i]:anomalies[1][i]].tolist()))).tolist()
		x.append(np.array(p))
	return np.array(x),np.array(labels)		


# x1,y1 = prepare(anomaliesdelhi,delhilabelsnew,[retailpriceseriesdelhi])
# x2,y2 = prepare(anomaliesmumbai,mumbailabelsnew,[retailpriceseriesmumbai])
# # print x2[0]
# x3,y3 = prepare(anomalieslucknow,lucknowlabelsnew,[retailpriceserieslucknow])

# x1,y1 = prepare(anomaliesdelhi,delhilabelsnew,[mandipriceseriesdelhi])
# x2,y2 = prepare(anomaliesmumbai,mumbailabelsnew,[mandipriceseriesmumbai])
# # print x2[0]
# x3,y3 = prepare(anomalieslucknow,lucknowlabelsnew,[mandipriceserieslucknow])

x1,y1 = prepare(anomaliesdelhi,delhilabelsnew,[retailpriceseriesdelhi/mandipriceseriesdelhi])
x2,y2 = prepare(anomaliesmumbai,mumbailabelsnew,[retailpriceseriesmumbai/mandipriceseriesmumbai])
# print x2[0]
x3,y3 = prepare(anomalieslucknow,lucknowlabelsnew,[retailpriceserieslucknow/mandipriceserieslucknow])

# x1,y1 = prepare(anomaliesdelhi,delhilabelsnew,[retailpriceseriesdelhi-mandipriceseriesdelhi,mandiarrivalseriesdelhi])
# x2,y2 = prepare(anomaliesmumbai,mumbailabelsnew,[retailpriceseriesmumbai-mandipriceseriesmumbai,mandiarrivalseriesmumbai])
# # print x2[0]
# x3,y3 = prepare(anomalieslucknow,lucknowlabelsnew,[retailpriceserieslucknow-mandipriceserieslucknow,mandiarrivalserieslucknow])

# x1,y1 = prepare(anomaliesdelhi,delhilabelsnew,[retailpriceseriesdelhi,mandiarrivalseriesdelhi])
# x2,y2 = prepare(anomaliesmumbai,mumbailabelsnew,[retailpriceseriesmumbai,mandiarrivalseriesmumbai])
# # print x2[0]
# x3,y3 = prepare(anomalieslucknow,lucknowlabelsnew,[retailpriceserieslucknow,mandiarrivalserieslucknow])


def getKey(item):
	return item[0]

def partition(xseries,yseries,year,months):
	# min_month = datetime.strptime(min(year),'%Y-%m-%d')
	# max_month = datetime.strptime(max(year),'%Y-%m-%d')
	combined_series = zip(year,xseries,yseries)
	combined_series = sorted(combined_series,key=getKey)
	train = []
	train_labels = []
	fixed = combined_series[0]
	currx=[]
	curry=[]
	i=0
	for anomaly in combined_series:
		i += 1
		# print anomaly[0]
		if(datetime.strptime(anomaly[0],'%Y-%m-%d')-datetime.strptime(fixed[0],'%Y-%m-%d') <= timedelta(days=months*30)):
			currx.append(anomaly[1])
			curry.append(anomaly[2])
			# print "I min if",i
		else:
			train.append(currx)
			train_labels.append(curry)
			currx=[anomaly[1]]
			curry=[anomaly[2]]
			fixed = anomaly
	train.append(currx)
	train_labels.append(curry)
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
	# 	return 1
	# else:
	# 	return 0
	return test_pred



xall = np.array(x1.tolist()+x2.tolist()+x3.tolist())
yall = np.array(y1.tolist()+y2.tolist()+y3.tolist())
xall_new =[]
yall_new = []

yearall = np.array(delhi_anomalies_year+mumbai_anomalies_year+lucknow_anomalies_year)
assert(len(xall) == len(yearall))
total_data, total_labels = partition(xall_new,yall_new,yearall,6)
predicted = []
actual_labels = []
for i in range(0,len(total_data)):
	test_split = total_data[i]
	test_labels = total_labels[i]
	actual_labels = actual_labels + test_labels
	train_split = []
	train_labels_split = []
	for j in range(0,len(total_data)):
		if( j != i):
			train_split = train_split + total_data[j]
			train_labels_split = train_labels_split+total_labels[j]
	pred_test = get_score(train_split,test_split,train_labels_split,test_labels)	
	predicted = predicted + pred_test.tolist()
predicted = np.array(predicted)
actual_labels= np.array(actual_labels)
print len(actual_labels)
print sum(predicted == actual_labels)
print actual_labels
print predicted
print f1_score(actual_labels,predicted,labels=[1,2,3,4,5,6],average="macro")


from sklearn.metrics import confusion_matrix
cfmatrix1 = confusion_matrix(actual_labels,predicted)
print ['Transport','Weather','Inflation','Fuel','Hoarding','Normal']
print cfmatrix1


'''
LEAVE ONE OUT
'''

# # from sklearn.model_selection import LeaveOneOut
# # loo = LeaveOneOut()
# # xall = x3
# # yall = y3
# score = [0,0,0,0,0]
# total = [0,0,0,0,0]
# for y in range(2006,2015):
# 	#trainindices = a
# 	#testindex = b

# 	xtrain, xtest = partition(xall,yearall,y)
# 	ytrain, ytest = partition(yall,yearall,y)
# 	assert(len(xtrain) == len(ytrain))
# 	assert(len(xtest) == len(ytest))
# 	for t in range(0,len(ytest)):
# 		print len(xtest[t])
# 		print ytest[t], get_score(np.array(xtrain),np.array(xtest[t]),np.array(ytrain),np.array([ytest[t]]))
# 		score[ytest[t]-1] +=get_score(xtrain,[xtest[t]],ytrain,[ytest[t]])
# 		total[ytest[t]-1] +=1

# print ['Transport','Weather','Inflation','Fuel','Hoarding']
# print 'correct = ', score, 'overall = ', sum(score)
# print 'total = ', total, 'total = ', sum(total)
# print 'accuracy = ',[score[i]*1.0/total[i] for i in range(0,5)], 'overall = ', sum(score)*1.0/sum(total)  
