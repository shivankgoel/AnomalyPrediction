from datetime import datetime
import pandas as pd
import numpy as np
from constants import CONSTANTS
import matplotlib.pyplot as plt
import matplotlib
import sklearn.preprocessing as preprocessing

import os
cwd = os.getcwd()

# centresidx = CONSTANTS['CENTRESIDRITESH']
# from averageretail import dict_centreid_centrename
# for idx in centresidx:
# 	print dict_centreid_centrename[idx]

def get_anomalies(path):
	anomalies = pd.read_csv(path, header=None, index_col=None)
	anomalies[0] = [ datetime.strftime(datetime.strptime(date, '%d/%m/%Y'),'%Y-%m-%d') for date in anomalies[0]]
	anomalies[1] = [ datetime.strftime(datetime.strptime(date, ' %d/%m/%Y'),'%Y-%m-%d') for date in anomalies[1]]
	return anomalies

# anomaliesmumbai = get_anomalies('data/anomaly/mumbai.csv')
# anomaliesdelhi = get_anomalies('data/anomaly/delhi.csv')
# anomalieslucknow = get_anomalies('data/anomaly/lucknow.csv')

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

# from reading import retailP, mandiP, mandiA, retailPM, mandiPM, mandiAM
# retailpriceseriesmumbai = retailP[3]
# retailpriceseriesdelhi = retailP[1]
# retailpriceserieslucknow = retailP[2]

from averageretail import getcenter
retailpriceseriesmumbai = getcenter('MUMBAI')
retailpriceseriesdelhi = getcenter('DELHI')
retailpriceserieslucknow = getcenter('LUCKNOW')
retailpriceseriesbhub = getcenter('BHUBANESHWAR')
retailpriceseriespatna = getcenter('PATNA')
[retailpriceseriesbhub,retailpriceseriesdelhi,retailpriceserieslucknow,retailpriceseriesmumbai,retailpriceseriespatna] = whiten_series_list([retailpriceseriesbhub,retailpriceseriesdelhi,retailpriceserieslucknow,retailpriceseriesmumbai,retailpriceseriespatna])
#[retailpriceseriesdelhi,retailpriceserieslucknow,retailpriceseriesmumbai] = whiten_series_list([retailpriceseriesdelhi,retailpriceserieslucknow,retailpriceseriesmumbai])

from averagemandi import getmandi
mandipriceseriesdelhi = getmandi('Azadpur',True)
mandiarrivalseriesdelhi = getmandi('Azadpur',False)
mandipriceserieslucknow = getmandi('Devariya',True)
mandiarrivalserieslucknow = getmandi('Devariya',False)
from averagemandi import mandipriceseries
from averagemandi import mandiarrivalseries 
[mandipriceseriesdelhi,mandipriceserieslucknow,mandipriceseriesmumbai] = whiten_series_list([mandipriceseriesdelhi,mandipriceserieslucknow,mandipriceseries])
[mandiarrivalseriesdelhi,mandiarrivalserieslucknow,mandiarrivalseriesmumbai] = whiten_series_list([mandiarrivalseriesdelhi,mandiarrivalserieslucknow,mandiarrivalseries])

# START = CONSTANTS['STARTDATE']
# END = CONSTANTS['ENDDATEOLD']
# retailpriceseriesmumbai = retailpriceseriesmumbai[START:END]
# retailpriceseriesdelhi = retailpriceseriesdelhi[START:END]
# retailpriceserieslucknow = retailpriceserieslucknow[START:END]

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

def prepare(anomalies,labels,priceserieslist):
	x = []
	for i in range(0,len(anomalies)):
		p=[]
		for j in range(0,len(priceserieslist)):
			p += (Normalise(np.array(priceserieslist[j][anomalies[0][i]:anomalies[1][i]].tolist()))).tolist()
		x.append(np.array(p))
	return np.array(x),np.array(labels)		

x1,y1 = prepare(anomaliesdelhi,delhilabels,[retailpriceseriesdelhi])
x2,y2 = prepare(anomaliesmumbai,mumbailabels,[retailpriceseriesmumbai])
x3,y3 = prepare(anomalieslucknow,lucknowlabels,[retailpriceserieslucknow])

'''
LEAVE ONE OUT
'''
def get_score(xtrain,xtest,ytrain,ytest):
	model = SVC(kernel='linear', C=0.8)
	model.fit(xtrain,ytrain)
	test_pred = model.predict(xtest)
	if(test_pred[0] == ytest[0]):
		return 1
	else:
		return 0

from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
# xall = x3
# yall = y3
xall = np.array(x1.tolist()+x2.tolist()+x3.tolist())
yall = np.array(y1.tolist()+y2.tolist()+y3.tolist())
score = [0,0,0,0,0]
total = [0,0,0,0,0]
for a,b in loo.split(xall,yall):
	#trainindices = a
	#testindex = b
	xtrain, xtest = xall[a], xall[b]
	ytrain, ytest = yall[a], yall[b]
	score[ytest[0]-1] +=get_score(xtrain,xtest,ytrain,ytest)
	total[ytest[0]-1] +=1

print ['Transport','Weather','Inflation','Fuel','Hoarding']
print 'correct = ', score, 'overall = ', sum(score)
print 'total = ', total, 'total = ', sum(total)
print 'accuracy = ',[score[i]*1.0/total[i] for i in range(0,5)], 'overall = ', sum(score)*1.0/sum(total)  
