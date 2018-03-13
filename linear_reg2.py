from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from datetime import datetime
# mandipriceseries = mandipriceseries['2013-02-01':'2013-10-24']
# retailpriceseries = retailpriceseries['2013-02-01':'2013-10-24']

font = {'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)

def get_anomalies(path):
	anomalies = pd.read_csv(path, header=None, index_col=None)
	anomalies[0] = [ datetime.strftime(datetime.strptime(date, '%d/%m/%Y'),'%Y-%m-%d') for date in anomalies[0]]
	anomalies[1] = [ datetime.strftime(datetime.strptime(date, ' %d/%m/%Y'),'%Y-%m-%d') for date in anomalies[1]]
	return anomalies

anomaliesmumbai = get_anomalies('data/anomaly/mumbai.csv')
anomaliesdelhi = get_anomalies('data/anomaly/delhi.csv')
anomalieslucknow = get_anomalies('data/anomaly/lucknow.csv')

# delhilabels = [2,4,1,3,1,2,2,2,3,4,1,2,2,1,4,2,5,5,2,2,3,1,5,4,2,5,5,5,3,5,3,5,2,2,5,2,2,5,5,5,2,5,5]
# lucknowlabels = [2,1,1,2,2,2,5,4,3,1,5,5,5,3,2,2,5,5,4,3,4,5,4,2,5,5,5,5,2,2]
# mumbailabels = [2,2,2,3,5,1,2,5,2,5,2,2,2,4,2,3,2,3,3,1,1,2,5,5,3,3,2,5,3,5,5,5,2,5,5,5,2,5,2,5]

delhilabels = [2,4,1,3,1,2,2,2,3,4,1,2,2,1,4,2,5,5,2,2,3,1,5,4,2,5,5,5,3,5,3,5,2,2,5,2,2,5,5,5,2,5,5,5,2,2,2,3,1,5,1,2]
lucknowlabels = [2,1,1,2,2,2,5,4,3,1,5,5,5,3,2,2,5,5,4,3,4,5,4,2,5,5,5,5,2,2,3,2,2,5,3,2,5,2]
mumbailabels = [2,2,2,3,5,1,2,5,2,5,2,2,2,4,2,3,2,3,3,1,1,2,5,5,3,3,2,5,3,5,5,5,2,5,5,5,2,5,2,5,3,2,5,2,5,3,2,1,5,5,2,1,2,2,2,1,5,5,2]

from averagemandi import mandipriceseries
from averagemandi import mandiarrivalseries
from averagemandi import expectedarrivalseries
from averagemandi import expectedmandiprice
from averageretail import retailpriceseries

mp = mandipriceseries
rp = retailpriceseries
idx = mp.index

idx,mp,rp = shuffle(idx,mp,rp)


train_size = (int)(0.80 * len(mp))
mp1 = mp[:train_size]
rp1 = rp[:train_size]
idx1 = idx[:train_size]

mp2 = mp[train_size:]
rp2 = rp[train_size:]
idx2 = idx[train_size:]


regr = linear_model.LinearRegression()
regr.fit(mp1.values.reshape(-1,1),rp1)

rp2_pred = regr.predict(mp2.values.reshape(-1,1))
rp1_pred = regr.predict(mp1.values.reshape(-1,1))

print('R2 Score ',r2_score(rp2, rp2_pred))
print('Mean Squared Error ' , mean_squared_error(rp2, rp2_pred))

# for i in range(0,len(rp2)):
# 	if abs(rp2_pred[i]-rp2.values[i]) > 1000:
# 		plt.scatter(mp2[i], rp2[i],  color='blue')
# 		print idx2[i]
# 	else:
# 		plt.scatter(mp2[i], rp2[i],  color='black')


def ishoarding(idx,anomalies,labels):
	idx  = datetime.strftime(idx,'%Y-%m-%d')
	for i in range(len(anomalies)):
		if anomalies[0][i]<=idx and anomalies[1][i]>=idx:
			if labels[i] == 5:
				return True
	return False

def isweather(idx,anomalies,labels):
	idx  = datetime.strftime(idx,'%Y-%m-%d')
	for i in range(len(anomalies)):
		if anomalies[0][i]<=idx and anomalies[1][i]>=idx:
			if labels[i] == 2:
				return True
	return False


h = False
w = False
n = False

for i in range(0,len(rp2)):
	if ishoarding(idx2[i],anomaliesmumbai,mumbailabels):
		if not h:
			plt.scatter(mp2[i], rp2[i],  color='blue', label='Hoarding')
			h = True
		else:
			plt.scatter(mp2[i], rp2[i],  color='blue')
	elif isweather(idx2[i],anomaliesmumbai,mumbailabels):
		if not w:
			plt.scatter(mp2[i], rp2[i],  color='green', label='Weather')
			w = True
		else:
			plt.scatter(mp2[i], rp2[i],  color='green')
	else:
		if not n:
			plt.scatter(mp2[i], rp2[i],  color='black', label='Normal')
			n = True
		else:
			plt.scatter(mp2[i], rp2[i],  color='black')


plt.plot(mp2, rp2_pred, color='red', linewidth=3, label='Predicted')
plt.xlabel('Mandi Price')
plt.ylabel('Retail Price')
plt.title('Retail vs Mandi Price')
plt.legend(loc='best')
plt.show()
