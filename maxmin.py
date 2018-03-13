from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from datetime import datetime
import math

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

delhilabels = [2,4,1,3,1,2,2,2,3,4,1,2,2,1,4,2,5,5,2,2,3,1,5,4,2,5,5,5,3,5,3,5,2,2,5,2,2,5,5,5,2,5,5,5,2,2,2,3,1,5,1,2]
lucknowlabels = [2,1,1,2,2,2,5,4,3,1,5,5,5,3,2,2,5,5,4,3,4,5,4,2,5,5,5,5,2,2,3,2,2,5,3,2,5,2]
mumbailabels = [2,2,2,3,5,1,2,5,2,5,2,2,2,4,2,3,2,3,3,1,1,2,5,5,3,3,2,5,3,5,5,5,2,5,5,5,2,5,2,5,3,2,5,2,5,3,2,1,5,5,2,1,2,2,2,1,5,5,2]


def getavgmaxmin2(series,anomalies,labels,lblnum):
	total = 0
	count = 0
	for i in range(len(anomalies)):
		if labels[i] == lblnum:
			a = anomalies[0][i]
			b = anomalies[1][i]
			newseries = series[a:b]
			maxi = newseries.max()
			mini = newseries.min()
			if(math.isnan(maxi) or math.isnan(mini)):
				print newseries
			#print i,maxi,mini
			total = total + maxi - mini
			count = count + 1
	return total,count


def getavgmaxmin(lblnum):
	from averageretail import getcenter
	delhi = getcenter('DELHI')
	mumbai = getcenter('MUMBAI')
	lucknow = getcenter('LUCKNOW')
	t1,c1 = getavgmaxmin2(delhi,anomaliesdelhi,delhilabels,lblnum)
	t2,c2 = getavgmaxmin2(mumbai,anomaliesmumbai,mumbailabels,lblnum)
	t3,c3 = getavgmaxmin2(lucknow,anomalieslucknow,lucknowlabels,lblnum)
	return (t1+t2+t3)/(c1+c2+c3)

'''
Weather 2,Horading 5, Inflation 3
Hoarding 1607
Weather 1005
Inflation 491
'''
