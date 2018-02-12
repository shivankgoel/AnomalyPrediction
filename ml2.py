from datetime import datetime
import pandas as pd
import numpy as np
from constants import CONSTANTS
import matplotlib.pyplot as plt
import matplotlib

import os
cwd = os.getcwd()
anomalies = pd.read_csv(CONSTANTS['ANOMALIES_NEWSPAPER'], header=None, index_col=None)
anomalies[0] = [ datetime.strftime(datetime.strptime(date, '%d/%m/%Y'),'%Y-%m-%d') for date in anomalies[0]]
anomalies[1] = [ datetime.strftime(datetime.strptime(date, ' %d/%m/%Y'),'%Y-%m-%d') for date in anomalies[1]]


from averagemandi import mandipriceseries
from averagemandi import mandiarrivalseries
from averageretail import retailpriceseries
from average_export import exportseries
from rainfallmonthly import rainfallmonthly
from fuelprice import fuelpricemumbai
from cpimonthlyseries import cpimonthlyseries
from oilmonthlyseries import oilmonthlyseries

d = {'retailp':retailpriceseries , 
	'mandip':mandipriceseries ,
	'mandiarr':mandiarrivalseries ,
	'export' : exportseries,
	'rainfall': rainfallmonthly,
	'fuel': fuelpricemumbai,
	'cpi': cpimonthlyseries,
	'oil' : oilmonthlyseries
	}

xdf = pd.DataFrame(data=d)
for i in range(1,30):
	xdf['retailp'+str(i)] = retailpriceseries.shift(i)
	xdf['mandip'+str(i)] = mandipriceseries.shift(i)
	xdf['mandiarr'+str(i)] = mandiarrivalseries.shift(i)
	xdf['export'+str(i)] = exportseries.shift(i)
	xdf['rainfall'+str(i)] = rainfallmonthly.shift(i)
	xdf['fuel'+str(i)] = fuelpricemumbai.shift(i)
	xdf['cpi'+str(i)] = cpimonthlyseries.shift(i)
	xdf['oil'+str(i)] = oilmonthlyseries.shift(i)
	# xdf['retailp'+str(-i)] = retailpriceseries.shift(-i)
	# xdf['mandip'+str(-i)] = mandipriceseries.shift(-i)
	# xdf['mandiarr'+str(-i)] = mandiarrivalseries.shift(-i)
	# xdf['export'+str(-i)] = exportseries.shift(-i)
	# xdf['rainfall'+str(-i)] = rainfallmonthly.shift(-i)
	# xdf['fuel'+str(-i)] = fuelpricemumbai.shift(-i)
	# xdf['cpi'+str(-i)] = cpimonthlyseries.shift(-i)
	# xdf['oil'+str(-i)] = oilmonthlyseries.shift(-i)

def give_anomaly_labels(category):
	seriesanomaly = pd.Series(index=mandipriceseries.index)
	seriesanomaly.fillna(0, inplace=True)
	for anomaly_num in range(0,30):
		startdate = anomalies.iloc[anomaly_num][0]
		enddate = anomalies.iloc[anomaly_num][1]
		labels = anomalies.iloc[anomaly_num][2]
		labels = labels.strip(' ').split(' ')
		if(category == ''):
			seriesanomaly[startdate:enddate] = 1
		elif(category in labels):
			seriesanomaly[startdate:enddate] = 1
	return seriesanomaly


alllabels = give_anomaly_labels('') #1234 out of 3461
hoardinglabels = give_anomaly_labels('Hoarding')
weatherlabels = give_anomaly_labels('Weather')
transportlabels = give_anomaly_labels('Transport')
inflationlabels = give_anomaly_labels('Inflation')
fuellabels = give_anomaly_labels('Fuel')


from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier


STARTDATE = '2006-02-01'
ENDDATE = '2015-06-23'
xdf = xdf[STARTDATE:ENDDATE]
trainsize = len(xdf[STARTDATE:'2010-12-31'])
xdf = preprocessing.scale(xdf)
ydf = alllabels
ydf = ydf[STARTDATE:ENDDATE]

train = xdf[:trainsize]
train_labels = ydf[:trainsize]
test = xdf[trainsize:]
test_labels = ydf[trainsize:]
train,train_labels = shuffle(train,train_labels,random_state=0)
test,test_labels = shuffle(test,test_labels,random_state=0)

from sklearn.decomposition import PCA
pca = PCA(n_components=8, whiten=True)
pca.fit(train)
train = pca.transform(train)
test = pca.transform(test)


#model = RandomForestClassifier(max_depth=None, random_state=0)
# from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
#from sklearn.ensemble import RandomForestClassifier, VotingClassifier
# clf1 = LogisticRegression(random_state=0)
# clf2 = RandomForestClassifier(random_state=0)
#model = GaussianNB()
# clf4 = SVC()
#model = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3),('svc', clf4)], voting='hard')
model = SVC()
from sklearn import svm, grid_search
# parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
# model = svm.SVC()
# model = grid_search.GridSearchCV(svr, parameters)
model.fit(train,train_labels)
test_pred = model.predict(test)
train_pred = model.predict(train)

# from sklearn.metrics import average_precision_score
# average_precision = average_precision_score(test_labels, test_pred)
# print average_precision

# from sklearn.metrics import confusion_matrix
# cfmatrix = confusion_matrix(test_labels,test_pred)
# print cfmatrix

def precision_recall(predicted_labels, actual_labels):
	test_pred = predicted_labels
	test_labels = actual_labels
	truepositive= np.sum(test_pred[test_labels == 1])
	falsenegative= len(test_pred[test_labels == 1]) - truepositive
	falsepositive= np.sum(test_pred[test_labels == 0])
	truenegative= len(test_pred[test_labels == 0]) - falsepositive

	precision = truepositive / (truepositive+falsepositive)
	recall = (truepositive*1.0) / (truepositive + falsenegative)
	fscore = (2* precision * recall)/(precision+recall)
	print 'Precision',precision
	print 'Recall',recall
	print 'FScore',fscore


print 'Train Set Results'
print(model.score(train,train_labels))
precision_recall(train_pred,train_labels)

print '\nTest Set Results'
print(model.score(test,test_labels))
precision_recall(test_pred,test_labels)
