from datetime import datetime
import pandas as pd
import numpy as np
from constants import CONSTANTS
import matplotlib.pyplot as plt
import matplotlib


def whiten(series):
  '''
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

def get_anomalies(path):
	anomalies = pd.read_csv(path, header=None, index_col=None)
	anomalies[0] = [ datetime.strftime(datetime.strptime(date, '%d/%m/%Y'),'%Y-%m-%d') for date in anomalies[0]]
	anomalies[1] = [ datetime.strftime(datetime.strptime(date, ' %d/%m/%Y'),'%Y-%m-%d') for date in anomalies[1]]
	return anomalies

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

anomaliesmumbai = get_anomalies('data/anomaly/mumbai.csv')
anomaliesdelhi = get_anomalies('data/anomaly/delhi.csv')
anomalieslucknow = get_anomalies('data/anomaly/lucknow.csv')

delhilabels = [2,4,1,3,1,2,2,2,3,4,1,2,2,1,4,2,5,5,2,2,3,1,5,4,2,5,5,5,3,5,3,5,2,2,5,2,2,5,5,5,2,5,5]
lucknowlabels = [2,1,1,2,2,2,5,4,3,1,5,5,5,3,2,2,5,5,4,3,4,5,4,2,5,5,5,5,2,2]
mumbailabels = [2,2,2,3,5,1,2,5,2,5,2,2,2,4,2,3,2,3,3,1,1,2,5,5,3,3,2,5,3,5,5,5,2,5,5,5,2,5,2,5]

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


def preparexdf(retailpriceseries,mandipriceseries,mandiarrivalseries):
	d = {'retailp':retailpriceseries, 
	'mandip':mandipriceseries,
	'mandiarr':mandiarrivalseries,
	}
	xdf = pd.DataFrame(data=d)
	for i in range(1,30):
		xdf['retailp'+str(i)] = retailpriceseries.shift(i).interpolate(method='pchip',limit_direction='both')
		xdf['mandip'+str(i)] = mandipriceseries.shift(i).interpolate(method='pchip',limit_direction='both')
		xdf['mandiarr'+str(i)] = mandiarrivalseries.shift(i).interpolate(method='pchip',limit_direction='both')
	return xdf

xdf1 = preparexdf(retailpriceseriesdelhi,mandipriceseriesdelhi,mandiarrivalseriesdelhi)
xdf2 = preparexdf(retailpriceseriesmumbai,mandipriceseriesmumbai,mandiarrivalseriesmumbai)
xdf3 = preparexdf(retailpriceserieslucknow,mandipriceserieslucknow,mandiarrivalserieslucknow)

num2label = {
	1:'Transport',
	2:'Weather',
	3:'Inflation',
	4:'Fuel',
	5:'Hoarding',
}

def give_anomaly_labels(category,anomalies,labels):
	seriesanomaly = pd.Series(index=mandipriceseriesmumbai.index)
	seriesanomaly.fillna(0, inplace=True)
	for anomaly_num in range(0,len(anomalies)):
		startdate = anomalies.iloc[anomaly_num][0]
		enddate = anomalies.iloc[anomaly_num][1]
		label = labels[anomaly_num]
		if(category == ''):
			seriesanomaly[startdate:enddate] = 1
		elif(category == num2label[label]):
			seriesanomaly[startdate:enddate] = 1
	return seriesanomaly

def create_labels(anomalies,labels):
	alllabels = give_anomaly_labels('',anomalies,labels) #1517 out of 3461
	hoardinglabels = give_anomaly_labels('Hoarding',anomalies,labels)
	weatherlabels = give_anomaly_labels('Weather',anomalies,labels)
	transportlabels = give_anomaly_labels('Transport',anomalies,labels)
	inflationlabels = give_anomaly_labels('Inflation',anomalies,labels)
	fuellabels = give_anomaly_labels('Fuel',anomalies,labels)
	return alllabels,hoardinglabels,weatherlabels,transportlabels,inflationlabels,fuellabels


a1,h1,w1,t1,i1,f1 = create_labels(anomaliesdelhi,delhilabels)
a2,h2,w2,t2,i2,f2 = create_labels(anomaliesmumbai,mumbailabels)
a3,h3,w3,t3,i3,f3 = create_labels(anomalieslucknow,lucknowlabels) 

from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

trainstart = '2006-01-01'
trainend='2012-06-01'

def givetrain(list1):
	return (list1[trainstart:trainend]).values

def givetest(list1):
	return (list1[trainend:]).values


from sklearn.preprocessing import MinMaxScaler
sc = preprocessing.MinMaxScaler()

def createtraintest(x1,x2,x3,y1,y2,y3,preprocess):
	train = np.concatenate((givetrain(x1),np.concatenate((givetrain(x2),givetrain(x3)))))
	train_labels = np.concatenate((givetrain(y1),np.concatenate((givetrain(y2),givetrain(y3)))))
	test = np.concatenate((givetest(x1),np.concatenate((givetest(x2),givetest(x3)))))
	test_labels = np.concatenate((givetest(y1),np.concatenate((givetest(y2),givetest(y3)))))
	if(preprocess):
		train = sc.fit_transform(train)
		test = sc.transform(test)
	return train,train_labels,test[30:],test_labels[30:]


train,train_labels,test,test_labels = createtraintest(xdf1,xdf2,xdf3,h1,h2,h3,True)
train,train_labels = shuffle(train,train_labels,random_state=0)
test,test_labels = shuffle(test,test_labels,random_state=0)

# from sklearn.decomposition import PCA
# pca = PCA(n_components=8, whiten=True)
# pca.fit(train)
# train = pca.transform(train)
# test = pca.transform(test)

#model = RandomForestClassifier()
model = SVC(C=1)
model.fit(train,train_labels)
test_pred = model.predict(test)
train_pred = model.predict(train)

# import keras
# from keras.models import Sequential
# from keras.layers import Dense

# model = Sequential()
# model.add(Dense(output_dim=45,input_dim=90,init='uniform',activation='relu'))
# model.add(Dense(output_dim=20,init='uniform',activation='relu'))
# model.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
# model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# model.fit(train,train_labels,batch_size=10,num_epoch=100)
# test_pred = model.predict(test)
# test_pred = test_pred > 0.5
# train_pred = model.predict(train)
# train_pred = train_pred > 0.5

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



def createtest(xdf,labels,anomalies,idx,preprocess):
	start = anomalies.iloc[idx,:][0]
	end = anomalies.iloc[idx,:][1]
	label = labels[idx]
	x = xdf[start:end]
	if(preprocess):
		x = sc.transform(x)
	return x,label


a,b = createtest(xdf1,delhilabels,anomaliesdelhi,-3,True)



