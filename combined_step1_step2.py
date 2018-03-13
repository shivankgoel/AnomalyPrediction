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
from sklearn import linear_model



import os
cwd = os.getcwd()

'''
This file checks the accuracies for only four classes of periods - Hoarding, Weather, Inflation and Normal
'''




delhilabels = [2,4,1,3,1,2,2,2,3,4,1,2,2,1,4,2,5,5,2,2,3,1,5,4,2,5,5,5,3,5,3,5,2,2,5,2,2,5,5,5,2,5,5,5,2,2,2,3,1,5,1,2]
lucknowlabels = [2,1,1,2,2,2,5,4,3,1,5,5,5,3,2,2,5,5,4,3,4,5,4,2,5,5,5,5,2,2,3,2,2,5,3,2,5,2]
mumbailabels = [2,2,2,3,5,1,2,5,2,5,2,2,2,4,2,3,2,3,3,1,1,2,5,5,3,3,2,5,3,5,5,5,2,5,5,5,2,5,2,5,3,2,5,2,5,3,2,1,5,5,2,1,2,2,2,1,5,5,2]

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

#[retailpriceseriesdelhi,retailpriceserieslucknow,retailpriceseriesmumbai] = whiten_series_list([retailpriceseriesdelhi,retailpriceserieslucknow,retailpriceseriesmumbai])

from averagemandi import getmandi
mandipriceseriesdelhi = getmandi('Azadpur',True)
mandiarrivalseriesdelhi = getmandi('Azadpur',False)
mandipriceserieslucknow = getmandi('Bahraich',True)
mandiarrivalserieslucknow = getmandi('Bahraich',False)
from averagemandi import mandipriceseries
from averagemandi import mandiarrivalseries 
mandipriceseriesmumbai = mandipriceseries
mandiarrivalseriesmumbai = mandiarrivalseries
# [mandipriceseriesdelhi,mandipriceserieslucknow,mandipriceseriesmumbai] = whiten_series_list([mandipriceseriesdelhi,mandipriceserieslucknow,mandipriceseries])
# [mandiarrivalseriesdelhi,mandiarrivalserieslucknow,mandiarrivalseriesmumbai] = whiten_series_list([mandiarrivalseriesdelhi,mandiarrivalserieslucknow,mandiarrivalseries])
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



# def newlabels(anomalies,oldlabels):
# 	# print len(anomalies[anomalies[2] != ' Normal']), len(oldlabels)
# 	labels = []
# 	k=0
# 	for i in range(0,len(anomalies)):
# 		if(anomalies[2][i] == ' Normal'):
# 			labels.append(7)
# 		elif(anomalies[2][i] == ' NormalR'):
# 			labels.append(6)
# 		else:
# 			labels.append(oldlabels[k])
# # print k,oldlabels[k]
# 			k = k+1
# 	return labels


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
			#p += (Normalise(np.array(priceserieslist[j][anomalies[0][i]:anomalies[1][i]].tolist()))).tolist()
			p += ((np.array(priceserieslist[j][anomalies[0][i]:anomalies[1][i]].tolist()))).tolist()
		x.append(np.array(p))
	return np.array(x),np.array(labels)

DAYS = 21


def cost_function(yaxis,start,end,days):
	xaxis = []
	for i in range(1,days+1):
		xaxis.append([i])
	yaxis = yaxis[0:days]
	xaxis = np.array(xaxis)
	# print len(xaxis), len(yaxis)
	regr = linear_model.LinearRegression()
	regr.fit(xaxis,yaxis)

	return regr.coef_[0]

def cost_function2(yaxis,days):
	yaxis = yaxis[0:days]
	a = yaxis.min()
	b = yaxis.max()
	# if( b- a > 10000):
	# 	print yaxis
	return b-a

def cost_function3(yaxis,start,end,series,days):
	avg = give_average_series(start,end,series)
	avg = np.array(avg)
	deviation = yaxis - avg
	# print len(avg), len(yaxis), len(deviation)
	# print type(avg), type(yaxis), type(deviation)
	# print avg
	# print yaxis
	# print deviation
	deviation = deviation[0:days]
	return deviation.mean()

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
			if(datetime.strptime(anomaly[0],'%Y-%m-%d') > fixed and datetime.strptime(anomaly[0],'%Y-%m-%d')- fixed <= timedelta(days=months*30)):
				currx.append(anomaly[1])
				curry.append(anomaly[2])
		train.append(currx)
		train_labels.append(curry)
		fixed = fixed +timedelta(days = months*30)
	
	return np.array(train),np.array(train_labels)

def partition2(xseries,yseries,year,months,anomaly_info,anomaly_loc):
	# min_month = datetime.strptime(min(year),'%Y-%m-%d')
	# max_month = datetime.strptime(max(year),'%Y-%m-%d')
	combined_series = zip(year,xseries,yseries,anomaly_info,anomaly_loc)
	combined_series = sorted(combined_series,key=getKey)
	train = []
	train_labels = []
	train_info = []
	train_loc = []
	fixed = datetime.strptime('2006-01-01','%Y-%m-%d')
	i=0
	while(fixed < datetime.strptime('2017-11-01','%Y-%m-%d')):
		currx=[]
		curry=[]
		currz = []
		currw = []
		for anomaly in combined_series:
			i += 1
			if(datetime.strptime(anomaly[0],'%Y-%m-%d') > fixed and datetime.strptime(anomaly[0],'%Y-%m-%d')- fixed <= timedelta(days=months*30)):
				currx.append(anomaly[1])
				curry.append(anomaly[2])
				currz.append(anomaly[3])
				currw.append(anomaly[4])
		train.append(currx)
		train_labels.append(curry)
		train_info.append(currz)
		train_loc.append(currw)
		fixed = fixed +timedelta(days = months*30)
	
	return np.array(train),np.array(train_labels),np.array(train_info), np.array(train_loc)

def generate_test():
	from datetime import timedelta
	periods = pd.DataFrame(columns = [0,1])
	startdate = datetime.strptime(CONSTANTS['STARTDATE'],'%Y-%m-%d')
	enddate = datetime.strptime(CONSTANTS['ENDDATE'],'%Y-%m-%d')
	date = startdate
	window = 42
	duration = timedelta(days = window)
	i=0
	startdates = []
	enddates = []
	while(duration < enddate - date):
		s = datetime.strftime(date,'%Y-%m-%d')
		e = datetime.strftime(date+duration,'%Y-%m-%d')
		date = date + timedelta(days=1)
		startdates.append(s)
		enddates.append(e)
	periods[0] = startdates
	periods[1] = enddates
	return periods



def overlapping(anomalies,s,e,labels):
  ans = False
  covered = []
  for i in range(0,len(anomalies)):
    if(labels[i] == 2 or labels[i] == 5 or labels[i] == 3):  
      if((anomalies[0][i]<=s and s<=anomalies[1][i]) or  (anomalies[0][i]<=e and e<=anomalies[1][i])):
        ans = True
        covered.append(i)

  return ans,covered

def calculate_actual(anomalies,s,e,labels):
  lbl = []
  durn = []
  for i in range(0,len(anomalies)):
    if(labels[i] == 2 or labels[i] == 5 or labels[i] == 3):  
      if(anomalies[0][i]<=s and s<=anomalies[1][i]):
      	endtime = datetime.strptime(anomalies[1][i],'%Y-%m-%d')
      	starttime = datetime.strptime(s,'%Y-%m-%d')
      	diff  = endtime - starttime
      	lbl.append(labels[i])
      	durn.append(diff)
      elif(anomalies[0][i]<=e and e<=anomalies[1][i]):
       	endtime = datetime.strptime(e,'%Y-%m-%d')
      	starttime = datetime.strptime(anomalies[0][i],'%Y-%m-%d')
      	diff  = endtime - starttime
      	lbl.append(labels[i])
      	durn.append(diff)
  if len(lbl) == 0:
  	return -1
  c = zip(durn,lbl)
  c.sort(reverse=True)
  return c[0][1]


def overlapping_step2(anomalies,s,e,labels,pred_label):
	ans = False
	covered = []
	for i in range(0,len(anomalies)):
		if(labels[i] == pred_label):  
			if((anomalies[0][i]<=s and s<=anomalies[1][i]) or  (anomalies[0][i]<=e and e<=anomalies[1][i])):
				ans = True
				covered.append(i)
	return ans,covered



def first_step(align_m,align_d,align_l,data_m,data_d,data_l,anomaliesmumbai,anomaliesdelhi,anomalieslucknow, mumbailabelsnew,delhilabelsnew,lucknowlabelsnew,data_m2,data_d2,data_l2):
	
	test_anomalies_delhi = generate_test()
	test_anomalies_mumbai = generate_test()
	test_anomalies_lucknow = generate_test()
	test_anomaliesall = pd.concat([test_anomalies_delhi,test_anomalies_mumbai,test_anomalies_lucknow],ignore_index = True)


	test_delhi_anomalies_year = get_anomalies_year(test_anomalies_delhi)
	test_mumbai_anomalies_year = get_anomalies_year(test_anomalies_mumbai)
	test_lucknow_anomalies_year = get_anomalies_year(test_anomalies_lucknow)
	test_anomalies_year = np.array(test_delhi_anomalies_year+test_mumbai_anomalies_year+test_lucknow_anomalies_year)


	x1_test,y1_test = prepare(test_anomalies_delhi,[],data_d)
	x2_test,y2_test = prepare(test_anomalies_mumbai,[],data_m)
	x3_test,y3_test = prepare(test_anomalies_lucknow,[],data_l)

	xall_test = np.array(x1_test.tolist()+x2_test.tolist()+x3_test.tolist())
	# yall_test = np.array(y1_test.tolist()+y2_test.tolist()+y3_test.tolist())
	
	print len(x1_test),len(x2_test)

	x1_test2,y1_test2 = prepare(test_anomalies_delhi,[],data_d2)
	x2_test2,y2_test2 = prepare(test_anomalies_mumbai,[],data_m2)
	x3_test2,y3_test2 = prepare(test_anomalies_lucknow,[],data_l2)

	
	# normal_points_index = []
	# normal_points_value = []
	# h_w_points_index = []
	# h_w_points_value = []
	# f_i_t_points_index = []
	# f_i_t_points_value = []
	

	# slope = []
	
	
	# for i in range(0,len(yall)):
	# 	# print i
	# 	# if( i < len(y1)):
	# 	# 	parameter = cost_function3(xall[i],anomaliesall[0][i],anomaliesall[1][i],retailpriceseriesdelhi,DAYS)
	# 	# elif( i < (len(y1)+len(y2)) ):
	# 	# 	parameter = cost_function3(xall[i],anomaliesall[0][i],anomaliesall[1][i],retailpriceseriesmumbai,DAYS)
	# 	# else:
	# 	# 	parameter = cost_function3(xall[i],anomaliesall[0][i],anomaliesall[1][i],retailpriceserieslucknow,DAYS)

	# 	parameter = cost_function(xall[i],anomaliesall[0][i],anomaliesall[1][i],DAYS)
	# 	# parameter = cost_function2(xall[i],anomaliesall[0][i],anomaliesall[1][i],DAYS)
	# 	# if( parameter == 0):
	# 	# 	if(yall[i] == 5):
	# 	# 		print xall[i][0:DAYS]
	# 	print i,"  ",anomaliesall[0][i],"------> ",parameter
	# 	slope.append((parameter,yall[i]))
	# 	if(yall[i] == 2 or yall[i] == 5 ):
	# 		h_w_points_index.append(i)
	# 		h_w_points_value.append(parameter)
	# 	elif(yall[i] == 8):
	# 		normal_points_index.append(i)
	# 		normal_points_value.append(parameter)
	# 	elif(yall[i] == 3):
	# 		f_i_t_points_index.append(i)
	# 		f_i_t_points_value.append(parameter)
		
	# plt.scatter(h_w_points_index,h_w_points_value,color = 'red', label = "H / W")
	# plt.scatter(normal_points_index,normal_points_value,color = 'green', label = "Normal")
	# plt.scatter(f_i_t_points_index,f_i_t_points_value,color = 'black', label = "F / I / T")
	
	
	# slope.sort(reverse = True)
	# plt.tick_params()
	# plt.xlabel('Anomaly',fontsize = 20)
 #  	# plt.ylabel('Parameter (Mean Deviation of Retail Price from average R.P(over all years))',fontsize = 20)
 #  	# plt.ylabel('Parameter (Slope of Retail Price during the window)',fontsize = 20)
 #  	# plt.ylabel('Parameter (Max - Min for Retail Price during the window)',fontsize = 20)
 #  	# plt.ylabel('Parameter (Mean Deviation of Mandi Price from average M.P(over all years))',fontsize = 20)
 #  	plt.ylabel('Parameter (Slope of Mandi Price during the window)',fontsize = 20)
 #  	# plt.ylabel('Parameter (Max - Min for Mandi Price during the window)',fontsize = 20)
 #  	# plt.ylabel('Parameter (Slope of Retail - Mandi Price during the window)',fontsize = 20)
 #  	# plt.ylabel('Parameter (Max - Min of Retail - Mandi Price during the window)',fontsize = 20)
 #  	# plt.ylabel('Parameter (Mean Deviation of Retail - Mandi Price during the window)',fontsize = 20)


 #  	plt.legend(loc = 'best')
	# plt.show()
	
	xall_test_new = []
	yall_test_new = []				#this is a dummy variable, it contains garbage information
	test_anomalies_year_new = []
	passed_step1_correctly = []
	anomalies_passed = []
	label_locations = []

	threshold = 2
	overlapped = 0
	passed=0
	i=0
	overlapped_delhi = []
	while(i < len(x1_test)):
		parameter = cost_function(x1_test[i],test_anomalies_delhi[0][i],test_anomalies_delhi[1][i],DAYS)
			
		if(parameter > threshold):
			passed += 1
			xall_test_new.append(x1_test2[i])
			label_locations.append(0)
			# yall_test_new.append(yall_test[i])
			anomalies_passed.append((test_anomalies_delhi[0][i],test_anomalies_delhi[1][i]))
			test_anomalies_year_new.append(test_delhi_anomalies_year[i])
			ans, covered_delhi = overlapping(anomaliesdelhi,test_anomalies_delhi[0][i],test_anomalies_delhi[1][i],delhilabelsnew)
			overlapped_delhi = overlapped_delhi + covered_delhi 
			actual = calculate_actual(anomaliesdelhi,test_anomalies_delhi[0][i],test_anomalies_delhi[1][i],delhilabelsnew)
			yall_test_new.append(actual)
			if( ans ):
				overlapped += 1
				passed_step1_correctly.append(1)
			else:
				passed_step1_correctly.append(0)
			i += 2*DAYS
		i += 1 
	print "Anomalies covered in Delhi -", (np.unique(np.array(overlapped_delhi)))


	overlapped_mumbai = []
	i=0
	while ( i < len(x2_test)):
		parameter = cost_function(x2_test[i],test_anomalies_mumbai[0][i],test_anomalies_mumbai[1][i],DAYS)
			
		if(parameter > threshold):
			passed += 1
			xall_test_new.append(x2_test2[i])
			# yall_test_new.append(-1)
			label_locations.append(1)
			anomalies_passed.append((test_anomalies_mumbai[0][i],test_anomalies_mumbai[1][i]))

			# yall_test_new.append(yall_test[i])
			test_anomalies_year_new.append(test_mumbai_anomalies_year[i])
			ans, covered_mumbai =  overlapping(anomaliesmumbai,test_anomalies_mumbai[0][i],test_anomalies_mumbai[1][i],mumbailabelsnew)
			overlapped_mumbai = overlapped_mumbai + covered_mumbai
			actual = calculate_actual(anomaliesmumbai,test_anomalies_mumbai[0][i],test_anomalies_mumbai[1][i],mumbailabelsnew)
			yall_test_new.append(actual)
			if(ans ):
				overlapped += 1
				passed_step1_correctly.append(1)
			else:
				passed_step1_correctly.append(0)
			i += 2*DAYS
		i+=1

	print "Anomalies covered in Mumbai -", (np.unique(np.array(overlapped_mumbai)))


	overlapped_lucknow = []
	i=0	
	while ( i < len(x3_test)):
		parameter = cost_function(x3_test[i],test_anomalies_lucknow[0][i],test_anomalies_lucknow[1][i],DAYS)
			
		if(parameter > threshold):
			passed += 1
			xall_test_new.append(x3_test2[i])
			# yall_test_new.append(-1)
			label_locations.append(2)
			anomalies_passed.append((test_anomalies_lucknow[0][i],test_anomalies_lucknow[1][i]))

			# yall_test_new.append(yall_test[i])
			test_anomalies_year_new.append(test_lucknow_anomalies_year[i])
			ans, covered_lucknow = overlapping(anomalieslucknow,test_anomalies_lucknow[0][i],test_anomalies_lucknow[1][i],lucknowlabelsnew)
			overlapped_lucknow = overlapped_lucknow + covered_lucknow
			actual = calculate_actual(anomalieslucknow,test_anomalies_lucknow[0][i],test_anomalies_lucknow[1][i],lucknowlabelsnew)
			yall_test_new.append(actual)
			if( ans ):
				overlapped += 1
				passed_step1_correctly.append(1)
			else:
				passed_step1_correctly.append(0)
			i += 2*DAYS
		i += 1
				
	print "Anomalies covered in Lucknow -", (np.unique(np.array(overlapped_lucknow)))

	print passed,overlapped, len(xall_test_new)

	final_test_data, final_test_labels, final_test_info, final_test_loc = partition2(xall_test_new,yall_test_new,test_anomalies_year_new,6,anomalies_passed,label_locations)
	# temp = 0
	# for t in range(0,len(final_test_data)):
	# 	temp = temp + len(final_test_data[t])
	# assert(temp == len(xall_test_new))
	assert(len(label_locations) == len(anomalies_passed))
	assert(len(label_locations) == len(xall_test_new))
	return final_test_data, final_test_labels, label_locations, anomalies_passed, yall_test_new, final_test_info, final_test_loc

def analyse(predicted, test_anomaliesall, label_locations, anomaliesdelhi, anomaliesmumbai, anomalieslucknow, delhilabelsnew, mumbailabelsnew, lucknowlabelsnew):
	delhi_hoarding = []
	delhi_weather = []
	mumbai_hoarding = []
	mumbai_weather = []
	lucknow_hoarding = []
	lucknow_weather = []

	false_positive_hoarding = 0
	false_positive_weather = 0
	for i in range(0,len(test_anomaliesall)):
		if(label_locations[i] == 0):
			ans, covered_delhi_anomalies = overlapping_step2(anomaliesdelhi, test_anomaliesall[i][0], test_anomaliesall[i][1], delhilabelsnew, predicted[i])
			# ans, covered_delhi_anomalies = overlapping(anomaliesdelhi, test_anomaliesall[i][0], test_anomaliesall[i][1], delhilabelsnew)

			if(ans == True):
				if(predicted[i] == 5):
					false_positive_hoarding += 1
				elif(predicted[i] == 2):
					false_positive_weather += 1
			if(predicted[i] == 5):
				delhi_hoarding = delhi_hoarding + covered_delhi_anomalies
			elif(predicted[i] == 2):
				delhi_weather = delhi_weather + covered_delhi_anomalies
		elif(label_locations[i] == 1):
			ans, covered_mumbai_anomalies = overlapping_step2(anomaliesmumbai, test_anomaliesall[i][0], test_anomaliesall[i][1], mumbailabelsnew, predicted[i])
			# ans, covered_mumbai_anomalies = overlapping(anomaliesmumbai, test_anomaliesall[i][0], test_anomaliesall[i][1], mumbailabelsnew)

			if(ans == True):
				if(predicted[i] == 5):
					false_positive_hoarding += 1
				elif(predicted[i] == 2):
					false_positive_weather += 1
			if(predicted[i] == 5):
				mumbai_hoarding = mumbai_hoarding + covered_mumbai_anomalies
			elif(predicted[i] == 2):
				mumbai_weather = mumbai_weather + covered_mumbai_anomalies
		elif(label_locations[i] == 2):
			ans, covered_lucknow_anomalies = overlapping_step2(anomalieslucknow, test_anomaliesall[i][0], test_anomaliesall[i][1], lucknowlabelsnew, predicted[i])
			# ans, covered_lucknow_anomalies = overlapping(anomalieslucknow, test_anomaliesall[i][0], test_anomaliesall[i][1], lucknowlabelsnew)

			if(ans == True):
				if(predicted[i] == 5):
					false_positive_hoarding += 1
				elif(predicted[i] == 2):
					false_positive_weather += 1
			if(predicted[i] == 5):
				lucknow_hoarding = lucknow_hoarding + covered_lucknow_anomalies
			elif(predicted[i] == 2):
				lucknow_weather = lucknow_weather + covered_lucknow_anomalies

	# delhi_hoarding = np.array(delhi_hoarding)
	# delhi_weather = np.array(delhi_weather)
	# mumbai_hoarding = np.array(mumbai_hoarding)
	# mumbai_weather = np.array(mumbai_weather)
	# lucknow_hoarding = np.array(lucknow_hoarding)
	# lucknow_weather = np.array(lucknow_weather)

	# print len(np.unique(delhi_hoarding))+len(np.unique(mumbai_hoarding))+len(np.unique(lucknow_hoarding))
	# print len(np.unique(delhi_weather))+len(np.unique(mumbai_weather))+len(np.unique(lucknow_weather))
	print false_positive_weather, false_positive_hoarding


def get_score(xtrain,xtest,ytrain,ytest):
	scaler = preprocessing.StandardScaler().fit(xtrain)
	xtrain = scaler.transform(xtrain)
	xtest = scaler.transform(xtest)
	model = RandomForestClassifier(max_depth=2, random_state=0)
	# model = SVC(kernel='rbf', C=0.8)
	model.fit(xtrain,ytrain)
	test_pred = np.array(model.predict(xtest))
	
	return test_pred

def train_test_function(align_m,align_d,align_l,data_m,data_d,data_l):
	# align = [1,2,3]

	anomaliesmumbai = get_anomalies('data/anomaly/normal_h_w_mumbai.csv',align_m)
	anomaliesdelhi = get_anomalies('data/anomaly/normal_h_w_delhi.csv',align_d)
	anomalieslucknow = get_anomalies('data/anomaly/normal_h_w_lucknow.csv',align_l)
	# anomaliesall = pd.concat([anomaliesdelhi,anomaliesmumbai,anomalieslucknow],ignore_index = True)
	delhilabelsnew = newlabels(anomaliesdelhi,delhilabels)
	lucknowlabelsnew = newlabels(anomalieslucknow,lucknowlabels)
	mumbailabelsnew = newlabels(anomaliesmumbai,mumbailabels)
	test_data, test_labels, label_locations, test_anomaliesall, actual_test_labels, test_info, test_loc = first_step(retailpriceseriesmumbai,retailpriceseriesdelhi,retailpriceserieslucknow,[retailpriceseriesmumbai],[retailpriceseriesdelhi],[retailpriceserieslucknow],anomaliesmumbai,anomaliesdelhi,anomalieslucknow,mumbailabelsnew,delhilabelsnew,lucknowlabelsnew,data_m,data_d,data_l)


	x1,y1 = prepare(anomaliesdelhi,delhilabelsnew,data_d)
	x2,y2 = prepare(anomaliesmumbai,mumbailabelsnew,data_m)
	x3,y3 = prepare(anomalieslucknow,lucknowlabelsnew,data_l)
	

	
	delhi_anomalies_year = get_anomalies_year(anomaliesdelhi)
	mumbai_anomalies_year = get_anomalies_year(anomaliesmumbai)
	lucknow_anomalies_year = get_anomalies_year(anomalieslucknow)
	xall = np.array(x1.tolist()+x2.tolist()+x3.tolist())
	yall = np.array(y1.tolist()+y2.tolist()+y3.tolist())
	yearall = np.array(delhi_anomalies_year+mumbai_anomalies_year+lucknow_anomalies_year)
	
	xall_new = []
	yall_new = []
	yearall_new = []

	for y in range(0,len(yall)):
		if( yall[y] == 2 or yall[y]==5 ):
			xall_new.append(xall[y])
			yall_new.append(yall[y])
			yearall_new.append(yearall[y])
		# elif(yall[y] == 3):
		# 	xall_new.append(xall[y])
		# 	yall_new.append(0)
		# 	yearall_new.append(yearall[y])

	assert(len(xall_new) == len(yearall_new))
	# print len(xall_new)
	train_data, train_labels = partition(xall_new,yall_new,yearall_new,6)

	# temp = 0
	# for t in range(0,len(test_data)):
	# 	temp += len(test_data[t])
	# assert(temp == len(test_anomaliesall))
	assert(len(train_data) == len(test_data))

	# print len(train_data), len(test_data)
	predicted = []
	actual_labels = []
	actual_info = []
	actual_loc = []
	temp = 0
	for i in range(0,len(train_data)):
		if( len(test_data[i]) != 0):
			test_split = test_data[i]
			test_labels_split = test_labels[i]
			actual_labels = actual_labels + test_labels_split
			actual_info = actual_info + test_info[i]
			actual_loc = actual_loc + test_loc[i]
			train_split = []
			train_labels_split = []
			for j in range(0,len(train_data)):
				if( j != i):
					train_split = train_split + train_data[j]
					train_labels_split = train_labels_split+train_labels[j]
			pred_test = get_score(train_split,test_split,train_labels_split,test_labels[i])
			# temp = temp + len(pred_test)	
			predicted = predicted + pred_test.tolist()
		# print len(train_data[i]),len(test_data[i])
	predicted = np.array(predicted)
	# print temp, len(predicted), len(test_anomaliesall)
	# assert(len(predicted) == temp)
	assert(len(predicted) == len(test_anomaliesall))
	assert(len(actual_info) == len(actual_loc))
	assert(len(actual_info)==len(predicted))
	analyse(predicted, test_anomaliesall, label_locations, anomaliesdelhi, anomaliesmumbai, anomalieslucknow, delhilabelsnew, mumbailabelsnew, lucknowlabelsnew)
	# print len(predicted), len(actual_labels), "Line 652"
	# assert(len(predicted) == len(actual_test_labels))
	actual_test_labels_new = []
	predicted_new  = []
	# for x in range(0,len(predicted)):
	# 	if(actual_labels[x] == 2 or actual_labels[x] == 5):
	# 		actual_test_labels_new.append(actual_labels[x])
	# 		predicted_new.append(predicted[x])
	count_hoarding = 0
	count_weather = 0
	count_inflation = 0
	temp_count = 0

	for x in range(0,len(predicted)):
		if(predicted[x] == 2 and actual_labels[x] == -1):
			
			test_parameter = -200
			if(actual_loc[x] == 0):
				test_parameter = cost_function2(retailpriceseriesdelhi[actual_info[x][0]:actual_info[x][1]],2*DAYS+1)
			elif(actual_loc[x] == 1):
				test_parameter = cost_function2(retailpriceseriesmumbai[actual_info[x][0]:actual_info[x][1]],2*DAYS+1)

			elif(actual_loc[x] == 2):
				test_parameter = cost_function2(retailpriceserieslucknow[actual_info[x][0]:actual_info[x][1]],2*DAYS+1)
			else:
				print "Some error"
			if(test_parameter < 300 and test_parameter > 0):
				temp_count += 1
			# if(actual_labels[x] == 2):
			# 	count_weather += 1
			# elif(actual_labels[x] == 5):
			# 	count_hoarding += 1
			# elif(actual_labels[x] == 3):
			# 	count_inflation += 1
	print count_hoarding, count_weather, count_inflation,temp_count
	# exit()
	# actual_labels= np.array(actual_labels)
	# print len(actual_labels)
	# print sum(predicted == actual_labels)


	# from sklearn.metrics import confusion_matrix
	# print confusion_matrix(actual_test_labels_new,predicted_new)

	# train_data = []
	# train_labels = []
	# actual_train_labels = []
	# for i in range(0,len(total_data)):
	# 	if(len(total_data[i])!= 0):
	# 		train_data = train_data + total_data[i]
	# 		train_labels = train_labels + total_labels[i]
	# 		actual_train_labels = actual_train_labels + total_labels[i]
	# pred_test = get_score(train_data,train_data,train_labels,train_labels)	
	# print sum(pred_test == actual_train_labels)/104.0

	# print actual_labels
	# print predicted
	# print f1_score(actual_labels,predicted,labels=[2,3,5],average="macro")


# train_test_function(retailpriceseriesmumbai,retailpriceseriesdelhi,retailpriceserieslucknow,[retailpriceseriesmumbai],[retailpriceseriesdelhi],[retailpriceserieslucknow])
# train_test_function(retailpriceseriesmumbai,retailpriceseriesdelhi,retailpriceserieslucknow,[mandipriceseriesmumbai],[mandipriceseriesdelhi],[mandipriceserieslucknow])
# train_test_function(retailpriceseriesmumbai,retailpriceseriesdelhi,retailpriceserieslucknow,[retailpriceseriesmumbai,mandipriceseriesmumbai],[retailpriceseriesdelhi,mandipriceseriesdelhi],[retailpriceserieslucknow,mandipriceserieslucknow])
train_test_function(retailpriceseriesmumbai,retailpriceseriesdelhi,retailpriceserieslucknow,[retailpriceseriesmumbai-mandipriceseriesmumbai,mandiarrivalseriesmumbai],[retailpriceseriesdelhi-mandipriceseriesdelhi,mandiarrivalseriesdelhi],[retailpriceserieslucknow-mandipriceserieslucknow,mandiarrivalserieslucknow])
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


#Change the argmax to idxmin for running the part below

# train_test_function(mandiarrivalseriesmumbai,mandiarrivalseriesdelhi,mandiarrivalserieslucknow,[retailpriceseriesmumbai],[retailpriceseriesdelhi],[retailpriceserieslucknow])
# train_test_function(mandiarrivalseriesmumbai,mandiarrivalseriesdelhi,mandiarrivalserieslucknow,[mandipriceseriesmumbai],[mandipriceseriesdelhi],[mandipriceserieslucknow])
# train_test_function(mandiarrivalseriesmumbai,mandiarrivalseriesdelhi,mandiarrivalserieslucknow,[retailpriceseriesmumbai,mandipriceseriesmumbai],[retailpriceseriesdelhi,mandipriceseriesdelhi],[retailpriceserieslucknow,mandipriceserieslucknow])
# train_test_function(mandiarrivalseriesmumbai,mandiarrivalseriesdelhi,mandiarrivalserieslucknow,[retailpriceseriesmumbai-mandipriceseriesmumbai,mandiarrivalseriesmumbai],[retailpriceseriesdelhi-mandipriceseriesdelhi,mandiarrivalseriesdelhi],[retailpriceserieslucknow-mandipriceserieslucknow,mandiarrivalserieslucknow])
# train_test_function(mandiarrivalseriesmumbai,mandiarrivalseriesdelhi,mandiarrivalserieslucknow,[retailpriceseriesmumbai-mandipriceseriesmumbai],[retailpriceseriesdelhi-mandipriceseriesdelhi],[retailpriceserieslucknow-mandipriceserieslucknow])
# train_test_function(mandiarrivalseriesmumbai,mandiarrivalseriesdelhi,mandiarrivalserieslucknow,[retailpriceseriesmumbai,mandiarrivalseriesmumbai],[retailpriceseriesdelhi,mandiarrivalseriesdelhi],[retailpriceserieslucknow,mandiarrivalserieslucknow])
# train_test_function(mandiarrivalseriesmumbai,mandiarrivalseriesdelhi,mandiarrivalserieslucknow,[retailpriceseriesmumbai,mandipriceseriesmumbai,mandiarrivalseriesmumbai],[retailpriceseriesdelhi,mandipriceseriesdelhi,mandiarrivalseriesdelhi],[retailpriceserieslucknow,mandipriceserieslucknow,mandiarrivalserieslucknow])
# train_test_function(mandiarrivalseriesmumbai,mandiarrivalseriesdelhi,mandiarrivalserieslucknow,[retailpriceseriesmumbai/mandipriceseriesmumbai],[retailpriceseriesdelhi/mandipriceseriesdelhi],[retailpriceserieslucknow/mandipriceserieslucknow])

