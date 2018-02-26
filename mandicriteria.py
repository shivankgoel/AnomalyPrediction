from datetime import datetime
import pandas as pd
import numpy as np
from constants import CONSTANTS
import matplotlib.pyplot as plt
import matplotlib
import sklearn.preprocessing as preprocessing
import os


mandi_info = pd.read_csv('data/original/mandis.csv')
dict_centreid_mandicode = mandi_info.groupby('centreid')['mandicode'].apply(list).to_dict()
dict_mandicode_mandiname = mandi_info.groupby('mandicode')['mandiname'].apply(list).to_dict()
dict_mandicode_statecode = mandi_info.groupby('mandicode')['statecode'].apply(list).to_dict()
dict_mandicode_centreid = mandi_info.groupby('mandicode')['centreid'].apply(list).to_dict()
dict_mandiname_mandicode = mandi_info.groupby('mandiname')['mandicode'].apply(list).to_dict()

centre_info = pd.read_csv('data/original/centres.csv')
dict_centreid_centrename = centre_info.groupby('centreid')['centrename'].apply(list).to_dict()
dict_centreid_statecode = centre_info.groupby('centreid')['statecode'].apply(list).to_dict()
dict_statecode_centreid = centre_info.groupby('statecode')['centreid'].apply(list).to_dict()
dict_centrename_centreid = centre_info.groupby('centrename')['centreid'].apply(list).to_dict()

state_info = pd.read_csv('data/original/states.csv')
dict_statecode_statename = state_info.groupby('statecode')['state'].apply(list).to_dict() 
dict_statename_statecode = state_info.groupby('state')['statecode'].apply(list).to_dict() 


from averagemandi import getmandi2
m1 = getmandi2('Azadpur',True)
m2 = getmandi2('Devariya',True)
m3 = getmandi2('Ahmednagar',True)
m4 = getmandi2('Lasalgaon',True)
m5 = getmandi2('Newasa(Ghodegaon)',True)
m6 = getmandi2('Pune',True)
m7 = getmandi2('Rahuri',True)
m8 = getmandi2('Pimpalgaon',True)

mandis = [m1,m2,m3,m4,m5,m6,m7,m8]


def getzeros(series,idx):
	c = 0
	while(idx<len(series) and series[idx]==0):
		idx = idx+1
		c = c+1
	return c

def checkcriteria(series):
	series = series[series.index.dayofweek < 5]
	a = len(series)
	b = len(series[series!=0])
	ratio = b*1.0/a
	maxgap = 0
	for i in range(0,len(series)):
		num = getzeros(series,i)
		if num>0:
			i = i+num-1
		if num>maxgap:
			maxgap = num
	return ratio,maxgap

# for series in mandis:
# 	r,m = checkcriteria(series)
# 	print r,m


lucknowmandis = [797, 898, 1092, 295, 1141, 522, 549, 1107, 277, 558, 1120, 545, 323,1086, 1085, 405, 749, 1093, 550, 761, 887, 284, 1121, 524, 324, 319, 1087, 885, 1089, 1094, 584]
lucknowmandis2 = [884, 553, 285, 326, 278, 288, 1158, 951, 906, 272]
lnames = ['Gonda','Lakhimpur','Sitapur' ,'Devariya','Bijnaur' ,'Lucknow' ,'Chandoli' ,'Bahraich' , 'Faizabad' ]

# for name in lnames:
# 	#name = dict_mandicode_mandiname[m][0]
# 	series = getmandi2(name,True)
# 	r,m = checkcriteria(series)
# 	if r > 0.50:
# 		print name,r,m


'''
m3 = getmandi2('Ahmednagar',True)
m4 = getmandi2('Lasalgaon',True)
m5 = getmandi2('Newasa(Ghodegaon)',True)
m6 = getmandi2('Pune',True)
m7 = getmandi2('Rahuri',True)
m8 = getmandi2('Pimpalgaon',True)
'''

'''
0.894178192345 36
0.718237375362 225
#0.135413316179 268
0.65648118366 58
#0.2206497266 327
0.720167256353 27
#0.208748793824 334
0.443229334191 87
'''

'''
Gonda 0.6678802589 196
Lakhimpur 0.71642394822 62
Sitapur 0.758090614887 97
Devariya 0.717233009709 225
Bijnaur 0.696601941748 457
Lucknow 0.791666666667 101
Chandoli 0.634708737864 123
Bahraich 0.81067961165 22
Faizabad 0.894822006472 74
'''


'''
m1 = getmandi2('Azadpur',True)
m2 = getmandi2('Bahraich',True)
m4 = getmandi2('Lasalgaon',True)
m6 = getmandi2('Pune',True)
'''

'''
0.894178192345 36
0.81067961165 22
0.65648118366 58
0.720167256353 27
'''