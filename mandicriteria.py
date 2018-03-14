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
mumbaimandis = [430, 798, 424, 655, 155, 1551, 162, 505, 378, 1367, 930, 1365, 854, 720, 157, 602, 1169, 149, 544, 143, 375, 150, 838, 426, 153, 905, 537, 1428, 163, 1377, 377, 429, 156, 1190, 423, 428, 939, 1041, 427, 422, 146, 474, 792, 853, 862, 160, 477, 540, 937, 144, 769, 425, 476, 897, 502, 1029, 1058, 1249, 1378, 148]
nagpurmandis = [1053, 1579, 1227, 379,667]

lnames = ['Gonda','Lakhimpur','Sitapur' ,'Devariya','Bijnaur' ,'Lucknow' ,'Chandoli' ,'Bahraich' , 'Faizabad' ]


cname = 'MUMBAI'

lmandis = dict_centreid_mandicode[dict_centrename_centreid[cname][0]]
cols = ['name','ratio','missing','average']
ans = pd.DataFrame(columns = cols )
for i in range(len(lmandis)):
	mcode = lmandis[i]
	if mcode in dict_mandicode_mandiname.keys(): 
		name = dict_mandicode_mandiname[mcode][0]
		series = getmandi2(name,False)
		r,m = checkcriteria(series)
		l = [name,r,m,series.mean()]
		ans.loc[i] = l
ans.to_excel('data/'+cname+'mandis.xlsx')




'''
CentreName CentreId 
Lucknow 	40

'''


############################################
#########  RESULTS   #######################
############################################

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

'''
Gonda 0.6678802589 196 26.5979485698
Lakhimpur 0.71642394822 62 7.7436174516
Sitapur 0.758090614887 97 14.8438890494
Devariya 0.717233009709 225 44.468650679
Bijnaur 0.696601941748 457 1.51853221612
Lucknow 0.791666666667 101 55.3026004045
Chandoli 0.634708737864 123 3.19803524993
Bahraich 0.81067961165 22 26
Faizabad 0.894822006472 74 13.2631175961
'''


'''

Lucknow

Sultanpur 0.7809585075587006 171 34.86169577205882
Salon 0.018012222579607592 1792 0.07202435661764701
Misrikh 0.0077195239626889674 1109 0.04565716911764708
Gonda 0.7426825345770344 196 26.177297794117653
Safdarganj 0.0829848825989064 1759 0.6268727022058824
Balrampur 0.5744612415567707 720 7.65561580882352
Panchpedwa 0.08941781923448054 1370 0.8952435661764706
Golagokarnath 0.029591508523641043 1755 0.12695542279411764
Bachranwa 0.07172724348665166 1561 0.12866038602941177
Sandila 0.34287552267610166 633 2.514522058823538
Rasda 0.04824702476680605 1738 2.3637867647058823
Lakhimpur 0.7732389835960116 46 11.70890625
Sitapur 0.7774203924091347 97 27.368589154411783
Utraula 0.00964940495336121 883 0.4169806985294117
Maholi 0.006754583467352846 1109 0.043382352941176476
Devariya 0.822129302026375 21 71.87670036764706
Jayas 0.17658411064651014 426 2.328504136029408
Mihipurwa 0.008041170794467674 1737 0.9721507352941177
Payagpur 0.40431006754583465 1196 2.2999540441176465
Naanpara 0.15857188806690253 1239 2.3558823529411743
Mohammdi 0.014152460598263108 576 0.08100873161764707
Bijnaur 0.6500482470247668 457 2.1615326286764702
Tikonia 0.014152460598263108 1781 0.03188189338235294
Barabanki 0.66420070762303 222 15.189131433823531
Sultanpur 0.7809585075587006 171 34.86169577205882
Sahiyapur 0.16757799935670634 841 2.3013786764705877
Viswan 0.03473785783210035 1109 0.1977481617647059
Atrauli 0.06239948536506915 1390 1.1331801470588236
Mehmoodabad 0.0508201994210357 988 0.4746438419117646
Risia 0.0077195239626889674 1736 0.4074448529411765
Lucknow 0.8118366034094564 101 63.87130055147058
Maigalganj 0.047603731103248635 1068 0.06958869485294118
Purwa 0.1466709552910904 795 0.11171875
Chandoli 0.622386619491798 123 3.770174632352942
Tulsipur 0.05275008041170794 820 0.843658088235294
Bahraich 0.8610485686715986 22 20.23566176470588
Faizabad 0.8481826954004503 147 13.792130055147041
Rudauli 0.0032164683177870698 1782 0.07479319852941177
Nawabganj 0.0739787713091026 1501 1.807513786764707
Hargaon (Laharpur) 0.06979736249597941 1438 0.22609374999999995
Aliganj 0.05210678674815053 1065 0.19749540441176472
'''
