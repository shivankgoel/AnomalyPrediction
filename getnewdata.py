from datetime import datetime
import pandas as pd
import numpy as np
import scipy
from constants import CONSTANTS
import matplotlib.pyplot as plt
import math
from os import listdir
import datetime as datetime

def isInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

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


src = 'c:/btp/shivankfinal/data/new/Delhi'
dest = 'c:/btp/shivankfinal/data/new/Delhi'

files = [f for f in listdir(src+'/Wholesale')]

code=-1

newfile = open(src+'/wholesale3.csv','a')

for j in range(0,len(files)):
	file = files[j]
	with open(src+'/Wholesale/'+file) as f:
		content = f.readlines()
	for i in range(1,len(content)):
		temp = content[i].strip().split(',')
		if(len(temp) > 8):
			temp[0:2] = [''.join(temp[0:2])]
		mandi = temp[0]
		date = temp[1]
		#print 1,mandi
		if date != '':
			date = datetime.datetime.strptime(date,'%d/%m/%Y').strftime('%Y-%m-%d')
			arrival = temp[2]
			variety = temp[3]
			minp = temp[4]
			maxp = temp[5]
			modalp = temp[6]
			if not isInt(minp):
				minp = '0'
			if not isInt(maxp):
				maxp = '0'
			if not isInt(modalp):
				modalp = '0'
			if mandi != '':
				print 1,mandi
				if mandi in dict_mandiname_mandicode.keys():
					code = dict_mandiname_mandicode[mandi]
				else:
					code = -1
				print 2,mandi
			if code != -1 and minp != 'NR':
				mystr = date+','+str(code[0])+','+arrival+',NR,'+variety+','+minp+','+maxp+','+modalp+'\n'
				newfile.write(mystr)

newfile.close()




		
