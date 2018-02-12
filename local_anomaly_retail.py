from datetime import datetime
import pandas as pd
import numpy as np
import scipy
from constants import CONSTANTS
import matplotlib.pyplot as plt
import math

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

retailP = pd.read_csv(CONSTANTS['ORIGINALRETAIL'], header=None)

RP = 2
START = CONSTANTS['STARTDATE']
END = CONSTANTS['ENDDATE']
CENTREID = 1
retailP = retailP[retailP[RP] != 0]
retailP = retailP[np.isfinite(retailP[RP])]
retailP = retailP[retailP[0] >= START]
retailP = retailP[retailP[0] <= END]
retailP = retailP.drop_duplicates(subset=[0, 1], keep='last')
retailP = retailP[retailP[0] >= START]
retailP = retailP[retailP[0] <= END]
start_date = '2006-01-01'
end_date = '2015-06-23'



def CreateCentreSeries(Centre, RetailPandas):
	rc = RetailPandas[RetailPandas[1] == Centre]
	rc = rc.sort_values([0], ascending=[True])
	rc[3] = rc.apply(lambda row: datetime.strptime(row[0], '%Y-%m-%d'), axis=1)
	rc.drop(rc.columns[[0, 1]], axis=1, inplace=True)
	rc.set_index(3, inplace=True)
	rc.index.names = [None]
	idx = pd.date_range(START, END)
	rc = rc.reindex(idx, fill_value=0)
	return rc * 100



def RemoveNaNFront(series):
	index = 0
	while True:
		if(not np.isfinite(series[index])):
			index += 1
		else:
			break
	if(index < len(series)):
		for i in range(0, index):
			series[i] = series[index]
	return series


from os import listdir
imagenames = [f for f in listdir('plots/bigmandis10')]


centrenames = []
for imagename in imagenames:
	imagename = imagename.replace('.','_')
	[statename,centrename,mandiname,_] = imagename.split('_')
	if centrename not in centrenames:
		centrenames.append(centrename)

centreSeries = []
for cname in centrenames:
	code = dict_centrename_centreid[cname][0]
	series = CreateCentreSeries(code,retailP)
	price = series[RP]   
	price = price.replace(0.0, np.NaN, regex=True)
	price = price.interpolate(method='pchip')
	price = RemoveNaNFront(price)
	centreSeries.append(price)
	

centreDF = pd.DataFrame()
for i in range(0, len(centreSeries)):
	centreDF[i] = centreSeries[i]


colors = [ '#e6194b', '#3cb44b' ,'#ffe119', '#0082c8', '#f58231', '#911eb4',  '#000080' , '#800000'  ,'#000000' ]

def plot_year_series(inpseries,idx):
	centrename = centrenames[idx]
	print(centrename)
	yaxis = list(inpseries)
	xaxis = list(range(0,len(inpseries)))
	plt.plot(xaxis,yaxis, color = colors[idx] ,label=str(centrename))


def plot_year_series_avg(inpseries,lbl):
	yaxis = list(inpseries)
	xaxis = list(range(0,len(inpseries)))
	plt.plot(xaxis,yaxis, color = '#000000' ,label=lbl)



for i in range(0,len(centreDF.T)):
	if i == 2:
		centreseries = centreDF[i]
		centreseries = centreseries.groupby([centreseries.index.month, centreseries.index.day]).mean()
		#centreseries = (centreseries - centreseries.mean())/centreseries.std()
		plot_year_series(centreseries,i)

meantrend = centreDF.mean(axis=1)
meantrend = meantrend.groupby([meantrend.index.month, meantrend.index.day]).mean()
#meantrend = (meantrend - meantrend.mean())/meantrend.std()

plot_year_series_avg(meantrend,'Average')
plt.title('Retail Price Yearly Average')
plt.xlabel('Time')
plt.ylabel('Normalised Price')
plt.legend(loc='best')
plt.show()
