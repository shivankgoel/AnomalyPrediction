'''
Check if ritesh's 30 points can be quantified.
'''
from constants import CONSTANTS
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib 
from datetime import datetime
import dateutil.relativedelta

font = {'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 14}

matplotlib.rc('font', **font)


from averagemandi import mandipriceseries
from averageretail import retailpriceseries
from averagemandi import mandiarrivalseries
from rainfallmonthly import rainfallmonthly
from fuelprice import fuelpricemumbai
from oilmonthlyseries import oilmonthlyseries
from cpimonthlyseries import cpimonthlyseries
from average_export import exportseries



colors = [ '#e6194b', '#3cb44b' ,'#ffe119', '#0082c8', '#f58231', '#911eb4',  '#000080' , '#800000'  ,'#000000', '#900c3f' ]
anomalies = pd.read_csv(CONSTANTS['ANOMALIES_NEWSPAPER'], header=None, index_col=None)
anomalies[0] = [ datetime.strftime(datetime.strptime(date, '%d/%m/%Y'),'%Y-%m-%d') for date in anomalies[0]]
anomalies[1] = [ datetime.strftime(datetime.strptime(date, ' %d/%m/%Y'),'%Y-%m-%d') for date in anomalies[1]]
#print(anomalies[1])

def showanomaly(anomaly_num):
	global retailpriceseries,mandipriceseries,mandiarrivalseries,rainfallmonthly
	retailpriceseries = retailpriceseries.rolling(window=7,center=True).mean()
	mandipriceseries = mandipriceseries.rolling(window=7,center=True).mean()
	mandiarrivalseries = mandiarrivalseries.rolling(window=7,center=True).mean()
	rainfallmonthly = rainfallmonthly.rolling(window=7,center=True).mean()
	startdate = anomalies.iloc[anomaly_num][0]
	enddate = anomalies.iloc[anomaly_num][1]
	labels = anomalies.iloc[anomaly_num][2]
	labels = labels.strip(' ').split(' ')
	mandiprice = mandipriceseries[startdate:enddate]
	mandiarrival = mandiarrivalseries[startdate:enddate]
	retailprice = retailpriceseries[startdate:enddate]
	rainfall = rainfallmonthly[startdate:enddate]
	fuel = fuelpricemumbai[startdate:enddate]
	xaxis = list(mandiarrival.index)
	oil = oilmonthlyseries[startdate:enddate]
	cpi = cpimonthlyseries[startdate:enddate]
	export = exportseries[startdate:enddate]
	fig,yaxis1 = plt.subplots() 
	yaxis1.plot(xaxis, list(mandiprice), color =colors[4] , label='mandiprice')
	yaxis1.plot(xaxis, list(retailprice), color =colors[5] , label='retailprice')
	yaxis1.legend(loc = (0.05,0.9), frameon = False)
	yaxis1.set_ylabel('Mandi and Retail Prices')
	if 'Weather' in labels:
		yaxis2 = yaxis1.twinx()
		yaxis2.plot(xaxis, list(rainfall), color =colors[6] , label='rainfall')
		yaxis2.legend(loc = (0.05,0.85), frameon = False)
		yaxis2.set_ylabel('Rainfall')
	elif 'Transport' in labels:
		yaxis2 = yaxis1.twinx()
		yaxis2.plot(xaxis, list(oil), color =colors[6] , label='crude_oil_india')
		yaxis2.legend(loc = (0.05,0.85), frameon = False)
		yaxis2.set_ylabel('Crude Oil India')
	elif 'Fuel' in labels:
		yaxis2 = yaxis1.twinx()
		yaxis2.plot(xaxis, list(fuel), color =colors[6] , label='fuel')
		yaxis2.legend(loc = (0.05,0.85), frameon = False)
		yaxis2.set_ylabel('Fuel')
	elif 'Inflation' in labels:
		yaxis2 = yaxis1.twinx()
		yaxis2.plot(xaxis, list(cpi), color =colors[6] , label='cpi')
		yaxis2.legend(loc = (0.05,0.85), frameon = False)
		yaxis2.set_ylabel('CPI')
	elif 'Hoarding' in labels:
		yaxis2 = yaxis1.twinx()
		yaxis2.plot(xaxis, list(export), color =colors[6] , label='export')
		yaxis2.legend(loc = (0.05,0.85), frameon = False)
		yaxis2.set_ylabel('Export Series')
	fig.tight_layout()
	fig.suptitle(startdate +' : '+enddate+' '+anomalies.iloc[anomaly_num][2])
	fig.show()


columns = [ 'mandiprice' , 'retailprice','pricedifference' ,'rainfall','rainfall1','rainfall2','rainfall3','rainfall4','export','fuelprice','crudeoil','cpi']
mydf = pd.DataFrame(index=range(0,30), columns=columns)

for anomaly_num in range(0,30):
	startdate = anomalies.iloc[anomaly_num][0]
	enddate = anomalies.iloc[anomaly_num][1]
	labels = anomalies.iloc[anomaly_num][2]
	labels = labels.strip(' ').split(' ')
	mandiprice = mandipriceseries[startdate:enddate]
	mandiarrival = mandiarrivalseries[startdate:enddate]
	retailprice = retailpriceseries[startdate:enddate]
	rainfall = rainfallmonthly[startdate:enddate]
	fuel = fuelpricemumbai[startdate:enddate]
	oil = oilmonthlyseries[startdate:enddate]
	cpi = cpimonthlyseries[startdate:enddate]
	export = exportseries[startdate:enddate]
	mydf.iloc[anomaly_num]['mandiprice'] = mandiprice.mean()
	mydf.iloc[anomaly_num]['retailprice'] = retailprice.mean()
	mydf.iloc[anomaly_num]['pricedifference'] = (retailprice - mandiprice).mean()
	mydf.iloc[anomaly_num]['rainfall'] = rainfall.mean()
	mydf.iloc[anomaly_num]['export'] = export.mean()
	mydf.iloc[anomaly_num]['fuelprice'] = fuel.mean()
	mydf.iloc[anomaly_num]['crudeoil'] = oil.mean()
	mydf.iloc[anomaly_num]['cpi'] = cpi.mean()
	for i in range(1,5):
		s = datetime.strptime(startdate, "%Y-%m-%d")
		newstart = s - dateutil.relativedelta.relativedelta(months=i)
		e = datetime.strptime(enddate, "%Y-%m-%d")
		newend = e - dateutil.relativedelta.relativedelta(months=i)
		rain = rainfallmonthly[newstart:newend]
		mydf.iloc[anomaly_num]['rainfall'+str(i)] = rain.mean()


#mydf.round(2).to_excel('anomalies.xlsx', sheet_name='Sheet1')


def plotseries(series,clr,lbl):
		yaxis = list(series)
		xaxis = list(series.index)
		if (lbl != ''):
			plt.plot(xaxis,yaxis,color = colors[clr],label =lbl,alpha=0.7)
		else:
			plt.plot(xaxis,yaxis,color = colors[clr])

def plotanomalies(category):
	mandipriceseriesanomaly = pd.Series(index=mandipriceseries.index)
	retailpriceseriesanomaly = pd.Series(index=retailpriceseries.index)
	for anomaly_num in range(0,30):
		startdate = anomalies.iloc[anomaly_num][0]
		enddate = anomalies.iloc[anomaly_num][1]
		labels = anomalies.iloc[anomaly_num][2]
		labels = labels.strip(' ').split(' ')
		if(category == ''):
			mandipriceseriesanomaly[startdate:enddate] = mandipriceseries[startdate:enddate]
			retailpriceseriesanomaly[startdate:enddate] = retailpriceseries[startdate:enddate]
		elif(category in labels):
			mandipriceseriesanomaly[startdate:enddate] = mandipriceseries[startdate:enddate]
			retailpriceseriesanomaly[startdate:enddate] = retailpriceseries[startdate:enddate]
	plotseries(mandipriceseries,4,'Mandi Price')
	plotseries(retailpriceseries,3,'Retail Price')
	plotseries(mandipriceseriesanomaly,8,'')
	plotseries(retailpriceseriesanomaly,8,'')
	plt.title('Newspaper Anomalies '+category)
	plt.ylabel('Price')
	plt.xlabel('Time')
	plt.legend(loc='best')
	plt.show()

#plotanomalies('')

from reading_timeseries import retailP
# if 0 in retailP.columns:
# 	print('Deleting Ahmedabad')
# 	del retailP[0]

num_centers = retailP.shape[1]
if retailP.index.dtype != 'M8[ns]':
	retailP.index = [datetime.strptime(date,'%Y-%m-%d') for date in retailP.index]
retailpriceseries_old = retailP.mean(axis=1)

def plotanomalies_old(category):
	retailpriceseriesanomaly_old = pd.Series(index=retailpriceseries_old.index)
	for anomaly_num in range(0,30):
		startdate = anomalies.iloc[anomaly_num][0]
		enddate = anomalies.iloc[anomaly_num][1]
		labels = anomalies.iloc[anomaly_num][2]
		labels = labels.strip(' ').split(' ')
		if(category == ''):
			retailpriceseriesanomaly_old[startdate:enddate] = retailpriceseries_old[startdate:enddate]
		elif(category in labels):
			retailpriceseriesanomaly_old[startdate:enddate] = retailpriceseries_old[startdate:enddate]
	plotseries(retailpriceseries_old,3,'Retail Price')
	plotseries(retailpriceseriesanomaly_old,8,'')
	plt.title('Newspaper Anomalies '+category)
	plt.ylabel('Price')
	plt.xlabel('Time')
	plt.legend(loc='best')
	plt.show()

# plotanomalies_old('')
# plotanomalies_old('Hoarding')
# plotanomalies_old('Weather')
# plotanomalies_old('Fuel')
# plotanomalies_old('Transport')
# plotanomalies_old('Inflation')

def Normalise(df):
	num_cols = df.shape[1]
	for i in range(1,num_cols+1):
		m = df[i].mean()
		am = df[i].min()
		aM = df[i].max()
		df[i] -= m
		df[i] /= (aM - am)
	return df

def plotsignatures(category):
	data = [0]*43
	temp = pd.Series(data)
	c = 0
	for anomaly_num in range(0,30):
		startdate = anomalies.iloc[anomaly_num][0]
		enddate = anomalies.iloc[anomaly_num][1]
		labels = anomalies.iloc[anomaly_num][2]
		labels = labels.strip(' ').split(' ')
		if(category == ''):
			temp.values += Normalise(retailP[startdate:enddate]).mean(axis=1).values
			c+=1
		elif(category in labels):
			temp += Normalise(retailP[startdate:enddate]).mean(axis=1).values
			c+=1
	temp = temp / c
	plotseries(temp,8,'')
	plt.title('Signatures '+category)
	plt.ylabel('Price')
	plt.xlabel('Time')
	plt.legend(loc='best')
	plt.show()

#plotsignatures('Inflation')



seasons = pd.read_csv('data/seasons.csv', header=None, index_col=None)
num_seasons = seasons.shape[0]
mydf = pd.DataFrame(index=range(0,num_seasons), columns=columns)

for season_num in range(0,num_seasons):
	startdate = seasons.iloc[season_num][0]
	enddate = seasons.iloc[season_num][1]
	mandiprice = mandipriceseries[startdate:enddate]
	mandiarrival = mandiarrivalseries[startdate:enddate]
	retailprice = retailpriceseries[startdate:enddate]
	rainfall = rainfallmonthly[startdate:enddate]
	fuel = fuelpricemumbai[startdate:enddate]
	oil = oilmonthlyseries[startdate:enddate]
	cpi = cpimonthlyseries[startdate:enddate]
	export = exportseries[startdate:enddate]
	mydf.iloc[season_num]['mandiprice'] = mandiprice.mean()
	mydf.iloc[season_num]['retailprice'] = retailprice.mean()
	mydf.iloc[season_num]['pricedifference'] = (retailprice - mandiprice).mean()
	mydf.iloc[season_num]['rainfall'] = rainfall.mean()
	mydf.iloc[season_num]['export'] = export.mean()
	mydf.iloc[season_num]['fuelprice'] = fuel.mean()
	mydf.iloc[season_num]['crudeoil'] = oil.mean()
	mydf.iloc[season_num]['cpi'] = cpi.mean()
	for i in range(1,5):
		s = datetime.strptime(startdate, "%Y-%m-%d")
		newstart = s - dateutil.relativedelta.relativedelta(months=i)
		e = datetime.strptime(enddate, "%Y-%m-%d")
		newend = e - dateutil.relativedelta.relativedelta(months=i)
		rain = rainfallmonthly[newstart:newend]
		mydf.iloc[season_num]['rainfall'+str(i)] = rain.mean()

#mydf.round(2).to_excel('seasons.xlsx', sheet_name='Sheet1')



