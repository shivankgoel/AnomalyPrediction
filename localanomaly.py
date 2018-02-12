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

START = CONSTANTS['STARTDATE']
END = CONSTANTS['ENDDATE']

WP = 7
WA = 2
wholeSalePA = pd.read_csv(CONSTANTS['ORIGINALMANDI'], header=None)
wholeSalePA = wholeSalePA[wholeSalePA[WA] != 0]
wholeSalePA = wholeSalePA[wholeSalePA[WP] != 0]
wholeSalePA = wholeSalePA[np.isfinite(wholeSalePA[WA])]
wholeSalePA = wholeSalePA[np.isfinite(wholeSalePA[WP])]
wholeSalePA = wholeSalePA[wholeSalePA[0] >= START]
wholeSalePA = wholeSalePA[wholeSalePA[0] <= END]
wholeSalePA = wholeSalePA.drop_duplicates(subset=[0, 1], keep='last')


def CreateMandiSeries(Mandi, MandiPandas):
  mc = MandiPandas[MandiPandas[1] == Mandi]
  mc = mc.sort_values([0], ascending=[True])
  mc[8] = mc.apply(lambda row: datetime.strptime(row[0], '%Y-%m-%d'), axis=1)
  mc.drop(mc.columns[[0, 1, 3, 4, 5, 6]], axis=1, inplace=True)
  mc.set_index(8, inplace=True)
  mc.index.names = [None]
  idx = pd.date_range(START, END)
  mc = mc.reindex(idx, fill_value=0)
  return mc

start_date = '2006-01-01'
end_date = '2015-06-23'



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


mandinames = []
for imagename in imagenames:
  imagename = imagename.replace('.','_')
  [statename,centrename,mandiname,_] = imagename.split('_')
  if mandiname not in mandinames:
  	mandinames.append(mandiname)

mseries = []
for mname in mandinames:
  code = dict_mandiname_mandicode[mname][0]
  series = CreateMandiSeries(code,wholeSalePA)
  price = series[7]   
  price = price.replace(0.0, np.NaN, regex=True)
  price = price.interpolate(method='pchip')
  price = RemoveNaNFront(price)
  mseries.append(price)
  

mandiDF = pd.DataFrame()
for i in range(0, len(mseries)):
  mandiDF[i] = mseries[i]


colors = [ '#e6194b', '#3cb44b' ,'#ffe119', '#0082c8', '#f58231', '#911eb4',  '#000080' , '#800000'  ,'#00f000' ]

def plot_year_series(inpseries,idx):
	centrename = mandinames[idx]
	print(centrename)
	yaxis = list(inpseries)
	xaxis = list(range(0,len(inpseries)))
	plt.plot(xaxis,yaxis, color = colors[idx] ,label=str(centrename))

def plot_year_series_avg(inpseries,lbl):
  yaxis = list(inpseries)
  xaxis = list(range(0,len(inpseries)))
  plt.plot(xaxis,yaxis, color = '#000000' ,label=lbl)


for i in range(0,len(mandiDF.T)):
	if i!= -1:
		mandiseries = mandiDF[i]
		mandiseries = mandiseries.groupby([mandiseries.index.month, mandiseries.index.day]).mean()
		#mandiseries = (mandiseries - mandiseries.mean())/mandiseries.std()
		plot_year_series(mandiseries,i)


meantrend = mandiDF.mean(axis=1)
meantrend = meantrend.groupby([meantrend.index.month, meantrend.index.day]).mean()
#meantrend = (meantrend - meantrend.mean())/meantrend.std()
plot_year_series_avg(meantrend,'Average')
plt.title('Arrival Yearly Average')
plt.xlabel('Time')
plt.ylabel('Normalised Arrival')
plt.legend(loc='best')
plt.show()
