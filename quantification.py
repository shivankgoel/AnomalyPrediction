from datetime import datetime
import pandas as pd
import numpy as np
import scipy
from constants import CONSTANTS
import matplotlib
import matplotlib.pyplot as plt
import math

font = {'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 14}

matplotlib.rc('font', **font)

colors = [ '#e6194b', '#3cb44b' ,'#ffe119', '#0082c8', '#f58231', '#911eb4',  '#000080' , '#800000'  ,'#000000', '#900c3f' ]

from averagemandi import mandipriceseries
from averageretail import retailpriceseries
from averagemandi import expectedarrivalseries
from averagemandi import mandiarrivalseries


differenceseries = retailpriceseries - mandipriceseries
thresh_hold = differenceseries.mean() + 2 * differenceseries.std()
anomalous_points = differenceseries[differenceseries > thresh_hold]
anomalies = anomalous_points.groupby([anomalous_points.index.year, anomalous_points.index.month]).count()
anomalies = anomalies[anomalies >= 7]
print('Global Anomalies ')
print(anomalies)

'''
Output 

year  Month Frequency

2010  12     9
2011  1     31
2013  10     7
      11    30
      12    26
'''


retailpricelow = retailpriceseries[mandipriceseries<700]
mandipricelow = mandipriceseries[mandipriceseries<700]
disadvantageseries = retailpricelow - mandipricelow
thresh_hold = disadvantageseries.mean() + 2 * disadvantageseries.std()
anomalous_points = disadvantageseries[disadvantageseries > thresh_hold]
anomalies = anomalous_points.groupby([anomalous_points.index.year, anomalous_points.index.month]).count()
anomalies = anomalies[anomalies >= 7]
print('Farmer Disadvantage Periods ')
print(anomalies)

'''
Output 

2014  2    10
      3    12
      4    10
'''


def arrivalstatus(year):
	startdate = '01-01-'+str(year)
	enddate = '31-12-'+str(year)
	avgarrival = expectedarrivalseries[startdate:enddate]
	r7 = mandiarrivalseries[startdate:enddate]
	# peak1 = avgrainfall[avgrainfall == avgrainfall.max()].index[0]
	# peak2 = r7[r7 == r7.max()].index[0]
	# delay = (peak2 - peak1).days
	# print(str(year)+' : Delay in peaks ',delay)
	excess = r7 - avgarrival
	#print(str(year)+' : Difference in overall quantity ',excess.sum())
	monthwiseexcess = excess.groupby(excess.index.month).mean()
	print(monthwiseexcess)
	print(str(year)+' : Anomalous average decrease in arrival monthwise')
	monthwiseexcess1 = monthwiseexcess[monthwiseexcess <= -400]
	print(monthwiseexcess1)
	print(str(year)+' : Anomalous average increase in arrival monthwise')
	monthwiseexcess2 = monthwiseexcess[monthwiseexcess >= 400]
	print(monthwiseexcess2)



from rainfallmonthly import rainfallmonthly
from rainfallmonthly import avgrainfallmonthly
from rainfallmonthly import avgrainfallexpected


def rainfallstatus(year):
	startdate = '01-01-'+str(year)
	enddate = '31-12-'+str(year)
	avgrainfall = avgrainfallexpected[startdate:enddate]
	r7 = rainfallmonthly[startdate:enddate]
	peak1 = avgrainfall[avgrainfall == avgrainfall.max()].index[0]
	peak2 = r7[r7 == r7.max()].index[0]
	delay = (peak2 - peak1).days
	print(str(year)+' : Delay in peaks ',delay)
	excess = r7 - avgrainfall
	print(str(year)+' : Difference in overall quantity ',excess.sum())
	monthwiseexcess = excess.groupby(excess.index.month).mean()
	print(str(year)+' : Anomalous average difference in quantity monthwise')
	monthwiseexcess = monthwiseexcess[abs(monthwiseexcess)>=30]
	print(monthwiseexcess)