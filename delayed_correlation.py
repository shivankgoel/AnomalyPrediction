from datetime import datetime
import pandas as pd
import numpy as np
import scipy
from constants import CONSTANTS
import matplotlib
import matplotlib.pyplot as plt
import math

from rainfallmonthly import rainfallmonthly
from averagemandi import mandiarrivalseries
from averagemandi import mandipriceseries
from averageretail import retailpriceseries


matplotlib.rcParams.update({'font.size': 22})


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

mean_arr_series = mandiarrivalseries
mean_arr_series = mean_arr_series.rolling(window=30).mean()
mean_mandi_price_series = mandipriceseries
mean_mandi_price_series = mean_mandi_price_series.rolling(window=30).mean()
mean_retail_price_series = retailpriceseries
mean_retail_price_series = mean_retail_price_series.rolling(window = 30).mean()

# mean_arr_series = RemoveNaNFront(mean_arr_series)

def delay_corr(datax,datay,min,max):
	delay_corr_values=[]
	delays = []
	for delay in range(min,max+1):
		curr_corr = datax.corr(datay.shift(delay*30))
		delay_corr_values.append(curr_corr)
		delays.append(delay)
	plt.plot(delays,delay_corr_values)
	plt.show()
# ------------------------------------------------------------------------------------
#
def delay_corr2(datax,datay,min,max,namex,namey):
	delay_corr_values_early_kharif=[]
	delay_corr_values_kharif=[]
	delay_corr_values_rabi =[]
	delays = []
	for delay in range(min,max+1):
		new_datay = datay.shift(-1*delay)
		curr_corr_early = 0
		for year in range(1,8):
			short_y = new_datay[year*365+285:365*(year+1)+365]
			short_x = datax[year*365+285:365*(year+1)+365]
			curr_corr_early = curr_corr_early + short_x.corr(short_y)
		# curr_corr = datax.corr(datay.shift(delay*30))
		curr_corr_early = curr_corr_early/7
		delay_corr_values_early_kharif.append(curr_corr_early)

		curr_corr_kharif = 0
		for year in range(0,8):
			short_y = new_datay[year*365+365:365*(year+1)+120]
			short_x = datax[year*365+365:365*(year+1)+120]
			curr_corr_kharif = curr_corr_kharif + short_x.corr(short_y)
		# curr_corr = datax.corr(datay.shift(delay*30))
		curr_corr_kharif = curr_corr_kharif/8
		delay_corr_values_kharif.append(curr_corr_kharif)

		curr_corr_rabi = 0
		for year in range(1,9):
			short_y = new_datay[year*365+100:365*(year)+240]
			short_x = datax[year*365+100:365*(year)+240]
			curr_corr_rabi = curr_corr_rabi + short_x.corr(short_y)
		# curr_corr = datax.corr(datay.shift(delay*30))
		curr_corr_rabi = curr_corr_rabi/8
		delay_corr_values_rabi.append(curr_corr_rabi)
		delays.append(delay)
	plt.plot(delays,delay_corr_values_early_kharif,color='r', label='Early Kharif')
	plt.plot(delays,delay_corr_values_kharif, color='g', label = 'Kharif')
	plt.plot(delays,delay_corr_values_rabi, color='b', label='Rabi')
	plt.xlabel('<-----------'+namex+' follows '+namey+'\n'+namey+' follows '+namex+'------------>\n'+'Correlation of '+namex +' with '+namey+' X days later ------>')
	plt.ylabel('Shifted Correlations')
	plt.legend(loc='best')
	plt.title('Shifted Correlations - '+namex+' vs '+namey)
	plt.show()


def delay_corr2_rainfall(datax,datay,min,max,namex,namey):
	delay_corr_values_early_kharif=[]
	delay_corr_values_kharif=[]
	delay_corr_values_rabi =[]
	delays = []
	for delay in range(min,max+1):
		new_datay = datay.shift(-1*delay)
		# plt.plot(new_datay,label=datay)
		# plt.plot(datax,label=datax)
		# plt.show()
		curr_corr_early = 0
		for year in range(1,8):
			short_y = new_datay[year*12+9:12*(year+1)+12]
			short_x = datax[year*12+9:12*(year+1)+12]
			curr_corr_early = curr_corr_early + short_x.corr(short_y)
			if(delay == 4):
				print short_x.corr(short_y)
				plt.plot(short_x,label=namex)
				plt.plot(short_y,label=namey)
				plt.show()

		curr_corr_early = curr_corr_early/7
		delay_corr_values_early_kharif.append(curr_corr_early)

		curr_corr_kharif = 0
		for year in range(0,8):
			short_y = new_datay[year*12+12:12*(year+1)+4]
			short_x = datax[year*12+12:12*(year+1)+4]
			curr_corr_kharif = curr_corr_kharif + short_x.corr(short_y)
		curr_corr_kharif = curr_corr_kharif/8
		delay_corr_values_kharif.append(curr_corr_kharif)

		curr_corr_rabi = 0
		for year in range(1,9):
			short_y = new_datay[year*12+3:12*(year)+8]
			short_x = datax[year*12+3:12*(year)+8]
			curr_corr_rabi = curr_corr_rabi + short_x.corr(short_y)
			
		curr_corr_rabi = curr_corr_rabi/8
		delay_corr_values_rabi.append(curr_corr_rabi)
		delays.append(delay)
	plt.plot(delays,delay_corr_values_early_kharif,color='r', label='Early Kharif')
	plt.plot(delays,delay_corr_values_kharif, color='g', label = 'Kharif')
	plt.plot(delays,delay_corr_values_rabi, color='b', label='Rabi')
	plt.xlabel('<-----------'+namex+' follows '+namey+'\n'+namey+' follows '+namex+'------------>\n'+'Correlation of '+namex +' with '+namey+' X days later ------>')
	plt.ylabel('Shifted Correlations')
	plt.legend(loc='best')
	plt.title('Shifted Correlations - '+namex+' vs '+namey)
	plt.show()


def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')



# delay_corr2(mean_mandi_price_series,mean_arr_series,-120,120,"Mandi Price","Arrivals")
# delay_corr2(mean_retail_price_series,mean_mandi_price_series,-90,90,"Retail Price","Mandi Price")
# delay_corr2(mean_retail_price_series,mean_arr_series,-120,120,"Retail Price","Arrivals")
# delay_corr2(mean_arr_series,rainfallmonthly,-6,6)


#scaling all the data to have the data monthly not daily
# print rainfallmonthly

original_rainfall = rainfallmonthly[0:len(rainfallmonthly):30]
# plt.plot(original_rainfall,color='r',label='Rainfall')

new_arrival_series = mean_arr_series[30:len(mean_arr_series):30]
# plt.plot(new_arrival_series,color='g',label='Arrival')
new_mandi_price_series = mean_mandi_price_series[30:len(mean_mandi_price_series):30]
new_retail_price_series = mean_retail_price_series[30:len(mean_retail_price_series):30]
# print len(mean_arr_series[15:len(mean_arr_series):30])
# plt.show()
# delay_corr2_rainfall(new_retail_price_series,original_rainfall,-6,6,"Retail Prices","Rainfall")
delay_corr2_rainfall(original_rainfall,new_arrival_series,-1,6,"Rainfall","Arrival")
# delay_corr2_rainfall(new_mandi_price_series,original_rainfall,-6,6,"Mandi Price","Rainfall")