from datetime import datetime
import pandas as pd
import numpy as np
from constants import CONSTANTS
import matplotlib.pyplot as plt
import matplotlib
import sklearn.preprocessing as preprocessing
import math
from datetime import timedelta



# def give_average_series(start,end,mandiarrivalseries):
#   mandiarrivalexpected = mandiarrivalseries.rolling(window=30,center=True).mean()
#   #mandiarrivalexpected = mandiarrivalseries
#   type(mandiarrivalexpected)
#   mandiarrivalexpected = mandiarrivalexpected.groupby([mandiarrivalseries.index.month, mandiarrivalseries.index.day]).mean()
#   idx = pd.date_range(start, end)
#   data = [ (mandiarrivalexpected[index.month][index.day]) for index in idx]
#   expectedarrivalseries = pd.Series(data, index=idx)
#   return expectedarrivalseries

'''
Get Retail Price Series
'''
from averageretail import getcenter
retailpriceseriesmumbai = getcenter('MUMBAI')
retailpriceseriesdelhi = getcenter('DELHI')
retailpriceserieslucknow = getcenter('LUCKNOW')
retailpriceseriesbhub = getcenter('BHUBANESHWAR')
retailpriceseriespatna = getcenter('PATNA')
# [retailpriceseriesbhub,retailpriceseriesdelhi,retailpriceserieslucknow,retailpriceseriesmumbai,retailpriceseriespatna] = whiten_series_list([retailpriceseriesbhub,retailpriceseriesdelhi,retailpriceserieslucknow,retailpriceseriesmumbai,retailpriceseriespatna])
#[retailpriceseriesdelhi,retailpriceserieslucknow,retailpriceseriesmumbai] = whiten_series_list([retailpriceseriesdelhi,retailpriceserieslucknow,retailpriceseriesmumbai])

'''
Get Mandi Price Series
'''
from averagemandi import getmandi
mandipriceseriesdelhi = getmandi('Azadpur',True)
mandiarrivalseriesdelhi = getmandi('Azadpur',False)
mandipriceserieslucknow = getmandi('Devariya',True)
mandiarrivalserieslucknow = getmandi('Devariya',False)
from averagemandi import mandipriceseries
from averagemandi import mandiarrivalseries 
# [mandipriceseriesdelhi,mandipriceserieslucknow,mandipriceseriesmumbai] = whiten_series_list([mandipriceseriesdelhi,mandipriceserieslucknow,mandipriceseries])
# [mandiarrivalseriesdelhi,mandiarrivalserieslucknow,mandiarrivalseriesmumbai] = whiten_series_list([mandiarrivalseriesdelhi,mandiarrivalserieslucknow,mandiarrivalseries])


def get_anomalies(path):
  anomalies = pd.read_csv(path, header=None, index_col=None)
  anomalies[0] = [ datetime.strftime(datetime.strptime(date, '%d/%m/%Y'),'%Y-%m-%d') for date in anomalies[0]]
  anomalies[1] = [ datetime.strftime(datetime.strptime(date, ' %d/%m/%Y'),'%Y-%m-%d') for date in anomalies[1]]
  return anomalies

anomaliesmumbai = get_anomalies('data/anomaly/mumbai.csv')
anomaliesdelhi = get_anomalies('data/anomaly/delhi.csv')
anomalieslucknow = get_anomalies('data/anomaly/lucknow.csv')

# Labelling 
# Transport:  1
# Weather:  2
# Inflation:  3
# Fuel:   4
# Hoarding: 5

delhilabels = [2,4,1,3,1,2,2,2,3,4,1,2,2,1,4,2,5,5,2,2,3,1,5,4,2,5,5,5,3,5,3,5,2,2,5,2,2,5,5,5,2,5,5,5,2,2,2,3,1,5,1,2]
lucknowlabels = [2,1,1,2,2,2,5,4,3,1,5,5,5,3,2,2,5,5,4,3,4,5,4,2,5,5,5,5,2,2,3,2,2,5,3,2,5,2]
mumbailabels = [2,2,2,3,5,1,2,5,2,5,2,2,2,4,2,3,2,3,3,1,1,2,5,5,3,3,2,5,3,5,5,5,2,5,5,5,2,5,2,5,3,2,5,2,5,3,2,1,5,5,2,1,2,2,2,1,5,5,2]



font = {'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)

colors = [ '#e6194b', '#3cb44b' ,'#ffe119', '#0082c8', '#f58231', '#911eb4',  '#000080' , '#800000'  ,'#000000', '#900c3f' ]


# def givePeriod(idx):
#   '''
#   Anomaly Time Period from the final Value
#   '''
#   start = datetime.strptime(anomalies[0][idx-1], '%d/%m/%Y')
#   end = datetime.strptime(anomalies[1][idx-1], ' %d/%m/%Y')
#   s = datetime.strftime(start, '%Y-%m-%d')
#   e = datetime.strftime(end, '%Y-%m-%d')
#   return (s, e)

def Normalise(arr):
  '''
  Normalise each sample
  '''
  m = arr.mean()
  am = arr.min()
  aM = arr.max()
  arr -= m
  arr /= (aM - am)
  #arr /= arr.std()
  return arr

def plotIndividualAnomaly(idx, centre):
  s,e = givePeriod(anomalies, idx)
  a = Normalise(np.array(retailP[centre][s:e].tolist()))
  plt.plot(a, linewidth=2.0)
  plt.show()

def plotClassAvg(clss, lb, cl , series,anomalies):
  '''
  This function will help you to plot the function
  '''
  # series = Normalise(series)
  # series = series.rolling(window=3,center=True).mean()
  print type(series)
  avg = np.zeros(shape=(43,))
  count = 0
  for i in range(0, len(anomalies)):
    check = (clss in anomalies[2][i])
    if check:
      count += 1
      s = anomalies[0][i]
      e = datetime.strftime(datetime.strptime(anomalies[1][i],'%Y-%m-%d')+timedelta(days=0),'%Y-%m-%d')
      a = Normalise(np.array(series[s:e].tolist()))
      avg += a
  avg /= count
  # avg = avg.rolling(window = 10,center = True).mean()
  # plt.plot(avg, color=cl, label=lb, linewidth=1.0)
  avg *= count
  return avg, count

# from averagemandi import mandipriceseries
# from averageretail import retailpriceserieswhiten
# from averageretail import retailpriceseries
# from averagemandi import mandiarrivalseries


# retailpriceseries = retailpriceseries.rolling(window=3,center=True).mean()
# mandipriceseries = mandipriceseries.rolling(window=7,center=True).mean()
# mandiarrivalseries = mandiarrivalseries.rolling(window=7,center=True).mean()

plt.title('Hoarding and Weather Signatures')
# plotClassAvg('Weather' , '', colors[3],retailpriceserieswhiten)
delhi_plot, delhi_count = plotClassAvg('Weather' , 'Delhi', colors[3],retailpriceseriesdelhi,anomaliesdelhi)
mumbai_plot, mumbai_count = plotClassAvg('Weather' , 'Mumbai', colors[2],retailpriceseriesmumbai,anomaliesmumbai)
lucknow_plot, lucknow_count = plotClassAvg('Weather' , 'Lucknow', colors[1],retailpriceserieslucknow,anomalieslucknow)


total_avg = np.zeros(shape=(43,))
total_avg += delhi_plot
total_avg += mumbai_plot
total_avg += lucknow_plot
total_avg /= (delhi_count+mumbai_count+lucknow_count)
# total_avg = pd.Series(total_avg).rolling(window = 3,center = True).mean()
# total_avg = total_avg[2:]
plt.plot(total_avg, color='black', label='Average Weather Signature', linewidth=2.0)

# plt.title('Weather Signature')
# plotClassAvg('Weather' , '', colors[3],retailpriceserieswhiten)
delhi_plot, delhi_count = plotClassAvg('Hoarding' , 'Delhi', colors[3],retailpriceseriesdelhi,anomaliesdelhi)
mumbai_plot, mumbai_count = plotClassAvg('Hoarding' , 'Mumbai', colors[2],retailpriceseriesmumbai,anomaliesmumbai)
lucknow_plot, lucknow_count = plotClassAvg('Hoarding' , 'Lucknow', colors[1],retailpriceserieslucknow,anomalieslucknow)


total_avg = np.zeros(shape=(43,))
total_avg += delhi_plot
total_avg += mumbai_plot
total_avg += lucknow_plot
total_avg /= (delhi_count+mumbai_count+lucknow_count)
# total_avg = pd.Series(total_avg).rolling(window = 3,center = True).mean()

# total_avg = total_avg[2:]

plt.plot(total_avg, color='red', label='Average Hoarding Signature', linewidth=2.0)
# plotClassAvg('Hoarding' , '', colors[3],retailpriceserieswhiten)
#plotClassAvg('Fuel' , '', colors[3],retailpriceseries)
# plotClassAvg('Inflation' , '', colors[3],retailpriceserieswhiten)
plt.xlabel('Period Of Anomaly')
plt.ylabel('Normalised Retail Prices')
plt.legend(loc = 'best')
plt.show()


# plt.title('Mandi Price Signatures')
# plotClassAvg('Weather' , 'Weather', colors[2],mandipriceseries)
# plotClassAvg('Transport' , 'Tranport', colors[3],mandipriceseries)
# plotClassAvg('Hoarding' , 'Hoarding', colors[4],mandipriceseries)
# plotClassAvg('Fuel' , 'Fuel', colors[5],mandipriceseries)
# plotClassAvg('Inflation' , 'Inflation', colors[6],mandipriceseries)
# plt.legend()
# plt.show()

# plt.title('Mandi Arrival Signatures')
# plotClassAvg('Weather' , 'Weather', colors[2],mandiarrivalseries)
# plotClassAvg('Transport' , 'Tranport', colors[3],mandiarrivalseries)
# plotClassAvg('Hoarding' , 'Hoarding', colors[4],mandiarrivalseries)
# plotClassAvg('Fuel' , 'Fuel', colors[5],mandiarrivalseries)
# plotClassAvg('Inflation' , 'Inflation', colors[6],mandiarrivalseries)
# plt.legend()
# plt.show()


'''
def RMSError(clss,centre_idx):
  # Calculate the RMS Error 
  # i.e the avg L2 norm of differencebetween mean prices and actual prices
  # First calculate the average prices for that class
  avg = np.zeros(shape=(43,))
  count = 0
  for i in range(0, len(anomalies)):
    check = (clss in anomalies[2][i])
    if check:
      count += 1
      s,e = givePeriod(anomalies, i+1)
      a = Normalise(np.array(retailP[centre_idx][s:e].tolist()))
      avg += a
  avg /= count
  RMS = []
  for i in range(0, len(anomalies)):
    check = (clss in anomalies[2][i])
    if check:
      s,e = givePeriod(anomalies, i+1)
      b = Normalise(np.array(retailP[centre_idx][s:e].tolist()))
      RMS.append(np.sqrt(((avg - b) ** 2).mean(axis=None)))
  return (np.array(RMS)).mean()


#'#12efff','#eee111','#eee00f','#e00fff','#123456'
PATH = 'plots/signatures_retailprice_timeseries/'
centrenames = CONSTANTS['CENTRENAMES']
for i in range(0,len(centrenames)):
  plotClassAvg('Weather', centrenames[i] + ' Weather', '#12efff' , i , retailP)
  plotClassAvg('Transport', centrenames[i] + ' Transport', '#eee111' , i , retailP)
  plotClassAvg('Inflation', centrenames[i] + ' Inflation', '#abc222' , i , retailP)
  plotClassAvg('Hoarding', centrenames[i] + ' Hoarding', '#e00fff' , i , retailP)
  plotClassAvg('Fuel', centrenames[i] + ' Fuel', '#123456' , i , retailP)
  plt.title(' Different Anomalies For '+centrenames[i])
  plt.savefig(PATH+centrenames[i]+'.png')
  plt.close()


anomalynames = CONSTANTS['ANOMALYNAMES']
colors = ['#12efff','#eee111','#e00fff','#123456','#abc222','#2edf4f']
for i in range(0,len(anomalynames)):
  for j in range(0,len(centrenames)):
    plotClassAvg(anomalynames[i], anomalynames[i] + ' '+centrenames[j] , colors[j] , j , retailP) 
  plt.title(' Different Centres For '+anomalynames[i])
  plt.legend(loc='best')
  plt.savefig(PATH+anomalynames[i]+'.png')
  plt.close()
'''