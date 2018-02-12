from datetime import datetime
import pandas as pd
import numpy as np
from constants import CONSTANTS
import matplotlib.pyplot as plt
import matplotlib

anomalies = pd.read_csv(CONSTANTS['ANOMALIES_NEWSPAPER'], header=None, index_col=None)
'''
Note whereever idx is function argument it means which anomaly out of 1 to len(anomalies) 
'''


font = {'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 14}

matplotlib.rc('font', **font)

colors = [ '#e6194b', '#3cb44b' ,'#ffe119', '#0082c8', '#f58231', '#911eb4',  '#000080' , '#800000'  ,'#000000', '#900c3f' ]


def givePeriod(idx):
  '''
  Anomaly Time Period from the final Value
  '''
  start = datetime.strptime(anomalies[0][idx-1], '%d/%m/%Y')
  end = datetime.strptime(anomalies[1][idx-1], ' %d/%m/%Y')
  s = datetime.strftime(start, '%Y-%m-%d')
  e = datetime.strftime(end, '%Y-%m-%d')
  return (s, e)

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

def plotClassAvg(clss, lb, cl , series):
  '''
  This function will help you to plot the function
  '''
  series = Normalise(series)
  series = series.rolling(window=30,center=True).mean()
  avg = np.zeros(shape=(43,))
  count = 0
  for i in range(0, len(anomalies)):
    check = (clss in anomalies[2][i])
    if check:
      count += 1
      s,e = givePeriod(i+1)
      a = Normalise(np.array(series[s:e].tolist()))
      avg += a
  avg /= count
  plt.plot(avg, color=cl, label=lb, linewidth=2.0)

from averagemandi import mandipriceseries
from averageretail import retailpriceserieswhiten
from averageretail import retailpriceseries
from averagemandi import mandiarrivalseries


# retailpriceseries = retailpriceseries.rolling(window=3,center=True).mean()
# mandipriceseries = mandipriceseries.rolling(window=7,center=True).mean()
# mandiarrivalseries = mandiarrivalseries.rolling(window=7,center=True).mean()

plt.title('Transport Strike Signature')
# plotClassAvg('Weather' , '', colors[3],retailpriceserieswhiten)
plotClassAvg('Transport' , '', colors[3],retailpriceserieswhiten)
# plotClassAvg('Hoarding' , '', colors[3],retailpriceserieswhiten)
#plotClassAvg('Fuel' , '', colors[3],retailpriceseries)
# plotClassAvg('Inflation' , '', colors[3],retailpriceserieswhiten)
plt.legend()
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