from datetime import datetime
import pandas as pd
import numpy as np
import scipy
from constants import CONSTANTS
import matplotlib
import matplotlib.pyplot as plt
import math

# from rainfallmonthly import rainfallmonthly
# from averagemandi import specificarrivalseries
# from averagemandi import mandipriceseries
# from averageretail import retailpriceseries


matplotlib.rcParams.update({'font.size': 22})
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

def obtain_Series(name,col):
  mandiseries=[]
  mcode = dict_mandiname_mandicode[name][0]
  series = CreateMandiSeries(mcode,wholeSalePA)
  arrival = series[col]   
  arrival = arrival.replace(0.0, np.NaN, regex=True)
  arrival = arrival.interpolate(method='pchip')
  arrival = RemoveNaNFront(arrival)
  mandiseries.append(arrival)
  mandiDF = pd.DataFrame()
  for i in range(0, len(mandiseries)):
    mandiDF[i] = mandiseries[i]

  meanseries = mandiDF.mean(axis=1)
  meanseries = meanseries.replace(0.0, np.NaN, regex=True)
  meanseries = meanseries.interpolate(method='pchip')
  specificarrivalseries = RemoveNaNFront(meanseries)
  return specificarrivalseries







def delay_corr(datax,datay,min,max,namex,namey):
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
  # print len(delay_corr_values_early_kharif)
  return (delays,delay_corr_values_early_kharif,delay_corr_values_kharif,delay_corr_values_rabi)
  # plt.plot(delays,delay_corr_values_early_kharif,color='r', label='Early Kharif')
  # plt.plot(delays,delay_corr_values_kharif, color='g', label = 'Kharif')
  # plt.plot(delays,delay_corr_values_rabi, color='b', label='Rabi')
  # plt.xlabel( namey+' is ahead of '+namex+' by number of months -----------> \n '+namex+' follows '+namey)
  # plt.ylabel('Shifted Correlations')
  # plt.legend(loc='best')
  # plt.title('Shifted Correlations - '+namex+' vs '+namey)
  # plt.show()

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

source_mandi = ["Gondal","Solapur","Pimpalgaon","Lasalgaon","Ahmednagar","Rahuri"]
terminal_mandi = ["Bangalore","Azadpur","Mumbai"]

def source_terminal():
  namex="Source Mandi Prices"
  namey = "Terminal Mandi Prices"
  final_corr_early = []
  final_corr_kharif = []
  final_corr_rabi = []
  for temp in range(-90,91):
    final_corr_early.append(0)
    final_corr_kharif.append(0)
    final_corr_rabi.append(0)
  final_corr_early = pd.Series(final_corr_early)
  final_corr_kharif = pd.Series(final_corr_kharif)
  final_corr_rabi = pd.Series(final_corr_rabi)
  delays=[]
  for x in source_mandi:
    for y in terminal_mandi:
      source_arr_x = obtain_Series(x,7)       # 7 is for mandi prices and 2 is for arrival
      terminal_arr_y = obtain_Series(y,7)
      delay,src_ter_corr_early,src_ter_corr_kharif,src_ter_corr_rabi = delay_corr(source_arr_x,terminal_arr_y,-90,90,"Source Mandi Prices","Terminal Mandi Prices")
      # print "line 158 - ",len(src_ter_corr_early)
      
      final_corr_early = final_corr_early + src_ter_corr_early
      final_corr_kharif = final_corr_kharif + src_ter_corr_kharif
      final_corr_rabi = final_corr_rabi + src_ter_corr_rabi
      delays=delay
      
  # final_corr_early = pd.Series(final_corr_early)
  # final_corr_kharif =pd.Series(final_corr_kharif)
  # final_corr_rabi = pd.Series(final_corr_rabi)
  
  final_corr_early = final_corr_early/(len(source_mandi)*len(terminal_mandi))
  final_corr_kharif = final_corr_kharif/(len(source_mandi)*len(terminal_mandi))
  final_corr_rabi= final_corr_rabi/(len(source_mandi)*len(terminal_mandi))
  
  print len(delays),"  ",len(final_corr_early),"  ",len(final_corr_kharif)
  plt.plot(delays,final_corr_early,color='r', label='Early Kharif')
  plt.plot(delays,final_corr_kharif, color='g', label = 'Kharif')
  plt.plot(delays,final_corr_rabi, color='b', label='Rabi')
  plt.xlabel('Correlation of '+namex +' with '+namey+' X days later ------>')
  plt.ylabel('Shifted Correlations for Mandi Prices')
  plt.legend(loc='best')
  plt.title('Shifted Correlations for Mandi Prices- '+namex+' vs '+namey)
  plt.show()

source_terminal()
