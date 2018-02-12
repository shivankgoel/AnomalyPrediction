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
END = CONSTANTS['ENDDATEOLD']


def load_wholesale_data():
  WP = 7
  WA = 2
  START = CONSTANTS['STARTDATE']
  END = CONSTANTS['ENDDATE']
  wholeSalePA = pd.read_csv(CONSTANTS['ORIGINALMANDI'], header=None)
  wholeSalePA = wholeSalePA[wholeSalePA[WA] > 0]
  wholeSalePA = wholeSalePA[wholeSalePA[WP] > 0]
  wholeSalePA = wholeSalePA[np.isfinite(wholeSalePA[WA])]
  wholeSalePA = wholeSalePA[np.isfinite(wholeSalePA[WP])]
  wholeSalePA = wholeSalePA[wholeSalePA[0] >= START]
  wholeSalePA = wholeSalePA[wholeSalePA[0] <= END]
  wholeSalePA = wholeSalePA.drop_duplicates(subset=[0, 1], keep='last')
  return wholeSalePA

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


# WP = 7
# WA = 2
wholeSalePA = load_wholesale_data()

def getmandi(mandiname,price):
  if price:
    switch = 7
  else:
    switch = 2 
  mcode = dict_mandiname_mandicode[mandiname][0]
  series = CreateMandiSeries(mcode,wholeSalePA)
  arrival = series[switch]   
  arrival = arrival.replace(0.0, np.NaN, regex=True)
  #arrival = arrival.interpolate(method='pchip',limit_direction='both')
  arrival = arrival.interpolate(method='pchip')
  arrival = RemoveNaNFront(arrival)
  return arrival


def give_df_imagenames(isprice,imagenames):
  mandiseries = []
  for imagename in imagenames:
    imagename = imagename.replace('.','_')
    [statename,centrename,mandiname,_] = imagename.split('_')
    arrival = getmandi(mandiname,isprice)
    mandiseries.append(arrival)

  mandiDF = pd.DataFrame()
  for i in range(0, len(mandiseries)):
    mandiDF[i] = mandiseries[i]
  #print(mandiDF[mandiDF<0])
  return mandiDF


def give_average_of_df(mandiDF):
  meanseries = mandiDF.mean(axis=1)
  meanseries = meanseries.replace(0.0, np.NaN, regex=True)
  meanseries = meanseries.interpolate(method='pchip')
  mandiarrivalseries = RemoveNaNFront(meanseries)
  return mandiarrivalseries

mandiDF = give_df_imagenames(False,imagenames)
mandiarrivalseries = give_average_of_df(mandiDF)

mandiDF = give_df_imagenames(True,imagenames)
mandipriceseries = give_average_of_df(mandiDF)

mandiDF = give_df_imagenames(False,['Delhi_DELHI_Azadpur.png'])
specificarrivalseries = give_average_of_df(mandiDF)


mandiDF = give_df_imagenames(True,['Delhi_DELHI_Azadpur.png'])
specificpriceseries = give_average_of_df(mandiDF)


def give_avg_series(mandiarrivalseries):
  mandiarrivalexpected = mandiarrivalseries.rolling(window=30,center=True).mean()
  mandiarrivalexpected = mandiarrivalexpected.groupby([mandiarrivalseries.index.month, mandiarrivalseries.index.day]).mean()
  idx = pd.date_range(START, END)
  data = [ (mandiarrivalexpected[index.month][index.day]) for index in idx]
  expectedarrivalseries = pd.Series(data, index=idx)
  return expectedarrivalseries

expectedarrivalseries = give_avg_series(mandiarrivalseries)
expectedmandiprice = give_avg_series(mandipriceseries)
expectedspecificarrivalseries = give_avg_series(specificarrivalseries)

