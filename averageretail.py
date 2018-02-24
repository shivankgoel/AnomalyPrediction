
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

top_10_states = [ 'Maharashtra', 'Karnataka' , 'Madhya Pradesh', 'Bihar', 'Gujarat', 
'Rajasthan', 'Haryana' , 'Andhra Pradesh' , 'Telangana' , 'Uttar Pradesh']

relevant_centres_id = []

for state in top_10_states:
  statecode = dict_statename_statecode[state]
  centreids = dict_statecode_centreid[statecode[0]]
  relevant_centres_id = relevant_centres_id + centreids



RP = 2
START = CONSTANTS['STARTDATE']
END = CONSTANTS['ENDDATEOLD']
CENTREID = 1


def rwhiten(series):
  '''
  Whitening Function
  Formula is
    W[x x.T] = E(D^(-1/2))E.T
  Here x: is the observed series
  Read here more:
  https://www.cs.helsinki.fi/u/ahyvarin/papers/NN00new.pdf
  '''
  EigenValues, EigenVectors = np.linalg.eig(series.cov())
  D = [[0.0 for i in range(0, len(EigenValues))] for j in range(0, len(EigenValues))]
  for i in range(0, len(EigenValues)):
    D[i][i] = EigenValues[i]
  DInverse = np.linalg.matrix_power(D, -1)
  DInverseSqRoot = scipy.linalg.sqrtm(D)
  V = np.dot(np.dot(EigenVectors, DInverseSqRoot), EigenVectors.T)
  series = series.apply(lambda row: np.dot(V, row.T).T, axis=1)
  return series


def load_retail_data():
  RP = 2
  CENTREID = 1
  retailP = pd.read_csv(CONSTANTS['ORIGINALRETAIL'], header=None)
  retailP = retailP[retailP[RP] != 0]
  retailP = retailP[np.isfinite(retailP[RP])]
  retailP = retailP[retailP[0] >= START]
  retailP = retailP[retailP[0] <= END]
  retailP = retailP.drop_duplicates(subset=[0, 1], keep='last')
  retailP = retailP[retailP[0] >= START]
  retailP = retailP[retailP[0] <= END]
  return retailP

retailP = load_retail_data()

def CreateCentreSeries(Centre, RetailPandas):
  rc = RetailPandas[RetailPandas[1] == Centre]
  rc.groupby(0, group_keys=False).mean()
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

'''
centreSeries = []
for cid in relevant_centres_id :
  if(np.sum(retailP[1]==cid) > 0):
    tempseries = (CreateCentreSeries(cid, retailP))[RP]
    tempseries = tempseries.replace(0.0, np.NaN, regex=True)
    if(np.sum(tempseries != 0)>1):
      tempseries = tempseries.interpolate(method='pchip')
      if(np.sum(tempseries != np.NaN)>0):
        tempseries = RemoveNaNFront(tempseries)
        centreSeries.append(tempseries)
'''

from os import listdir
imagenames = [f for f in listdir('plots/bigmandis10')]

def getcenter(centrename):
  code = dict_centrename_centreid[centrename][0]
  series = CreateCentreSeries(code,retailP)
  price = series[RP]   
  price = price.replace(0.0, np.NaN, regex=True)
  #price = price.interpolate(method='pchip',limit_direction='both')
  price = price.interpolate(method='pchip')
  price = RemoveNaNFront(price)
  return price

def give_df_imagenames_center(imagenames):
  centreSeries = []
  RP=2
  for imagename in imagenames:
    if len(imagename.split('_'))>1:
      imagename = imagename.replace('.','_')
      print imagename
      [statename,centrename,mandiname,_] = imagename.split('_')
      price = getcenter(centrename)
      centreSeries.append(price)

  centreDF = pd.DataFrame()
  for i in range(0, len(centreSeries)):
    centreDF[i] = centreSeries[i]
  
  return centreDF


def give_average_of_df(mandiDF):
  meanseries = mandiDF.mean(axis=1)
  meanseries = meanseries.replace(0.0, np.NaN, regex=True)
  meanseries = meanseries.interpolate(method='pchip',)
  mandiarrivalseries = RemoveNaNFront(meanseries)
  return mandiarrivalseries



centreDF = give_df_imagenames_center(imagenames)
retailpriceseries = give_average_of_df(centreDF)

centreDF1 = give_df_imagenames_center(['dummy_MUMBAI_dummy.png'])
specificretailprice = give_average_of_df(centreDF1)

# centreDFwhite = rwhiten(give_df_imagenames_center(['dummy_MUMBAI_dummy.png','dummy_MUMBAI_dummy.png']))
# retailpriceserieswhiten = give_average_of_df(centreDFwhite)

retailpriceexpected = retailpriceseries.rolling(window=30,center=True).mean()
retailpriceexpected = retailpriceexpected.groupby([retailpriceexpected.index.month, retailpriceexpected.index.day]).mean()
idx = pd.date_range(START, END)
data = [ (retailpriceexpected[index.month][index.day]) for index in idx]
expectedretailprice = pd.Series(data, index=idx)





