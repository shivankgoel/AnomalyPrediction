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
dict_centrename_centreid = centre_info.groupby('centrename')['centreid'].apply(list).to_dict()
dict_centreid_statecode = centre_info.groupby('centreid')['statecode'].apply(list).to_dict()
dict_statecode_centreid = centre_info.groupby('statecode')['centreid'].apply(list).to_dict()

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
  #for centreid in centreids:
  #  print(state,dict_centreid_centrename[centreid])

'''
Maharashtra ['MUMBAI']
Maharashtra ['NAGPUR']
Karnataka ['BENGALURU']
Karnataka ['DHARWAD']
Karnataka ['MANGALORE']
Karnataka ['MYSORE']
Madhya Pradesh ['BHOPAL']
Madhya Pradesh ['GWALIOR']
Madhya Pradesh ['INDORE']
Madhya Pradesh ['JABALPUR']
Madhya Pradesh ['REWA']
Madhya Pradesh ['SAGAR']
Bihar ['BHAGALPUR']
Bihar ['PATNA']
Bihar ['PURNIA']
Gujarat ['AHMEDABAD']
Gujarat ['RAJKOT']
Gujarat ['BHUJ']
Gujarat ['SURAT']
Rajasthan ['JAIPUR']
Rajasthan ['JODHPUR']
Rajasthan ['KOTA']
Haryana ['GURGAON']
Haryana ['HISAR']
Haryana ['KARNAL']
Haryana ['PANCHKULA']
Andhra Pradesh ['HYDERABAD']
Andhra Pradesh ['VIJAYWADA']
Andhra Pradesh ['VISAKHAPATNAM']
Telangana ['WARANGAL']
Telangana ['KARIMNAGAR']
Telangana ['ADILABAD']
Telangana ['SURYAPET']
Telangana ['JADCHERLA']
Uttar Pradesh ['AGRA']
Uttar Pradesh ['KANPUR']
Uttar Pradesh ['LUCKNOW']
Uttar Pradesh ['VARANASI']
Uttar Pradesh ['JHANSI']
Uttar Pradesh ['MEERUT']
Uttar Pradesh ['ALLAHABAD']
Uttar Pradesh ['GORAKHPUR']
'''


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

WP = 7
WA = 2
wholeSalePA = pd.read_csv(CONSTANTS['ORIGINALMANDI'], header=None)
wholeSalePA = wholeSalePA[wholeSalePA[WA] != 0]
wholeSalePA = wholeSalePA[wholeSalePA[WP] != 0]
wholeSalePA = wholeSalePA[np.isfinite(wholeSalePA[WA])]
wholeSalePA = wholeSalePA[np.isfinite(wholeSalePA[WP])]
wholeSalePA = wholeSalePA[wholeSalePA[0] >= START]
wholeSalePA = wholeSalePA[wholeSalePA[0] <= END]
retailP = retailP[retailP[0] >= START]
retailP = retailP[retailP[0] <= END]
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
  #print(series)
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

def plot_series(series,figname,path):
  yaxis = list(series)
  xaxis = list(series.index)
  plt.plot(xaxis,yaxis)
  plt.xlabel('Time')
  plt.ylabel('Arrival')
  plt.title(figname)
  plt.savefig(path+figname+'.png')
  plt.close()

'''
for centreid in relevant_centres_id:
  if(np.sum(mandi_info['centreid'] == centreid) != 0 ):
    statename = dict_statecode_statename[dict_centreid_statecode[centreid][0]][0]
    mandicodelist = dict_centreid_mandicode[centreid]
    for mcode in mandicodelist:
      if(np.sum(wholeSalePA[1]==mcode) > 0):
        series = CreateMandiSeries(mcode,wholeSalePA)
        arrival = series[2]
        if(np.sum(arrival > 0.0) >= 260 ):
          mandiname = dict_mandicode_mandiname[mcode][0]
          print(mandiname)
          centrename = dict_centreid_centrename[centreid][0]
          #print(statename,centrename,mandiname)
          arrival = arrival.replace(0.0, np.NaN, regex=True)
          arrival = arrival.interpolate(method='pchip')
          arrival = RemoveNaNFront(arrival)
          if(arrival.mean()>=1000):
            plot_series(arrival,statename+'_'+centrename+'_'+mandiname,'plots/midtermbigmandis1000/')
'''


total_mandi_ids = dict_mandicode_statecode.keys()
for mandiid in total_mandi_ids:
  print(mandiid)
  if(np.sum(wholeSalePA[1]==mandiid) > 0):
    mandiname = dict_mandicode_mandiname[mandiid][0]
    centrename = dict_centreid_centrename[dict_mandicode_centreid[mandiid][0]][0]
    statename = dict_statecode_statename[dict_mandicode_statecode[mandiid][0]][0]
    if statename not in top_10_states:
      series = CreateMandiSeries(mandiid,wholeSalePA)
      arrival = series[2]   
      if(np.sum(arrival > 0.0) >= 260 ):
        #print(statename,centrename,mandiname)
        arrival = arrival.replace(0.0, np.NaN, regex=True)
        arrival = arrival.interpolate(method='pchip')
        arrival = RemoveNaNFront(arrival)
        if(arrival.mean()>=1000):
          plot_series(arrival,statename+'_'+centrename+'_'+mandiname,'plots/blabla/')


'''
from os import listdir
imagenames = [f for f in listdir('plots/bigmandis10')]


mylist = []
for imagename in imagenames:
  #print(imagename)
  imagename = imagename.replace('.','_')
  [statename,centrename,mandiname,_] = imagename.split('_')
  mcode = dict_mandiname_mandicode[mandiname][0]
  series = CreateMandiSeries(mcode,wholeSalePA)
  arrival = series[2]   
  arrival = arrival.replace(0.0, np.NaN, regex=True)
  arrival = arrival.interpolate(method='pchip')
  #multimean = arrival.resample("M",how='mean')
  #print(multimean)
  monthlymean = arrival.groupby(arrival.index.month).mean()
  mymean = np.round(monthlymean.mean(),decimals=2)
  mystd = np.round(math.sqrt(monthlymean.var()),decimals=2)
  print(statename,'     ',centrename,'      ',mandiname,'       ',mymean,'  ',mystd)
'''




'''

def DTWDistance(s1, s2):
  DTW={}
  for i in range(len(s1)):
      DTW[(i, -1)] = float('inf')
  for i in range(len(s2)):
      DTW[(-1, i)] = float('inf')
  DTW[(-1, -1)] = 0
  for i in range(len(s1)):
      for j in range(len(s2)):
          dist= (s1[i]-s2[j])**2
          DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
  return math.sqrt(DTW[len(s1)-1, len(s2)-1])


def DTWDistanceFast(s1, s2,w):
    DTW={}
    w = max(w, abs(len(s1)-len(s2)))
    for i in range(-1,len(s1)):
        for j in range(-1,len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0
    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
    return math.sqrt(DTW[len(s1)-1, len(s2)-1])


def ReturnYearSeries(inpseries,givenyear):
  s = str(givenyear)+'-01-01'
  e = str(givenyear)+'-12-31'
  s = datetime.strptime(s,'%Y-%m-%d')
  e = datetime.strptime(e,'%Y-%m-%d')
  series = inpseries[s:e]
  series.index = range(0,len(series))
  return series

dtw_dist = {}
for i in range(0,9):
  for j in range(i+1,9):
    series1 = ReturnYearSeries(meanseries,2006+i)
    series2 = ReturnYearSeries(meanseries,2006+j)
    dtw_dist[(i,j)] = DTWDistance(series1,series2)
    print(i,j)


import collections
dtw_dist = collections.OrderedDict(sorted(dtw_dist.items()))

import csv
w = csv.writer(open("plots/globalanomaly/yearDTW_Dist.txt", "w"))
for key, val in dtw_dist.items():
    w.writerow([key[0]+2006,key[1]+2006, np.round(val,decimals=1)])



def LB_Keogh(s1,s2,r):
  LB_sum=0
  for ind,i in enumerate(s1):
    lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
    upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
    if i>upper_bound:
      LB_sum=LB_sum+(i-upper_bound)**2
    elif i<lower_bound:
      LB_sum=LB_sum+(i-lower_bound)**2
  return math.sqrt(LB_sum)

import random
def k_means_clust(data,num_clust,num_iter,w=5):
  centroids=random.sample(data,num_clust)
  counter=0
  for n in range(0,num_iter):
    counter+=1
    assignments={}
    #assign data points to clusters
    for ind,i in enumerate(data):
      min_dist=float('inf')
      closest_clust=None
      for c_ind,j in enumerate(centroids):
        if LB_Keogh(i,j,5)<min_dist:
          cur_dist=DTWDistanceFast(i,j,w)
          if cur_dist<min_dist:
            min_dist=cur_dist
            closest_clust=c_ind
      if closest_clust in assignments:
        assignments[closest_clust].append(ind)
      else:
        assignments[closest_clust]=[]
    #recalculate centroids of clusters
    for key in assignments:
      clust_sum=0
      for k in assignments[key]:
        clust_sum=clust_sum+data[k]
      centroids[key]=[m/len(assignments[key]) for m in clust_sum]
  return centroids


data = []
for i in range(0,9):
  series = ReturnYearSeries(meanseries,2006+i)
  series = series[0:365]
  data.append(series)


clustercentres= k_means_clust(data,3,100)

'''