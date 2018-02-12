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
imagenames = [f for f in listdir('plots/mediummandis10')]


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
  arrival = RemoveNaNFront(arrival)
  #multimean = arrival.resample("M",how='mean')
  #print(multimean)
  monthlymean = arrival.groupby(arrival.index.month).mean()
  mymean = np.round(monthlymean.mean(),decimals=2)
  mystd = np.round(math.sqrt(monthlymean.var()),decimals=2)
  myratio = np.round(mystd/mymean, decimals=4) 
  print(statename,centrename,mandiname,mymean,mystd,myratio)



'''
Maharashtra MUMBAI    Pune              1167.3  486.08 0.4164
Maharashtra SURAT     Pimpalgaon        1721.8  372.06 0.2161
Maharashtra MUMBAI    Lasalgaon         1339.74 336.39 0.2511
Maharashtra MUMBAI    Rahuri            1038.81 260.84 0.2511
Maharashtra MUMBAI    Ahmednagar        1657.64 291.65 0.1759
Maharashtra MUMBAI    Newasa(Ghodegaon) 1208.61 248.39 0.2055
Karnataka   BENGALURU Bangalore         2761.76 1411.0 0.5109
Maharashtra DHARWAD   Solapur           1160.95 626.0  0.5392
Gujarat     RAJKOT    Gondal            1036.22 393.57 0.3798

Orissa      SAMBALPUR Bargarh           207.77 123.99 0.5968
West Bengal KOLKATA   Bara Bazar        271.27 20.21  0.0745
Delhi       DELHI     Shahdara          379.39 84.99  0.224
Orissa      CUTTACK   Kamakhyanagar     127.73 11.32  0.0886
Punjab      AMRITSAR  Amritsar(Mewa)    145.79 63.57  0.436
JandK       JAMMU     NarwalJammu(F&V)  109.07 16.51  0.1514
Uttrakhand  DEHRADUN  Dehradoon         300.6  105.1  0.3496
Delhi       DELHI     Keshopur          125.37 26.42  0.2107
Delhi       DELHI     Azadpur           907.38 110.56 0.1218
Punjab      LUDHIANA  FirozepurCity     134.36 75.26  0.5601
Punjab      LUDHIANA  Hoshiarpur        107.77 15.96  0.1481
Kerala      THRISSUR  Madhavapuram      100.43 9.84   0.098
WestBengal  KOLKATA   Katwa             132.4  28.57  0.2158
Orissa      BHUBAnesh Jatni             209.41 27.26  0.1302


Gujarat     SURAT     Pimpalgaon        341.07 107.93 0.3164
UttarP      AGRA      Agra              236.05 52.88 0.224
Rajasthan   JODHPUR   Nagour(FV)        119.89 17.01 0.1419
Gujarat     AHMEDABAD Vadodara          136.93 67.43 0.4924
Maharashtra MUMBAI    Rahata            695.94 187.64 0.2696
Karnataka   DHARWAD   Bijapur           108.86 35.49 0.326
Gujarat     SURAT     Satana            584.7 84.57 0.1446
Maharashtra MUMBAI    Junnar            260.28 86.17 0.3311
Maharashtra MUMBAI    Sinner            366.48 136.75 0.3731
UttarP      VARANASI  Gorakhpur         135.43 23.26 0.1717
Maharashtra MUMBAI    Maanachar         136.98 24.17 0.1764
Gujarat     SURAT     Dindori(Vani)     358.99 74.69 0.2081
Karnataka   DHARWAD   Belgaum           360.76 155.16 0.4301
Rajasthan   JAIPUR    Jaipur(F&V)       198.01 51.53 0.2602
Maharashtra MUMBAI    Kada              182.97 76.17 0.4163
Gujarat     AHMEDABAD AHMEDABAD         1260.67 31.06 0.0246
Telangana   KARIMNAGAR Bowenpally        288.44 91.85 0.3184
UttarP      KANPUR    Kanpur(Grain)     120.24 25.04 0.2083
Maharashtra MUMBAI    Lasalgaon(Niphad) 305.99 61.88 0.2022
Haryana     GURGAON   Gurgaon           102.08 33.02 0.3235
Maharashtra MUMBAI    Lonand            178.56 160.85 0.9008
MadhyaP     BHOPAL    Shujalpur         525.33 502.91 0.9573
Maharashtra MUMBAI    Palthan           147.08 46.06 0.3132
AndhraP     HYDERABAD Kurnool           285.17 217.33 0.7621
Telangana   KARIMNAGAR Hyderabad(F&V)   762.01 96.27 0.1263
Haryana     GURGAON   Alwar(FV)         483.02 825.82 1.7097
Maharashtra MUMBAI    Kopargaon         330.58 130.34 0.3943
Maharashtra MUMBAI    Kalvan            452.42 73.65 0.1628
Gujarat     SURAT     Manmad            744.25 302.44 0.4064
Maharashtra MUMBAI    Rahuri(Vambori)   156.07 55.77 0.3573
Karnataka   MYSORE    Mysore            168.89 63.35 0.3751
Rajasthan   JODHPUR   JODHPUR           146.29 25.47 0.1741
Gujarat     RAJKOT    Jamnagar          586.62 19.18 0.0327
Maharashtra MUMBAI    Akole             215.19 119.75 0.5565
Maharashtra NAGPUR    Nagpur            183.13 43.15 0.2356
MadhyaP     REWA      Allahabad         113.79 48.89 0.4297
Gujarat     RAJKOT    Rajkot            119.88 26.2 0.2186
Telangana   KARIMNAGAR Devala            375.24 79.52 0.2119
UttarP      LUCKNOW   Devariya          110.68 24.44 0.2208
Karnataka   DHARWAD   Hubli(Amaragol)  455.7 392.15 0.8605
Maharashtra MUMBAI    Parner            515.52 387.11 0.7509
Maharashtra MUMBAI    Sangamner         564.39 218.8 0.3877
Gujarat     SURAT     Malegaon          424.36 91.6 0.2159
Gujarat     SURAT     Chandvad          542.62 222.07 0.4093
Gujarat     SURAT     Bhavnagar         608.31 894.4 1.4703
Gujarat     AHMEDABAD Mehsana           147.7 30.8 0.2085
Gujarat     SURAT     Dhule             220.04 106.69 0.4849
Maharashtra MUMBAI    Mumbai            953.46 65.34 0.0685
Maharashtra MUMBAI    Shrirampur        559.74 341.88 0.6108
MadhyaP     INDORE    Indore(F&V)       604.85 174.98 0.2893
Karnataka   MYSORE    Hassan            350.82 168.14 0.4793
Gujarat     SURAT     Surat             207.35 26.66 0.1286
Telangana   KARIMNAGAR Sadasivpet        119.4 109.79 0.9195
Maharashtra MUMBAI    Yeola             921.3 440.35 0.478

'''
