#!/usr/bin/env python3
'''
The module achieves the following:
1. Reads the data from the respective CSVs into a DataFrame
2. Creates new DataFrames for the possible centres and mandis
3. Creates a 2D Matrix of the data for better usage
'''

import pandas as pd
from constants import CONSTANTS

print('reading_timeseries_data...')
'''
Read the whitened data
'''
retailDFP = pd.read_csv(CONSTANTS['RETAILPRICES'], header=None, index_col=0)
mandiDFP = pd.read_csv(CONSTANTS['MANDIPRICES'], header=None, index_col=0)
mandiDFA = pd.read_csv(CONSTANTS['MANDIARRIVALS'], header=None, index_col=0)


'''
Create the DataFrame of the Chosen Centres and Mandis
'''
retailP = pd.DataFrame()
for i in range(0, len(CONSTANTS['CENTRESID'])):
  retailP[i] = retailDFP[i+1]

mandiP, mandiA = [], []
idx = 1
for i in range(0, len(CONSTANTS['MANDIIDS'])):
  mandiP.append(pd.DataFrame())
  mandiA.append(pd.DataFrame())
  for j in range(0, len(CONSTANTS['MANDIIDS'][i])):
    mandiP[i][j] = mandiDFP[idx]
    mandiA[i][j] = mandiDFA[idx]
    idx += 1

'''
Make the data available as a matrix as well
'''
retailPM = retailP.as_matrix()
mandiAM, mandiPM = [], []
for i in range(0, len(CONSTANTS['MANDIIDS'])):
  mandiAM.append(mandiA[i].as_matrix())
  mandiPM.append(mandiP[i].as_matrix())