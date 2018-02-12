from datetime import datetime
import pandas as pd
import numpy as np
import scipy
from constants import CONSTANTS
import matplotlib.pyplot as plt
import math

def monthlyseries(path,m_y_idx,colname):
	START = CONSTANTS['STARTDATE']
	END = CONSTANTS['ENDDATEOLD']
	oil = pd.read_csv(path)
	oil['newcol'] = oil.apply(lambda row: datetime.strptime(row[m_y_idx], '%m-%Y'), axis=1)
	oil.index = oil['newcol']
	idx = pd.date_range(START, END)
	oil = oil.reindex(idx, fill_value=0)
	oilseries = oil[colname]
	oilseries = oilseries.replace(0.0, np.NaN, regex=True)
	oilseries = oilseries.interpolate(method='pchip',limit_direction='both')
	return oilseries

def giveavg(avgrainfallmonthly):
	avgrainfallexpected = avgrainfallmonthly.groupby([avgrainfallmonthly.index.month, avgrainfallmonthly.index.day]).mean()
	idx = pd.date_range(START, END)
	data = [ (avgrainfallexpected[index.month][index.day]) for index in idx]
	avgrainfallexpected = pd.Series(data, index=idx)
	return avgrainfallexpected

#oilmonthlyseries = monthlyseries('data/original/crude_oil_india_monthly.csv',2,'Indian-Basket')
#cpimonthlyseries = monthlyseries('data/original/global_commodity_price_crude_oil.csv',4,'CPI')
mpionionmonthlyseries = monthlyseries('data/original/wpi_data.csv',3,'Onion')
mpimonthlyseries = monthlyseries('data/original/wpi_data.csv',3,'ALL COMMODITIES')
cpimonthlyseries = monthlyseries('data/original/cpi_data.csv',3,'General Index')