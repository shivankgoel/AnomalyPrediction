from datetime import datetime
import pandas as pd
import numpy as np
import scipy
from constants import CONSTANTS
import matplotlib.pyplot as plt
import math

START = CONSTANTS['STARTDATE']
END = CONSTANTS['ENDDATE']

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


def monthlycpiseries():
	cpi = pd.read_csv('data/original/global_commodity_price_crude_oil.csv')
	cpi['new_idx'] = cpi.apply(lambda row: datetime.strptime(row[4], '%m-%Y'), axis=1)
	cpi.index = cpi['new_idx']
	idx = pd.date_range(START, END)
	cpi = cpi.reindex(idx, fill_value=0)
	cpiseries = cpi['CPI']
	cpiseries = cpiseries.replace(0.0, np.NaN, regex=True)
	cpiseries = cpiseries.interpolate(method='linear',limit_direction='both')
	return cpiseries

cpimonthlyseries = monthlycpiseries()


avgcpiseries = cpimonthlyseries.groupby([cpimonthlyseries.index.month, cpimonthlyseries.index.day]).mean()
idx = pd.date_range(START, END)
data = [ (avgcpiseries[index.month][index.day]) for index in idx]
avgcpiseries = pd.Series(data, index=idx)
