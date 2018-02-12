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


def monthlyoilseries():
	oil = pd.read_csv('data/original/crude_oil_india_monthly.csv')
	oil[4] = oil.apply(lambda row: datetime.strptime(row[2], '%m-%Y'), axis=1)
	oil.index = oil[4]
	idx = pd.date_range(START, END)
	oil = oil.reindex(idx, fill_value=0)
	oilseries = oil['Indian-Basket']
	oilseries = oilseries.replace(0.0, np.NaN, regex=True)
	oilseries = oilseries.interpolate(method='linear',limit_direction='both')
	return oilseries

oilmonthlyseries = monthlyoilseries()


avgoilseries = oilmonthlyseries.groupby([oilmonthlyseries.index.month, oilmonthlyseries.index.day]).mean()
idx = pd.date_range(START, END)
data = [ (avgoilseries[index.month][index.day]) for index in idx]
avgoilseries = pd.Series(data, index=idx)
