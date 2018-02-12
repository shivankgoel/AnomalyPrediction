from datetime import datetime
import pandas as pd
import numpy as np
import scipy
from constants import CONSTANTS
import matplotlib.pyplot as plt
import math

START = CONSTANTS['STARTDATE']
END = CONSTANTS['ENDDATE']

def getexportseries():
	export = pd.read_csv('data/original/export_quantity.csv')
	export[3] = export.apply(lambda row: datetime.strptime(row[2], '%m-%Y'), axis=1)
	export.index = export[3]
	idx = pd.date_range(START, END)
	export = export.reindex(idx, fill_value=0)
	exportseries = export['Quantity(MT)']
	exportseries = exportseries.replace(0.0, np.NaN, regex=True)
	exportseries = exportseries.interpolate(method='linear',limit_direction='both')
	return exportseries


exportseries = getexportseries()


def giveavg(avgrainfallmonthly):
	avgrainfallexpected = avgrainfallmonthly.groupby([avgrainfallmonthly.index.month, avgrainfallmonthly.index.day]).mean()
	idx = pd.date_range(START, END)
	data = [ (avgrainfallexpected[index.month][index.day]) for index in idx]
	avgrainfallexpected = pd.Series(data, index=idx)
	return avgrainfallexpected

expectedexportseries = giveavg(exportseries)
