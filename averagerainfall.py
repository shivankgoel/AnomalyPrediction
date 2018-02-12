from datetime import datetime
import pandas as pd
import numpy as np
import scipy
from constants import CONSTANTS
import matplotlib.pyplot as plt
import math

START = CONSTANTS['STARTDATE']
END = CONSTANTS['ENDDATE']

def get_rain_series():
	rainfall = pd.read_csv('data/original/rainfall_Nashik.csv')
	rainfall[3] = rainfall.apply(lambda row: datetime.strptime(row[0], '%Y-%m-%d'), axis=1)
	rainfall.index = rainfall[3]
	idx = pd.date_range(START, END)
	rainfall = rainfall.reindex(idx, fill_value=0)
	series = rainfall['mean']
	return series


meanrainfallseries = get_rain_series()
