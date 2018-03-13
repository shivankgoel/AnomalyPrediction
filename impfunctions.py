'''
function to find the standard deviation across series
'''
def stddeviation(listofseries):
	import numpy as np
	import pandas as pd
	listofseries = [series.resample('W').mean() for series in listofseries]
	idx = listofseries[0].index
	l = len(listofseries[0])
	new = []
	for i in range(l):
		x = []
		for series in listofseries:
			x.append(series[i])
		new.append(np.array(x).std())
	return pd.Series(new,index=idx)

def resample(series):
	import numpy as np
	import pandas as pd
	series = series.resample('m').mean()
	return series

def giveharvestseries(series):
	harvestmonths = [11,12,1,2,3,4,5,6]
	indexby = [a or b for a, b in zip(series.index.month <=6, series.index.month>=11)]
	series = series[indexby]
	return series
