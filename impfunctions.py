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
