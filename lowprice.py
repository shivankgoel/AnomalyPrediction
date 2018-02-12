from averagemandi import mandipriceseries
from averagemandi import mandiarrivalseries
from averagemandi import expectedarrivalseries
from averagemandi import expectedmandiprice
from averageretail import retailpriceseries
from loadmonthlyseries import cpimonthlyseries
from constants import CONSTANTS
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas as pd

from loadmonthlyseries import mpimonthlyseries
from loadmonthlyseries import mpionionmonthlyseries
from loadmonthlyseries import cpimonthlyseries 


def linear_reg(xseries,yseries,XLABEL,YLABEL,rnd=0):
	from sklearn import linear_model
	from sklearn.metrics import mean_squared_error, r2_score
	from sklearn.utils import shuffle
	import matplotlib.pyplot as plt
	import numpy as np
	import pandas as pd
	mp,rp = shuffle(xseries,yseries)
	train_size = (int)(0.80 * len(mp))
	train = mp[:train_size]
	train_labels = rp[:train_size]
	test = mp[train_size:]
	test_labels = rp[train_size:]
	train = train_labels.groupby([train.round(rnd)]).min()
	test = test_labels.groupby([test.round(rnd)]).min()
	regr = linear_model.LinearRegression()
	regr.fit(np.array(train.index).reshape(-1,1), train)
	predicted_labels = regr.predict(np.array(test.index).reshape(-1,1))
	print('Variance score: %.2f' % r2_score(test, predicted_labels ))
	print("Mean squared error: %.2f" % mean_squared_error(test, predicted_labels))
	plt.scatter(mp, rp ,  color='black')
	plt.plot(mp, regr.predict(mp.reshape(-1,1)) , color='red', linewidth=1)
	plt.plot(mp,cpimonthlyseries, color='blue', linewidth=1)
	plt.title(YLABEL+' VS '+XLABEL)
	plt.xlabel(XLABEL)
	plt.ylabel(YLABEL)
	plt.show()
	return regr



# y = mpionionmonthlyseries
# x = cpimonthlyseries
# x = x[CONSTANTS['STARTDATE']:CONSTANTS['ENDDATEOLD']]
# y = y[CONSTANTS['STARTDATE']:CONSTANTS['ENDDATEOLD']]
# linear_reg(x,y,'CPI','MPI',4)
# mp = mandipriceseries.resample('W').mean()
# rp = retailpriceseries.resample('W').mean()

pstart = '2013-03-01'
pend = '2014-03-01'
fstart = CONSTANTS['STARTDATE']
fend = CONSTANTS['ENDDATE']
from plotall import plotdoubleseries
from plotall import plotsingleseries
from plotall import plotsingleseriesylimit
#plotdoubleseries(mandipriceseries,mpimonthlyseries,'Time','Time','Mandi Price','MPI',fstart,fend)
#plotsingleseries(cpimonthlyseries - mpionionmonthlyseries,'CPI - MPI(onion)','Time','CPI - MPI',fstart,fend,True,True,6)

#plotsingleseries(mpionionmonthlyseries,'MPI(onion)','Time','MPI',fstart,fend,True,True,6)

def getdeviatedseries(series,YLABEL):
	fstart = CONSTANTS['STARTDATE']
	fend = CONSTANTS['ENDDATE']
	series = series[fstart:fend]
	x = range(1,len(series)+1)
	x = pd.Series(data=x,index=series.index)
	y = series
	regr = linear_reg(x,y,'Trend',YLABEL)
	seriesinflation =  pd.Series(data=regr.predict(x.reshape(-1,1)) ,  index = series.index)
	deviatedseries = series - seriesinflation
	return deviatedseries,seriesinflation

# deviatedseries = getdeviatedseries(mpionionmonthlyseries,'MPI(onion)')
# plotsingleseriesylimit(deviatedseries,'MPI(onion) Deviation','Time','MPI',-100,500,fstart,fend,True,True,6)

# deviatedseries = getdeviatedseries(mpimonthlyseries,'MPI')
# plotsingleseriesylimit(deviatedseries,'MPI Deviation','Time','MPI',-100,100,fstart,fend,True,True,6)

# deviatedseries = getdeviatedseries(cpimonthlyseries,'CPI')
# plotsingleseriesylimit(deviatedseries,'CPI Deviation','Time','CPI',-100,100,fstart,fend,True,True,6)

# deviatedcpiseries,a = getdeviatedseries(cpimonthlyseries,'CPI')
# deviatedmpionionseries,b = getdeviatedseries(mpionionmonthlyseries,'MPI(onion)')
# difftrend = a - b
# diff = deviatedmpionionseries - deviatedcpiseries
# deviated_diff = diff - difftrend
# deviated_diff2,_ = getdeviatedseries(diff,'Diff') 
# # plotsingleseries(diff,'Difference Between CPI and MPI(onion) after removing trends','Time','Diff',fstart,fend,False,True)
# # plotsingleseries(difftrend,'CPI Trend - MPI Trend','Time','Diff',fstart,fend,False,True)
# # plotsingleseries(deviated_diff,'CPI Trend - MPI Trend','Time','Diff',fstart,fend,True,True)
# plotsingleseries(deviated_diff2,'Difference Between MPI(onion) and CPI  after removing trends','Time','Diff',fstart,fend,True,True,6)


#xyz,_ = getdeviatedseries(retailpriceseries-mandipriceseries,'Retail - Mandi') 
#plotsingleseries(cpimonthlyseries,'Difference Retail and Mandi after removing trend','Time','Diff',fstart,fend,True,True,6)

pstart = '2016-04-01'
pend = '2017-02-01'

xyz,_ = getdeviatedseries(mandipriceseries,'Mandi price') 
plotsingleseries(xyz,'Mandi Price Deviation','Time','Mandi Price',fstart,fend,True,True)
# plotsingleseries(mandipriceseries,'Mandi Price Series','Time','Mandi Price',pstart,pend,True,True)