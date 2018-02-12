from averagemandi import mandipriceseries
from averagemandi import mandiarrivalseries
from averageretail import retailpriceseries

def getminmax(series,start,end):
	series = series[start:end]
	return (series.min(),series.max())

def getavg(series,start,end):
	series = series[start:end]
	return series.mean()

def getavgmonth(series,year,month):
	if month<10:
		monthstr = '0'+str(month)
	else:
		monthstr = str(month)
	if month == 2:
		enddatestring = '28'
	elif month in [11,4,6,9]:
		enddatestring = '30'
	else:
		enddatestring = '31'
	startdate = str(year)+monthstr+'01'
	enddate = str(year)+monthstr+enddatestring
	return getavg(series,startdate,enddate)

def getavgyear(series,startyear,endyear):
	for year in range(startyear,endyear+1):
		print year
		for month in range(1,13):
			avg = getavgmonth(series,year,month)
			print avg

def getminmaxyear(series,startyear,endyear):
	for year in range(startyear,endyear+1):
		(min,max) = getminmax(series,str(year)+'-01-01',str(year)+'-12-31')
		print year,min,max

#getminmaxyear(mandipriceseries,2006,2014)

#getminmaxyear(retailpriceseries,2006,2014)

getavgyear(mandiarrivalseries,2006,2014)
